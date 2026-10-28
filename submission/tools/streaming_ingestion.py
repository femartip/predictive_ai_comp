"""Stdlib streaming client, shipped in the bundle and embedded in Modal.

The trusted coordinator owns the cursor, purchased evidence, and commitments.
Evaluation workers are recreated with frozen evidence at each checkpoint.
"""

import contextlib
import copy
import csv
import hashlib
import inspect
import io
import json
import math
import numbers
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

PIPELINE_RUNTIME_VERSION = 1


class AcquisitionHook:
    """Select a call shape without executing/retrying participant code.

    One-input hooks keep numeric pool-ranking semantics. Hooks accepting any
    of the new named arguments (or all four positional arguments) use streaming
    boolean decisions. Defaults are supported, but never retrofitted by catching
    a TypeError from inside a participant's function.
    """
    def __init__(self, function):
        self.function = function
        self.mode, self.positional, self.keywords = "random", False, ()
        if function is None:
            return
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cannot inspect acquisition_function signature") from exc
        names = ("prediction", "labeled", "context")
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        keywords = tuple(name for name in names if has_kwargs or (
            name in signature.parameters and signature.parameters[name].kind != inspect.Parameter.POSITIONAL_ONLY))
        shapes = [(False, names), (True, ())]
        if keywords and keywords != names:
            shapes.append((False, keywords))
        for positional, supplied in shapes:
            try:
                if positional:
                    signature.bind(None, None, None, None)
                else:
                    signature.bind(None, **{name: None for name in supplied})
            except TypeError:
                continue
            self.mode, self.positional, self.keywords = "stream", positional, supplied
            return
        try:
            signature.bind(None)
        except TypeError as exc:
            raise ValueError("acquisition_function must accept input and optional prediction, labeled, context") from exc
        self.mode = "legacy_rank"

    @property
    def needs_prediction(self):
        return self.positional or "prediction" in self.keywords

    def __call__(self, input, prediction=None, labeled=None, context=None):
        if self.mode == "legacy_rank":
            value = self.function(copy.deepcopy(input))
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError("One-argument acquisition_function must return a finite numeric priority")
            score = float(value)
            if not math.isfinite(score):
                raise ValueError("One-argument acquisition_function must return a finite numeric priority")
            return score
        values = {"prediction": prediction, "labeled": labeled, "context": context}
        if self.positional:
            result = self.function(copy.deepcopy(input), prediction, copy.deepcopy(labeled), copy.deepcopy(context))
        else:
            result = self.function(copy.deepcopy(input), **{k: copy.deepcopy(values[k]) for k in self.keywords})
        if type(result) is not bool:
            raise ValueError("Streaming acquisition_function(input, prediction, labeled, context) must return a bool")
        return result


def stream_exchange(url, submission_id, token, deadline):
    if not url or not submission_id or not token:
        raise ValueError("Streaming evaluation requires the data service and submission token")
    endpoint = url.rstrip("/") + "/v1/submissions/" + urllib.parse.quote(str(submission_id), safe="") + "/stream"
    role = os.environ.get("PAIEC_STREAM_ROLE", "")
    if role:
        if role not in ("acquisition", "budget-0", "budget-1", "budget-3", "budget-7", "budget-15", "budget-31"):
            raise ValueError("Unknown streaming role")
        endpoint += "/pipeline/" + role

    def exchange(payload):
        data = json.dumps(payload, allow_nan=False).encode()
        for attempt in range(4):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Streaming evaluation exceeded the time limit")
            request = urllib.request.Request(endpoint, data=data, method="POST", headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + token})
            try:
                with urllib.request.urlopen(request, timeout=min(60, remaining)) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                # Do not print the response body, credentials, or service URL.
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise RuntimeError(f"Streaming coordinator returned HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise RuntimeError("Streaming coordinator could not be reached") from None
            time.sleep(min(2 ** attempt, max(0, deadline - time.monotonic())))
        raise RuntimeError("Streaming coordinator retry limit exceeded")
    return exchange


def run_streaming(*, config_path, predict, acquisition_function, model_path,
                  output_dir, concurrency, deadline, per_call_timeout,
                  acquisition_timeout, call_window, validate_score,
                  prediction_stream, group_inputs, log, exchange):
    config = json.loads(Path(config_path).read_text())
    if config.get("protocol") != "streaming_alc_v1":
        raise ValueError("Unsupported streaming evaluation protocol")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hook = AcquisitionHook(acquisition_function)
    stats = {"evaluation_protocol": config["protocol"], "prediction_concurrency": concurrency,
             "acquisition_prediction_calls": 0, "prediction_calls_count": 0,
             "prediction_reuse_count": 0, "labels_resolved": 0,
             "acquisition_mode": hook.mode, "acquisition_calls": 0}
    log(f"Acquisition interface: {hook.mode}")
    event = exchange({})
    last_progress = time.monotonic()
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Streaming evaluation exceeded the time limit")
        if event.get("protocol") != config["protocol"]:
            raise ValueError("Unexpected streaming coordinator protocol")
        kind = event["type"]
        if kind == "configure":
            event = exchange({"event_id": event["event_id"],
                              "acquisition_mode": "legacy_rank" if hook.mode == "legacy_rank" else "stream"})
            continue
        if kind in ("acquisition_finished", "evaluation_finished"):
            if kind == "acquisition_finished":
                stats["labels_resolved"] = event["labels_resolved"]
            (output_dir / "labeling_log.json").write_text(json.dumps(stats, indent=2))
            log("Streaming role complete")
            return stats
        if kind == "finished":
            path = output_dir / "predictions.csv"
            tmp = path.with_suffix(".csv.tmp")
            tmp.write_text(event["predictions_csv"], encoding="utf-8")
            os.replace(tmp, path)
            stats["predictions_count"] = sum(1 for _ in csv.DictReader(io.StringIO(event["predictions_csv"])))
            (output_dir / "labeling_log.json").write_text(json.dumps(stats, indent=2))
            log("Streaming evaluation complete: all six budget checkpoints committed")
            return stats
        labeled = event["labeled"]
        stats["labels_resolved"] = len(labeled)
        payload = {"event_id": event["event_id"]}
        if kind == "rank":
            if hook.mode != "legacy_rank":
                raise ValueError("Coordinator acquisition mode does not match the submission hook")
            with call_window(acquisition_timeout, "acquisition_function"):
                payload["score"] = hook(event["input"])
            stats["acquisition_calls"] += 1
        elif kind == "acquire":
            current_input, context = event["input"], event["context"]
            if hook.mode == "legacy_rank":
                raise ValueError("Coordinator acquisition mode does not match the submission hook")
            if hook.mode == "random":
                # Uniform sampling without replacement, reproducible across submissions.
                key = json.dumps([context, current_input], sort_keys=True, separators=(",", ":"))
                uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
                query = uniform < min(1, context["labels_remaining"] / context["items_remaining"])
            else:
                probability = None
                if hook.needs_prediction:
                    with call_window(per_call_timeout, "predict"):
                        probability = validate_score(predict(copy.deepcopy(current_input), copy.deepcopy(labeled)))
                    stats["acquisition_prediction_calls"] += 1
                with call_window(acquisition_timeout, "acquisition_function"):
                    query = hook(current_input, probability, labeled, context)
                stats["acquisition_calls"] += 1
            payload["query"] = query
        elif kind == "evaluate":
            targets = event["targets"]
            unique, groups = group_inputs((row["input"] for row in targets), len(targets), deadline)
            count = len(unique)
            probabilities = [None] * len(targets)
            log(f"Budget {event['budget']}: {count} evaluation calls, up to {concurrency} workers")
            stream = prediction_stream(model_path, unique, labeled, concurrency=min(concurrency, count),
                                       deadline=deadline, per_call_timeout=per_call_timeout)
            with contextlib.closing(stream):
                for index, value in stream:
                    probability = validate_score(value)
                    for row_index in groups[index]:
                        probabilities[row_index] = probability
                    stats["prediction_calls_count"] += 1
                    if time.monotonic() - last_progress >= 15:
                        log(f"Streaming progress: {stats['prediction_calls_count']} evaluation calls completed")
                        last_progress = time.monotonic()
            if any(value is None for value in probabilities):
                raise ValueError("Streaming evaluation returned incomplete predictions")
            stats["prediction_reuse_count"] += len(targets) - count
            payload["probabilities"] = probabilities
            if event["budget"] == 0 and hook.mode == "legacy_rank" and not os.environ.get("PAIEC_STREAM_ROLE"):
                # Mode is locked with the zero-label checkpoint, before any reveal.
                payload["acquisition_mode"] = "legacy_rank"
        else:
            raise ValueError("Unknown streaming coordinator event")
        event = exchange(payload)
        if time.monotonic() - last_progress >= 15:
            log(f"Streaming progress: {stats['labels_resolved']} labels acquired")
            last_progress = time.monotonic()
