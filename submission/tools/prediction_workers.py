"""Bounded, stdlib-only prediction workers, embedded in both ingestion runtimes.

Each worker starts a fresh interpreter and imports model.py once. JSON travels
over a dedicated socket, independently of participant stdout/stderr. The parent
owns deadlines, input/result association, and cleanup; workers never see hidden
ground truth or organizer credentials. This is process isolation within the
existing submission container, not an additional security sandbox.
"""

import contextlib
import csv
import json
import os
from pathlib import Path
import selectors
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time


MAX_PREDICTION_CONCURRENCY = 16


def prediction_concurrency(submission_dir, limit=None):
    """Participants can reduce the organizer's worker limit, never raise it."""
    raw = os.environ.get("PREDICTION_CONCURRENCY", "4") if limit is None else limit
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        raise ValueError("PREDICTION_CONCURRENCY must be an integer from 1 to 16")
    if type(raw) not in (int, str) or not 1 <= limit <= MAX_PREDICTION_CONCURRENCY:
        raise ValueError("PREDICTION_CONCURRENCY must be an integer from 1 to 16")
    path = Path(submission_dir) / "prediction_config.json"
    if not path.exists():
        return limit
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("prediction_config.json must contain a JSON object") from exc
    return min(limit, validate_prediction_config(config))


def validate_prediction_config(config):
    if not isinstance(config, dict) or set(config) != {"concurrency"}:
        raise ValueError('prediction_config.json must contain only {"concurrency": <integer>}')
    requested = config["concurrency"]
    if type(requested) is not int or not 1 <= requested <= MAX_PREDICTION_CONCURRENCY:
        raise ValueError("prediction_config.json concurrency must be an integer from 1 to 16")
    return requested


class PredictionWorkerFailure(RuntimeError):
    def __init__(self, exit_code, message, index=None):
        self.exit_code = exit_code
        self.index = index
        self.message = message
        super().__init__(message)


def prediction_input_key(value):
    """Exact participant input identity, independent of dict insertion order.

    Revealed labels are fixed for the prediction phase. Response IDs and hidden
    outcomes do not enter this key; repeated outcomes share one probability.
    """
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def group_prediction_inputs(inputs, total, deadline):
    """Collect identical inputs before dispatch so duplicates cannot race."""
    unique, groups, index_by_key = [], [], {}
    row_count = 0
    for row_count, value in enumerate(inputs, 1):
        if time.monotonic() >= deadline:
            raise PredictionWorkerFailure(50, "Participant code exceeded the time limit")
        if row_count > total:
            raise PredictionWorkerFailure(40, "Prediction input count does not match test pairs")
        key = prediction_input_key(value)
        if key not in index_by_key:
            index_by_key[key] = len(unique)
            unique.append(value)
            groups.append([])
        groups[index_by_key[key]].append(row_count - 1)
    if row_count != total:
        raise PredictionWorkerFailure(40, "Prediction input count does not match test pairs")
    return unique, groups


# Fresh interpreters avoid forking participant libraries with active threads or
# initialized CUDA. Socket fd is the only extra descriptor inherited by a worker.
_WORKER_SOURCE = r'''
import copy, importlib.util, json, math, os, socket, sys, traceback

channel = socket.socket(fileno=int(sys.argv[1]))
reader = channel.makefile("rb")

def send(value):
    channel.sendall(json.dumps(value, allow_nan=False).encode("utf-8") + b"\n")

def failure(code, message):
    send({"error": code, "message": message[:6000]})

try:
    setup = json.loads(reader.readline())
    sys.path[:] = setup["sys_path"]
    path = setup["model_path"]
    spec = importlib.util.spec_from_file_location("participant_model", path)
    model = importlib.util.module_from_spec(spec)
    sys.modules["participant_model"] = model
    spec.loader.exec_module(model)
    sys.modules["model"] = model
    if not callable(getattr(model, "predict", None)):
        raise TypeError("model.py must define a callable predict(input, labeled=None)")
    labeled = setup["labeled"]
except BaseException as exc:
    failure(11, f"Failed to load model.py in prediction worker: {exc}\n{traceback.format_exc()}")
    raise SystemExit(1)

send({"ready": True})
for line in reader:
    task = json.loads(line)
    try:
        value = model.predict(task["input"], copy.deepcopy(labeled))
    except BaseException as exc:
        failure(40, f"predict() raised: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        break
    try:
        score = float(value)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"got {score}")
    except (TypeError, ValueError, OverflowError) as exc:
        failure(42, f"predict() must return a float in [0, 1]: {exc}")
        break
    send({"score": score})
'''


def parallel_predictions(model_path, inputs, labeled, *, concurrency, deadline,
                         per_call_timeout=0, worker_env=None):
    """Yield (input index, probability) as independent predictions finish.

    ``deadline`` is a monotonic deadline shared by startup and all workers. One
    request per worker is in flight. No prediction is retried automatically, and
    the first failure cancels remaining work. SIGALRM in participant code cannot
    disable the parent's per-call or overall deadlines.
    """
    iterator = enumerate(inputs)
    workers = []
    selector = selectors.DefaultSelector()
    sys_path = [str(Path(p or os.curdir).resolve()) for p in sys.path]
    model_path = str(Path(model_path).resolve())
    sys_path.insert(0, str(Path(model_path).parent))
    setup = json.dumps({"model_path": model_path, "sys_path": sys_path,
                        "labeled": labeled}).encode("utf-8") + b"\n"

    def queue(worker, payload):
        worker["out"] = payload
        selector.modify(worker["socket"], selectors.EVENT_READ | selectors.EVENT_WRITE, worker)

    def assign(worker):
        try:
            index, item = next(iterator)
        except StopIteration:
            worker["index"] = None
            return
        worker["index"] = index
        worker["started"] = time.monotonic()
        queue(worker, json.dumps({"input": item}).encode("utf-8") + b"\n")

    try:
        with tempfile.TemporaryDirectory(prefix="paiec_prediction_workers_") as work_dir:
            try:
                for number in range(concurrency):
                    if time.monotonic() >= deadline:
                        raise PredictionWorkerFailure(50, "Participant code exceeded the time limit")
                    parent, child = socket.socketpair()
                    child_dir = Path(work_dir) / str(number)
                    child_dir.mkdir()
                    env = os.environ.copy()
                    env.update(worker_env or {})
                    env.update(TMPDIR=str(child_dir), PAIEC_PREDICTION_WORKER_ID=str(number),
                               PYTHONUNBUFFERED="1")
                    # Label acquisition has already finished. A prediction
                    # worker only needs the revealed examples supplied above.
                    env.pop("ADAPTIVE_LABELS_TOKEN", None)
                    try:
                        proc = subprocess.Popen(
                            [sys.executable, "-u", "-c", _WORKER_SOURCE, str(child.fileno())],
                            pass_fds=(child.fileno(),), env=env, start_new_session=True,
                            cwd=str(Path(model_path).parent),
                        )
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    worker = {"process": proc, "socket": parent, "buffer": b"",
                              "out": setup, "ready": False, "index": None, "started": None}
                    workers.append(worker)
                    parent.setblocking(False)
                    selector.register(parent, selectors.EVENT_READ | selectors.EVENT_WRITE, worker)

                while any(not w["ready"] or w["index"] is not None for w in workers):
                    now = time.monotonic()
                    if now >= deadline:
                        raise PredictionWorkerFailure(50, "Participant code exceeded the time limit")
                    next_deadline = deadline
                    for w in workers:
                        if w["index"] is not None and per_call_timeout > 0:
                            call_deadline = w["started"] + per_call_timeout
                            if now >= call_deadline:
                                raise PredictionWorkerFailure(
                                    41, "predict() timed out: exceeded the per-call timeout", w["index"])
                            next_deadline = min(next_deadline, call_deadline)
                        if w["process"].poll() is not None and not select.select([w["socket"]], [], [], 0)[0]:
                            raise PredictionWorkerFailure(40, "Prediction worker exited unexpectedly", w["index"])
                    events = selector.select(max(0, min(0.1, next_deadline - now)))
                    for key, mask in events:
                        w = key.data
                        if mask & selectors.EVENT_WRITE and w["out"]:
                            try:
                                sent = w["socket"].send(w["out"])
                            except (BrokenPipeError, ConnectionResetError) as exc:
                                raise PredictionWorkerFailure(40, "Prediction worker exited unexpectedly", w["index"]) from exc
                            w["out"] = w["out"][sent:]
                            if not w["out"]:
                                selector.modify(w["socket"], selectors.EVENT_READ, w)
                        if not mask & selectors.EVENT_READ:
                            continue
                        try:
                            chunk = w["socket"].recv(65536)
                        except ConnectionResetError:
                            chunk = b""
                        if not chunk:
                            raise PredictionWorkerFailure(40, "Prediction worker exited unexpectedly", w["index"])
                        w["buffer"] += chunk
                        if len(w["buffer"]) > 65536:
                            raise PredictionWorkerFailure(40, "Invalid prediction worker response", w["index"])
                        while b"\n" in w["buffer"]:
                            line, w["buffer"] = w["buffer"].split(b"\n", 1)
                            try:
                                reply = json.loads(line)
                                if not isinstance(reply, dict):
                                    raise ValueError("expected object")
                            except (ValueError, UnicodeDecodeError) as exc:
                                raise PredictionWorkerFailure(40, "Invalid prediction worker response", w["index"]) from exc
                            if "error" in reply:
                                code = reply["error"]
                                if code not in (11, 40, 42):
                                    code = 40
                                raise PredictionWorkerFailure(code, str(reply.get("message", "Prediction failed")), w["index"])
                            if not w["ready"] and reply == {"ready": True}:
                                w["ready"] = True
                                assign(w)
                            elif w["ready"] and w["index"] is not None and set(reply) == {"score"}:
                                index = w["index"]
                                now = time.monotonic()
                                if now >= deadline:
                                    raise PredictionWorkerFailure(50, "Participant code exceeded the time limit")
                                if per_call_timeout > 0 and now - w["started"] >= per_call_timeout:
                                    raise PredictionWorkerFailure(41, "predict() timed out: exceeded the per-call timeout", index)
                                # Parent validates again in the ingestion writer.
                                yield index, reply["score"]
                                assign(w)
                            else:
                                raise PredictionWorkerFailure(40, "Invalid prediction worker response", w["index"])
            finally:
                # Kill the complete process groups, including participant
                # subprocesses, on success, failure, generator close, or timeout.
                for w in workers:
                    try:
                        os.killpg(w["process"].pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    w["socket"].close()
                for w in workers:
                    w["process"].wait(timeout=2)
    finally:
        selector.close()


def write_parallel_predictions(model_path, test_pairs, inputs, labeled, predictions_path,
                               *, concurrency, deadline, per_call_timeout, log, validate_score,
                               worker_env=None, stats=None):
    """Predict each unique input once and publish every original response row."""
    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    total = len(test_pairs)
    completed = next_row = 0
    pending = {}
    started = last_log = time.monotonic()
    groups = []
    try:
        unique, groups = group_prediction_inputs(inputs, total, deadline)
        call_count = len(unique)
        count = min(concurrency, call_count)
        log(f"Prediction concurrency: {count} isolated workers; "
            f"{call_count} prediction calls for {total} response rows")
        stream = parallel_predictions(model_path, unique, labeled, concurrency=count,
                                      deadline=deadline, per_call_timeout=per_call_timeout, worker_env=worker_env)
        with contextlib.closing(stream), tmp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["model_id", "item_id", "predicted_probability"])
            writer.writeheader()
            for index, value in stream:
                score = validate_score(value)
                for row_index in groups[index]:
                    pending[row_index] = score
                completed += 1
                while next_row in pending:
                    pair = test_pairs[next_row]
                    writer.writerow({"model_id": pair["model_id"], "item_id": pair["item_id"],
                                     "predicted_probability": pending.pop(next_row)})
                    next_row += 1
                now = time.monotonic()
                if now - last_log >= 15 or completed == call_count:
                    log(f"predict() progress: {completed}/{call_count} "
                        f"({completed / max(now - started, 1e-6):.1f} calls/s); "
                        f"{next_row}/{total} response rows written")
                    last_log = now
        if next_row != total:
            raise PredictionWorkerFailure(40, "Prediction workers returned incomplete results")
        os.replace(tmp_path, predictions_path)
        if stats is not None:
            stats.update(prediction_calls_count=completed, prediction_reuse_count=total - completed)
    except PredictionWorkerFailure as exc:
        if exc.index is not None:
            row_index = groups[exc.index][0]
            item_id = test_pairs[row_index]["item_id"]
            exc.index = row_index
            prefix = {42: "Invalid predict() output", 41: "predict() timed out"}.get(exc.exit_code, "predict() raised")
            exc.message = f"{prefix} on item_id={item_id}: {exc.message}"
            exc.args = (exc.message,)
            log(exc.message)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)
    return total
