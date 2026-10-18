"""
Decisions:
- The build keeps only instances graded resolved (1) or unresolved (0). Each run's results.json also lists
  `empty_error_patch_ids`: instances where the agent submitted no patch or the evaluation produced no test
  report. They are split with the run's released patches (raw/preds):
  - empty patch → kept as 0. The agent finished without changing the code: a model failure, as in
    SWE-rebench (empty-patch runs are kept there too) and as the upstream leaderboard counts it.
  - non-empty patch → dropped. results.json does not say whether the patch failed to apply (model) or the
    test environment failed (infrastructure), so these runs say nothing reliable about the item.
  - patch unknown (the prediction file is a Git LFS pointer or missing) → dropped, for the same reason.
  An empty-patch 0 is added only when the build has no response for that (subject, item): five Java items
  (elastic/logstash) already carry repeated build responses for some of these runs.
- `incomplete_ids` (evaluation did not finish) are dropped: infrastructure failures.
- Instances without a task definition in the build's items (a few, see the summary line) are dropped.
- Harness names are harmonised: MSWE-Agent → MSWE-agent; the iSWE-OpenModels submission, whose harness is
  blank, gets harness iSWE like the other iSWE submission.
- score = mean of the item's retained runs; n_runs = how many. Python items are SWE-bench Verified tasks
  run in the same experiments.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from common import DATA, benchmark_columns, huggingface_provenance, load_metadata, subject_columns, write

B = "multi_swebench"
REPO = "aims-foundations/measurement-db"
HARNESS = {"MSWE-Agent": "MSWE-agent"}
EVAL_TYPE = "unit_tests"  # FAIL_TO_PASS tests must pass and PASS_TO_PASS tests must keep passing


def unescape(name):
    """Upstream file names escape characters as _xHH_ (e.g. c_x2b__x2b_ = c++, _x28_Oct_x29_ = (Oct))."""
    return re.sub(r"_x([0-9a-f]{2})_", lambda m: chr(int(m.group(1), 16)), name)


def source_files(revision):
    """Per-run results and predictions from the same revision as the formatted tables."""
    api = HfApi()
    listing = {}
    for kind in ("results", "preds"):
        listing[kind] = sorted(e.path for e in api.list_repo_tree(
            REPO, path_in_repo=f"{B}/raw/{kind}", repo_type="dataset", revision=revision))
    return listing["results"], listing["preds"]


def cached_source_files(provenance):
    """Fetch the required files into the Hub cache at the run's pinned revision."""
    filenames = sorted(set(provenance["source_files"]))

    def download(filename):
        return Path(hf_hub_download(
            repo_id=REPO, filename=filename, repo_type="dataset",
            revision=provenance["source_revision"],
        ))

    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(zip(filenames, pool.map(download, filenames)))


def run_subject(subjects):
    """Map an upstream run name (<access date>_<harness>_<model>) to its subject_id."""
    key = [f"{h}_{d}" if isinstance(h, str) else d for h, d in zip(subjects.harness, subjects.display_name)]
    date = subjects.access_date.str.replace("-", "")

    def lookup(run):
        run_date, rest = run.split("_", 1)
        match = subjects[[k == rest for k in key]]
        if len(match) > 1:  # the same scaffold+model submitted twice: tell them apart by date
            match = match[date[match.index] == run_date]
        assert len(match) == 1, f"run {run!r} matches {len(match)} subjects"
        return match.subject_id.iloc[0]

    return lookup


def read_patches(path):
    """instance_id → submitted patch, or None if the file is not a readable prediction file."""
    try:
        with open(path, encoding="utf-8") as fh:
            return {(p := json.loads(line))["instance_id"]: p.get("model_patch") or "" for line in fh}
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        return None


def outcomes(results_files, files, subjects):
    """One row per (run, instance) of results.json, with its outcome group and whether the patch is empty."""
    lookup = run_subject(subjects)
    available = set(files)
    rows = []
    for name in results_files:
        language, run = unescape(PurePosixPath(name).stem).split("__", 1)
        res = json.loads(Path(files[name]).read_text(encoding="utf-8"))
        preds = name.replace("/results/", "/preds/").removesuffix(".json") + ".jsonl"
        patches = read_patches(files[preds]) if preds in available else None
        groups = {
            "resolved": res.get("resolved_ids", res.get("resolved", [])),
            "unresolved": res.get("unresolved_ids", []),
            "no_report": res.get("empty_error_patch_ids", []) + res.get("error_ids", []) + res.get("empty_patch_ids", []),
            "incomplete": res.get("incomplete_ids", []),
        }
        for group, ids in groups.items():
            for instance in ids:
                patch = None if patches is None else patches.get(instance)
                rows.append({"run": f"{language}__{run}", "subject_id": lookup(run), "raw_item_id": instance,
                             "group": group, "patch_empty": None if patch is None else not patch.strip()})
    return pd.DataFrame(rows)


def main():
    tables = {name: f"{B}/formatted_tables/{name}.parquet" for name in ("items", "subjects", "responses")}
    provenance = huggingface_provenance(REPO, tables.values())
    revision = provenance["source_revision"]
    results_files, preds_files = source_files(revision)
    provenance["source_files"] = sorted([*tables.values(), f"{B}/metadata.yaml", *results_files, *preds_files])
    print(f"Loading {B} from Hugging Face revision {revision}", flush=True)
    files = cached_source_files(provenance)
    meta = load_metadata(B, REPO, revision)

    items = pd.read_parquet(files[tables["items"]])
    subjects = pd.read_parquet(files[tables["subjects"]])
    resp = pd.read_parquet(files[tables["responses"]])

    # Outcomes the build leaves out: empty patches become 0, everything else without a grade is dropped.
    out = outcomes(results_files, files, subjects).merge(
        items[["item_id", "raw_item_id"]], on="raw_item_id", how="left")
    no_item = out.item_id.isna() & out.group.isin(["no_report"])
    no_report = out[(out.group == "no_report") & ~no_item]
    empty = no_report[no_report.patch_empty.eq(True)].drop_duplicates(["run", "raw_item_id"])
    graded = set(zip(resp.subject_id, resp.item_id))
    empty = empty[[pair not in graded for pair in zip(empty.subject_id, empty.item_id)]]
    added = empty[["subject_id", "item_id"]].assign(response=0.0)
    n_error_patch = no_report.patch_empty.eq(False).sum()
    n_unknown = no_report.patch_empty.isna().sum()
    n_incomplete = (out.group == "incomplete").sum()

    runs = pd.concat([resp[["subject_id", "item_id", "response"]], added], ignore_index=True)
    scores = runs.groupby(["subject_id", "item_id"]).response.agg(score="mean", n_runs="size").reset_index()

    # Items.
    items = pd.DataFrame({
        "build_item_id": items.item_id,
        "item_id": f"{B}:" + items.raw_item_id,
        "question": items.content,
        "ground_truth": items.grading_criterion.map(lambda g: json.loads(g)["reference_answer"]),
        "eval_type": EVAL_TYPE,
        "programming_language": items.item_features.str.extract(r"lang=([^;]+)", expand=False),
    })

    subjects = subjects.assign(harness=subjects.harness.replace(HARNESS))
    subjects.loc[subjects.harness.isna() & subjects.display_name.str.startswith("iSWE"), "harness"] = "iSWE"
    subj = subject_columns(subjects)

    df = (scores.rename(columns={"item_id": "build_item_id"}).merge(items, on="build_item_id")
          .merge(subj, on="subject_id", how="left")
          .assign(benchmark=B, tools=None, **benchmark_columns(meta)))
    assert len(df) == len(scores), "lost rows when joining items or subjects"

    path = write(df, B, provenance)
    print(f"{len(df):,} rows ({df.subject_id.nunique()} subjects × {df.item_id.nunique():,} items) from "
          f"{len(runs):,} runs → {path.relative_to(DATA.parent)}\n"
          f"added {len(added):,} empty-patch runs as 0; dropped {n_error_patch:,} runs whose non-empty patch has "
          f"no test report, {n_unknown:,} without a readable patch, {n_incomplete} incomplete and "
          f"{no_item.sum()} without a task definition")


if __name__ == "__main__":
    main()
