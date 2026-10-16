"""
Decisions:
- Drop runs that ended in an infrastructure error (API timeout, service unavailable, scaffold error,
  unknown). Keep step-limit, stuck-in-loop and empty-patch runs with their labels.
- Keep all items, including those that share a problem statement (different PRs and tests).
- score = mean of the item's retained runs; n_runs = how many.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from common import DATA, benchmark_columns, huggingface_provenance, load_metadata, subject_columns, write

B = "swe_rebench"
REPO = "aims-foundations/measurement-db"
INFRA_ERROR = r"Timeout|ServiceUnavailable|AttributeError|^unknown$"
EVAL_TYPE = {"swebench_harness": "unit_tests"}  # verifier spec kind → eval_type
PROGRAMMING_LANGUAGE = "python"  # SWE-rebench collects Python repositories only


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


def main():
    tables = {name: f"{B}/formatted_tables/{name}.parquet" for name in ("items", "subjects", "responses")}
    trajectory_file = f"{B}/raw/openhands_trajectories.parquet"
    provenance = huggingface_provenance(REPO, tables.values())
    provenance["source_files"] = sorted([*tables.values(), f"{B}/metadata.yaml", trajectory_file])
    print(f"Loading {B} from Hugging Face revision {provenance['source_revision']}", flush=True)
    files = cached_source_files(provenance)
    meta = load_metadata(B, REPO, provenance["source_revision"])

    # Responses, row-aligned with the raw trajectories that carry exit_status.
    resp = pd.read_parquet(files[tables["responses"]])
    traj = pd.read_parquet(files[trajectory_file], columns=["instance_id", "resolved", "exit_status"])
    items = pd.read_parquet(files[tables["items"]])
    raw_ids = resp.item_id.map(items.set_index("item_id").raw_item_id)
    assert raw_ids.tolist() == traj.instance_id.tolist(), "responses and trajectories have different item order"
    assert (resp.response.values == traj.resolved.values).all(), "responses and trajectories are not row-aligned"
    infra = traj.exit_status.str.contains(INFRA_ERROR, regex=True).values
    resp = resp[~infra]
    scores = (resp.groupby(["subject_id", "item_id"]).response.agg(score="mean", n_runs="size").reset_index())

    # Items.
    spec = items.verifier.map(lambda v: json.loads(json.loads(v)["spec"]))
    items = pd.DataFrame({
        "build_item_id": items.item_id,
        "item_id": f"{B}:" + items.raw_item_id,
        "question": items.content,
        "ground_truth": items.grading_criterion.map(lambda g: json.loads(g)["reference_answer"]),
        "eval_type": spec.map(lambda s: EVAL_TYPE[s["kind"]]),
        "programming_language": PROGRAMMING_LANGUAGE,
    })

    subjects = subject_columns(pd.read_parquet(files[tables["subjects"]]))

    df = (scores.rename(columns={"item_id": "build_item_id"}).merge(items, on="build_item_id")
          .merge(subjects, on="subject_id", how="left")
          .assign(benchmark=B, tools=None, **benchmark_columns(meta)))
    assert len(df) == len(scores), "lost rows when joining items or subjects"

    path = write(df, B, provenance)
    print(f"{len(df):,} rows ({df.subject_id.nunique()} subjects × {df.item_id.nunique():,} items) from "
          f"{len(resp):,} runs ({infra.sum()} infrastructure-error runs dropped) → {path.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
