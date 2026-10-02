"""
Decisions:
- Drop runs that ended in an infrastructure error (API timeout, service unavailable, scaffold error,
  unknown). Keep step-limit, stuck-in-loop and empty-patch runs with their labels.
- Keep all items, including those that share a problem statement (different PRs and tests).
- score = mean of the item's retained runs; n_runs = how many.
"""

import json
import pandas as pd
from common import DATA, benchmark_columns, load_metadata, subject_columns, write

B = "swe_rebench"
DIR = DATA / B
INFRA_ERROR = r"Timeout|ServiceUnavailable|AttributeError|^unknown$"
EVAL_TYPE = {"swebench_harness": "unit_tests"}  # verifier spec kind → eval_type
PROGRAMMING_LANGUAGE = "python"  # SWE-rebench collects Python repositories only


def main():
    meta = load_metadata(B)

    # Responses, row-aligned with the raw trajectories that carry exit_status.
    resp = pd.read_parquet(DIR / "response.parquet")
    traj = pd.read_parquet(DIR / "raw" / "openhands_trajectories.parquet", columns=["resolved", "exit_status"])
    assert (resp.response.values == traj.resolved.values).all(), "responses and trajectories are not row-aligned"
    infra = traj.exit_status.str.contains(INFRA_ERROR, regex=True).values
    resp = resp[~infra]
    scores = (resp.groupby(["subject_id", "item_id"]).response.agg(score="mean", n_runs="size").reset_index())

    # Items.
    items = pd.read_parquet(DIR / "items.parquet")
    spec = items.verifier.map(lambda v: json.loads(json.loads(v)["spec"]))
    items = pd.DataFrame({
        "build_item_id": items.item_id,
        "item_id": f"{B}:" + items.raw_item_id,
        "question": items.content,
        "ground_truth": items.grading_criterion.map(lambda g: json.loads(g)["reference_answer"]),
        "eval_type": spec.map(lambda s: EVAL_TYPE[s["kind"]]),
        "programming_language": PROGRAMMING_LANGUAGE,
    })

    subjects = subject_columns(pd.read_parquet(DIR / "subjects.parquet"))

    df = (scores.rename(columns={"item_id": "build_item_id"}).merge(items, on="build_item_id")
          .merge(subjects, on="subject_id", how="left")
          .assign(benchmark=B, tools=None, **benchmark_columns(meta)))
    assert len(df) == len(scores), "lost rows when joining items or subjects"

    path = write(df, B)
    print(f"{len(df):,} rows ({df.subject_id.nunique()} subjects × {df.item_id.nunique():,} items) from "
          f"{len(resp):,} runs ({infra.sum()} infrastructure-error runs dropped) → {path.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
