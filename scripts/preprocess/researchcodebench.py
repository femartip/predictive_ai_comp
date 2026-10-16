"""Convert ResearchCodeBench measurements from Hugging Face to assessor JSONL."""

import json

import pandas as pd
from datasets import load_dataset

from common import benchmark_columns, huggingface_provenance, load_metadata, subject_columns, write


BENCHMARK = "researchcodebench"
REPO = "aims-foundations/measurement-db"


def load_table(name: str, revision: str) -> pd.DataFrame:
    """Load a formatted benchmark table from one Hugging Face revision."""
    return load_dataset(
        "parquet",
        data_files=f"hf://datasets/{REPO}@{revision}/{BENCHMARK}/formatted_tables/{name}.parquet",
        split="train",
    ).to_pandas()


def reference_answer(criterion: object) -> object:
    """Extract the released reference implementation from an item criterion."""
    if criterion is None:
        return None
    if isinstance(criterion, str):
        criterion = json.loads(criterion)
    elif not isinstance(criterion, dict) and pd.isna(criterion):
        return None
    return criterion.get("reference_answer")


def main() -> None:
    table_names = ["items", "subjects", "responses"]
    source_tables = [f"{BENCHMARK}/formatted_tables/{name}.parquet" for name in table_names]
    provenance = huggingface_provenance(REPO, source_tables)
    revision = provenance["source_revision"]

    meta = load_metadata(BENCHMARK, REPO, revision)
    items = load_table("items", revision)
    subjects = load_table("subjects", revision)
    responses = load_table("responses", revision)

    item_columns = items[["item_id", "content", "grading_criterion"]].copy()
    item_columns["ground_truth"] = item_columns["grading_criterion"].map(reference_answer)

    # Collapse repeated trials to one observation per subject and item.
    output = (
        responses.groupby(["item_id", "subject_id"], as_index=False)
        .agg(score=("response", "mean"), n_runs=("response", "size"))
        .merge(
            item_columns[["item_id", "content", "ground_truth"]],
            on="item_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            subject_columns(subjects),
            on="subject_id",
            how="left",
            validate="many_to_one",
        )
    )

    output["item_id"] = f"{BENCHMARK}:" + output["item_id"]
    output["benchmark"] = BENCHMARK
    output["question"] = output["content"]
    output["eval_type"] = "external_judge"
    output["programming_language"] = "python"
    output["tools"] = None

    for column, value in benchmark_columns(meta).items():
        output[column] = value

    path = write(output, BENCHMARK, provenance)
    print(f"wrote {len(output):,} rows to {path}")


if __name__ == "__main__":
    main()
