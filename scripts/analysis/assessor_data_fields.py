"""Empty fields of an assessor_data build (scripts/preprocess/assessor_data.py), with a focus on normalized_name.

    poetry run python scripts/analysis/assessor_data_fields.py [BUILD_DIR]     (default: the newest build)

Writes results/analysis/assessor_data_fields/<build id>/:
    fields.csv              share of filled values per column, over cells (rows) and over subjects
    benchmarks.csv          per benchmark: cells, subjects, share of cells and subjects with a normalized_name
    model_coverage.csv      among named cells and named models: share with each model_* attribute
    models_without_epoch.csv  named models Epoch did not match (other sources may still fill them), by cells
    unnamed_subjects.csv    every subject without normalized_name: its model_name, cells, and the normalized_name
                            of a named subject whose model_name matches it after normalisation (empty if none)
    fields.png              filled share of the subject and model columns, by cells and by subjects
Matching for unnamed_subjects.csv (a measurement, not a fix of the data): lower case; keep the part after the last
" / " (benchmark prefixes such as "MMBench V1.1 / ...") and after the last "/" or "__" (organisations); drop a trailing
date; keep letters, digits and dots. Named subjects contribute their model_name and their normalized_name without its
first word (the provider); a key that leads to two different names is not used.
"""

import re
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
BUILDS = ROOT / "data" / "datasets" / "assessor_data"
OUT = ROOT / "results" / "analysis" / "assessor_data_fields"
LONG = {"question", "ground_truth", "response_ids", "trials", "raw_scores"}      # not needed here, and large
SUBJECT = ["normalized_name", "provider", "release_date", "access_date", "harness", "harness_version",
           "reasoning_effort", "thinking", "subject_features_extra"]
# assessors_v3's palette: cells, subjects
COLOURS = {"cells": "#2a78d6", "subjects": "#eb6834"}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#6b6a64", "#e4e3de"


def name_key(name):
    n = str(name).lower().split(" / ")[-1]
    n = re.split(r"/|__", n)[-1]
    n = re.sub(r"[-_ ]?(20\d\d[-_]?\d\d[-_]?\d\d|\d{4})$", "", n)
    return re.sub(r"[^a-z0-9.]", "", n)


def load(build):
    files = sorted(p for p in build.glob("*.parquet") if p.stem not in ("benchmarks", "models"))
    cols = [c for c in pq.read_schema(files[0]).names if c not in LONG]
    return pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)


def unnamed_subjects(df):
    subj = df.groupby(["benchmark", "subject_id"]).agg(model_name=("model_name", "first"),
                                                       normalized_name=("normalized_name", "first"),
                                                       cells=("model_name", "size")).reset_index()
    named = subj[subj["normalized_name"].notna()]
    keys = pd.concat([
        pd.DataFrame({"k": named["model_name"].map(name_key), "name": named["normalized_name"]}),
        pd.DataFrame({"k": named["normalized_name"].str.split(" ", n=1).str[-1].map(name_key),
                      "name": named["normalized_name"]})]).drop_duplicates()
    unique = keys.groupby("k")["name"].nunique()
    lut = keys[keys["k"].isin(unique[unique == 1].index)].drop_duplicates("k").set_index("k")["name"]
    un = subj[subj["normalized_name"].isna()].drop(columns="normalized_name")
    un["key"] = un["model_name"].map(name_key)
    un["matching_normalized_name"] = un["key"].map(lut)
    return un.sort_values("cells", ascending=False)


def plot(fields, path):
    t = fields.loc[[c for c in fields.index if c in SUBJECT or c.startswith("model_") and c != "model_name"]]
    t = t.sort_values("cells")
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(t) + 1.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(t))
    h = 0.38
    for off, col in ((h / 2, "cells"), (-h / 2, "subjects")):
        ax.barh([v + off for v in y], t[col], height=h * 0.9, color=COLOURS[col], label=f"share of {col}")
    ax.set_yticks(list(y), t.index, fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("Filled values of the subject and model columns", color=INK, fontsize=11, loc="left")
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK)
    fig.text(0.01, 0.005, "Every other column is filled on (almost) every row: see fields.csv.", color=MUTED,
             fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    build = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(p for p in BUILDS.iterdir() if p.is_dir())[-1]
    out = OUT / build.name
    out.mkdir(parents=True, exist_ok=True)
    df = load(build)
    subj = df.drop_duplicates(["benchmark", "subject_id"])
    fields = pd.DataFrame({"cells": df.notna().mean(), "subjects": subj.notna().mean()})
    fields.round(4).to_csv(out / "fields.csv", index_label="column")

    named = df["normalized_name"].notna()
    bench = df.assign(named=named).groupby("benchmark").agg(cells=("named", "size"), named_cells=("named", "mean"),
                                                            subjects=("subject_id", "nunique"))
    bench["named_subjects"] = subj.assign(named=subj["normalized_name"].notna()).groupby("benchmark")["named"].mean()
    bench.sort_values("cells", ascending=False).round(4).to_csv(out / "benchmarks.csv")

    mcols = [c for c in df.columns if c.startswith("model_") and c != "model_name"]
    n = df[named]
    pd.DataFrame({"named_cells": n[mcols].notna().mean(),
                  "named_models": n.drop_duplicates("normalized_name")[mcols].notna().mean()}) \
        .round(4).to_csv(out / "model_coverage.csv", index_label="column")
    models = pd.read_parquet(build / "models.parquet")
    cells = n["normalized_name"].value_counts()
    nomatch = models[(models["kind"] == "model") & models["epoch_model"].isna()] \
        .assign(cells=lambda t: t["normalized_name"].map(cells)).sort_values("cells", ascending=False)
    nomatch[["normalized_name", "provider", "release_date", "parameters", "context_window", "open_weights",
             "cells"]].to_csv(out / "models_without_epoch.csv", index=False)

    un = unnamed_subjects(df)
    un.to_csv(out / "unnamed_subjects.csv", index=False)
    plot(fields, out / "fields.png")

    rec = un["matching_normalized_name"].notna()
    print(f"{build.name}: {len(df):,} cells, {subj['subject_id'].nunique():,} subjects")
    print(f"normalized_name: {named.mean():.1%} of cells, {subj['normalized_name'].notna().mean():.1%} of subjects; "
          f"{int((bench['named_cells'] == 0).sum())} of {len(bench)} benchmarks have none "
          f"({int(bench.loc[bench['named_cells'] == 0, 'cells'].sum()):,} cells)")
    print(f"unnamed subjects whose model_name matches a named one: {int(rec.sum()):,} of {len(un):,} "
          f"({un.loc[rec, 'cells'].sum() / un['cells'].sum():.1%} of unnamed cells)")
    print(f"named models without an Epoch match: {len(nomatch)} ({int(nomatch['cells'].sum()):,} cells)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
