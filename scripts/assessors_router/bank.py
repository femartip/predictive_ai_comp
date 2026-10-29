"""Load an assessor_data build (scripts/preprocess/assessor_data.py) into assessors_v3's Data object, so the v3
protocol, prior and test-time code run on it unchanged.

What changes against assessors_v3's own build (build_data.py), and why:
    pairs     one per (subject, item): the build's cells of a subject on an item (one per test condition) are pooled,
              weighted by their runs, as v3 pools conditions: n = sum of n_runs, s = sum of score * n_runs. q (sum of
              squared run scores, for the Brier score over runs) is exact for binary cells (q = s); for other cells the
              build keeps only the mean, so q = n * score^2 (Brier then differs by a constant, the same for every
              predictor; evaluable benchmarks are at least 90% binary).
    model     the subject's model_key (a model identity for every subject; model_key_source in the build), not
              normalized_name: strengths, tiers and the model folds cover every subject.
    subjects  only the fields predict() gets on the platform: normalized_name = model_key, so components.S1 reads the
              recovered name; model_name and thinking are left empty (not given at inference); and a model table
              (model_key -> provider, release_date, parameters) is passed to S1 as components.MODELS_CSV: the
              bank's provider and release date where given, else the curated model_provider / model_release_date.
    items     text_hash and the I0 statistics from the item's question (assessors_v3/build_data.py's functions); the
              18 ADeLe expected levels as they are in the build.
Instances are items (instance_id = item_id), as in v3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import settings as S
import components
from build_data import ITEM_STAT_COLS, item_stats, text_hash
from common import log
from config import ADELE_CODES
from data import Data

SUBJECT_COLS = ["subject_id", "model_name", "model_key", "provider", "release_date", "reasoning_effort", "thinking",
                "harness", "harness_version", "access_date", "model_provider", "model_release_date",
                "model_parameters"]
EXP = [f"{c}_exp" for c in ADELE_CODES]


def builds():
    return sorted(p for p in S.DATASET_ROOT.iterdir() if p.is_dir() and (p / "manifest.json").exists())


class BankData(Data):
    """assessors_v3's Data, filled from an assessor_data build (see the module docstring)."""

    def __init__(self, build, models_csv):
        build = Path(build)
        files = sorted(p for p in build.glob("*.parquet") if p.stem not in ("benchmarks", "models"))
        pairs, items, subjects, bench_rows = [], [], [], []
        for f in files:
            t = pq.read_table(f, columns=["item_id", "score", "n_runs", "score_type", "agentic",
                                          *SUBJECT_COLS]).to_pandas()
            w = t["n_runs"].astype(float)
            t = t.assign(n=w, s=t["score"] * w,
                         q=np.where(t["score_type"] == "binary", t["score"] * w, t["score"] ** 2 * w))
            pairs.append(t.groupby(["subject_id", "item_id"], sort=False, as_index=False)[["n", "s", "q"]].sum()
                         .assign(benchmark=f.stem))
            subjects.append(t.drop_duplicates("subject_id")[SUBJECT_COLS])
            binary = float(w[t["score_type"] == "binary"].sum() / w.sum())
            bench_rows.append({"benchmark": f.stem, "binary_share": binary, "agentic": bool(t["agentic"].iloc[0]),
                               "mean_score": float(t["s"].sum() / w.sum())})
            it = pq.read_table(f, columns=["item_id", "question", *EXP]).to_pandas().drop_duplicates("item_id")
            stats = pd.DataFrame([item_stats(q) for q in it["question"]], index=it.index)
            items.append(pd.concat([it[["item_id", *EXP]], stats], axis=1)
                         .assign(text_hash=it["question"].map(text_hash), benchmark=f.stem))
        P = pd.concat(pairs, ignore_index=True)
        items = pd.concat(items, ignore_index=True)
        subj = pd.concat(subjects, ignore_index=True).drop_duplicates("subject_id").reset_index(drop=True)

        self.bench_names = np.array([f.stem for f in files], dtype=object)
        bcode = {b: k for k, b in enumerate(self.bench_names)}
        self.item_names = items["item_id"].to_numpy(dtype=object)
        icode = pd.Series(np.arange(len(items), dtype=np.int32), index=self.item_names)
        self.inst_names, self.inst_item = self.item_names, np.arange(len(items), dtype=np.int32)
        self.inst_bench = items["benchmark"].map(bcode).to_numpy(np.int32)
        self.subj_names = subj["subject_id"].to_numpy(dtype=object)
        scode = pd.Series(np.arange(len(subj), dtype=np.int32), index=self.subj_names)
        mcodes, self.model_names = pd.factorize(subj["model_key"])
        self.subj_model = mcodes.astype(np.int32)

        item = icode.reindex(P["item_id"].to_numpy()).to_numpy()
        self.P = pd.DataFrame({"bench": P["benchmark"].map(bcode).to_numpy(np.int32), "inst": item, "item": item,
                               "subj": scode.reindex(P["subject_id"].to_numpy()).to_numpy(),
                               "n": P["n"].to_numpy(np.float32), "s": P["s"].to_numpy(np.float32),
                               "q": P["q"].to_numpy(np.float32)})
        self.P["model"] = self.subj_model[self.P["subj"].to_numpy()]
        self.items = items[["text_hash", *ITEM_STAT_COLS]].reset_index(drop=True)
        self.adele = items[EXP].reset_index(drop=True)
        # Only what predict() gets on the platform (its subject fields, normalized_name as model_key): the bank's
        # model_name and thinking are not given there, so S1 (components.system_table) never sees them.
        self.subjects = pd.DataFrame({
            "subject_id": subj["subject_id"], "model_name": None, "normalized_name": subj["model_key"],
            "reasoning_effort": subj["reasoning_effort"], "harness": subj["harness"],
            "harness_version": subj["harness_version"], "access_date": subj["access_date"], "thinking": None})

        # model table for S1: the bank's provider / release date where a subject of the model has them, else curated
        first = lambda v: v.dropna().iloc[0] if v.notna().any() else None   # noqa: E731
        m = subj.groupby("model_key").agg(provider=("provider", first), release_date=("release_date", first),
                                          model_provider=("model_provider", first),
                                          model_release_date=("model_release_date", first),
                                          parameters=("model_parameters", first))
        # no provider anywhere: "(unknown)", not S1's fallback (the first word of the name, right for the bank's names
        # such as "OpenAI GPT-4o", wrong for derived ones such as "xlmrbase")
        m["provider"] = m["provider"].fillna(m["model_provider"]).fillna("(unknown)")
        m["release_date"] = m["release_date"].fillna(m["model_release_date"])
        Path(models_csv).parent.mkdir(parents=True, exist_ok=True)
        m[["provider", "release_date", "parameters"]].rename_axis("normalized_name").to_csv(models_csv)
        components.MODELS_CSV = Path(models_csv)

        bt = pd.DataFrame(bench_rows)
        cnt = self.P.groupby("bench").agg(pairs=("n", "size"), instances=("inst", "nunique"),
                                          subjects=("subj", "nunique"))
        self.benchmarks = pd.concat([bt, cnt.reindex(range(len(bt))).reset_index(drop=True)], axis=1)
        self.benchmarks["evaluable"] = self._evaluable()
        log(f"data {build.name}: {len(self.bench_names)} benchmarks ({int(self.benchmarks['evaluable'].sum())} "
            f"evaluable), {len(self.P):,} pairs, {len(self.item_names):,} items, {len(self.subj_names):,} subjects, "
            f"{len(self.model_names):,} models")
