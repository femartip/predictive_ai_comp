"""Load build_data.py's tables into one compact, integer-coded object (Data) used by every other module.

The bank has tens of millions of (subject, instance) pairs, so identifiers are replaced by integer codes once:
    P (DataFrame, one row per pair): bench, inst, item, subj, model (int32 codes); n, s, q (runs, sum, sum of squares)
    bench_names, inst_names, item_names, subj_names, model_names: code -> identifier
    inst_item, inst_bench: per instance code, its item and benchmark codes
    items (indexed by item code): text_hash and the I0 statistics;  adele (indexed by item code): 18 expected levels
    subjects (indexed by subject code): the subject's fields;  benchmarks (indexed by benchmark code): summary
A subject's model is its normalized_name (the unit of the model folds, D3, and of R2); a subject without one is its
own model ("subject:<id>").
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log  # noqa: E402
from config import ADELE_CODES, DATA, EVAL_BINARY_SHARE, EVAL_MIN_SUBJECTS, MIN_ITEMS  # noqa: E402


class Data:
    def __init__(self, root=DATA, benchmarks=None):
        root = Path(root)
        bt = pd.read_csv(root / "benchmarks.csv")
        bt = bt[bt["pairs"] > 0]                               # e.g. a benchmark whose responses are all ungraded
        if benchmarks:
            bt = bt[bt["benchmark"].isin(benchmarks)]
        self.bench_names = bt["benchmark"].to_numpy(dtype=object)
        bcode = {b: k for k, b in enumerate(self.bench_names)}

        # subjects and models: small tables, coded globally
        subjects = pd.read_parquet(root / "subjects.parquet")
        self.subj_names = subjects["subject_id"].to_numpy(dtype=object)
        scode = pd.Series(np.arange(len(subjects), dtype=np.int32), index=self.subj_names)
        model = subjects["normalized_name"].where(subjects["normalized_name"].notna(),
                                                  "subject:" + subjects["subject_id"])
        mcodes, self.model_names = pd.factorize(model)
        self.subj_model = mcodes.astype(np.int32)
        self.subjects = subjects.reset_index(drop=True)

        # pairs: instances and items are unique per benchmark (their ids start with "<benchmark>:"), so their global
        # code is an offset plus the file's own category code
        parts, inst_names, item_names, inst_item, inst_bench = [], [], [], [], []
        n_inst = n_item = 0
        for b in self.bench_names:
            p = pd.read_parquet(root / "pairs" / f"{b}.parquet")
            inst = p["instance_id"].astype("category")
            item = p["item_id"].astype("category")
            ic = inst.cat.codes.to_numpy().astype(np.int64)   # (pandas picks int8/int16 codes for small files)
            tc = item.cat.codes.to_numpy().astype(np.int64)
            # each instance belongs to one item: first item code seen for each instance code
            first = pd.Series(tc).groupby(ic).first()
            inst_names.append(inst.cat.categories.to_numpy(dtype=object))
            item_names.append(item.cat.categories.to_numpy(dtype=object))
            inst_item.append(first.reindex(range(len(inst.cat.categories))).to_numpy() + n_item)
            inst_bench.append(np.full(len(inst.cat.categories), bcode[b], np.int32))
            sc = scode.reindex(p["subject_id"].astype(str).to_numpy()).to_numpy()
            parts.append(pd.DataFrame({
                "bench": np.full(len(p), bcode[b], np.int32),
                "inst": (ic + n_inst).astype(np.int32),
                "item": (tc + n_item).astype(np.int32),
                "subj": sc.astype(np.int32),
                "n": p["n"].to_numpy(np.float32), "s": p["s"].to_numpy(np.float32),
                "q": p["q"].to_numpy(np.float32)}))
            n_inst += len(inst.cat.categories)
            n_item += len(item.cat.categories)
        self.P = pd.concat(parts, ignore_index=True)
        self.P["model"] = self.subj_model[self.P["subj"].to_numpy()]
        self.inst_names = np.concatenate(inst_names)
        self.item_names = np.concatenate(item_names)
        self.inst_item = np.concatenate(inst_item).astype(np.int32)
        self.inst_bench = np.concatenate(inst_bench)

        # item features, in item-code order
        items = pd.read_parquet(root / "items.parquet").drop_duplicates("item_id").set_index("item_id")
        self.items = items.reindex(self.item_names).reset_index(drop=True)
        adele = pd.read_parquet(root / "adele.parquet").set_index("item_id")
        self.adele = adele.reindex(self.item_names)[[f"{c}_exp" for c in ADELE_CODES]].reset_index(drop=True)
        self.benchmarks = bt.reset_index(drop=True)
        self.benchmarks["evaluable"] = self._evaluable()
        log(f"data: {len(self.bench_names)} benchmarks ({int(self.benchmarks['evaluable'].sum())} evaluable), "
            f"{len(self.P):,} pairs, {len(self.inst_names):,} instances, {len(self.subj_names):,} subjects, "
            f"{len(self.model_names):,} models")

    def _evaluable(self):
        """Protocol v3, section 2: binary outcomes and at least 3 subjects with at least 80 instances each."""
        per = self.P.groupby(["bench", "subj"]).size()
        ok_subj = (per >= MIN_ITEMS).groupby(level="bench").sum().reindex(range(len(self.bench_names)), fill_value=0)
        return ((self.benchmarks["binary_share"] >= EVAL_BINARY_SHARE).to_numpy()
                & (ok_subj.to_numpy() >= EVAL_MIN_SUBJECTS))

    @property
    def evaluable(self):
        return np.flatnonzero(self.benchmarks["evaluable"].to_numpy())

    def bench_mean(self, rows=None):
        """Mean score of each benchmark (code -> rate) over the given pair rows (default: all)."""
        P = self.P if rows is None else self.P.loc[rows]
        g = P.groupby("bench")[["s", "n"]].sum()
        return g["s"] / g["n"]


def load(benchmarks=None):
    return Data(benchmarks=benchmarks)


if __name__ == "__main__":       # quick check: python scripts/assessors_v3/data.py
    d = load()
    print(d.benchmarks[["benchmark", "pairs", "instances", "subjects", "binary_share", "evaluable"]].to_string())
    print(json.dumps({"pairs": len(d.P), "evaluable": int(len(d.evaluable))}))
