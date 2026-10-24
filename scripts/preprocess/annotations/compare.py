"""Check the vLLM annotations against reference annotations of the same items.

    python compare.py --parts parts_check --item-map data_check/item_map.csv --reference adele_measurement_db.parquet

The reference is a per-item table with <code>_p0 … <code>_p5 (e.g. predictive_ai_comp/annotations/adele/
adele_measurement_db.parquet, produced for the six benchmarks with the transformers script on A40s). Items are
matched on item_id. Reports, per rubric and overall: mean |Δ expected level|, the share of identical most likely
levels, the Pearson correlation of the expected levels and the largest probability difference.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", required=True)
    ap.add_argument("--item-map", required=True)
    ap.add_argument("--reference", required=True)
    a = ap.parse_args()
    long = pd.concat([pd.read_csv(f) for f in sorted(Path(a.parts).rglob("part-*.csv"))], ignore_index=True)
    imap = pd.read_csv(a.item_map)
    new = long.merge(imap[["item_id", "text_id"]].drop_duplicates("text_id"), on="text_id")
    ref = pd.read_parquet(a.reference).drop_duplicates("item_id").set_index("item_id")
    rows = []
    for code, g in new.groupby("rubric"):
        cols = [f"{code}_p{k}" for k in range(6)]
        if cols[0] not in ref:
            continue
        g = g[g["item_id"].isin(ref.index)]
        P_new = g[[f"p{k}" for k in range(6)]].to_numpy()
        P_ref = ref.loc[g["item_id"], cols].to_numpy()
        e_new, e_ref = P_new @ np.arange(6), P_ref @ np.arange(6)
        rows.append({"rubric": code, "items": len(g), "mean |Δ exp|": np.abs(e_new - e_ref).mean(),
                     "argmax agree": (P_new.argmax(1) == P_ref.argmax(1)).mean(),
                     "corr exp": np.corrcoef(e_new, e_ref)[0, 1] if len(g) > 2 else np.nan,
                     "max |Δ p|": np.abs(P_new - P_ref).max()})
    t = pd.DataFrame(rows).set_index("rubric")
    print(t.round(3).to_string())
    print(f"overall: mean |Δ exp| {t['mean |Δ exp|'].mean():.3f}, argmax agreement {t['argmax agree'].mean():.3f}, "
          f"median corr {t['corr exp'].median():.3f}")


if __name__ == "__main__":
    main()
