"""Write the final tables (adele_v1.parquet) of benchmarks whose scoring is complete, from their saved parts, without
loading any model (e.g. after a run that scored everything but could not write the table).

    python finalize.py --root <annotation root> (--benchmarks a b ... | --benchmark-file list.txt | --all)
"""

import argparse
from pathlib import Path

import annotate_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--benchmarks", nargs="+")
    who.add_argument("--benchmark-file")
    who.add_argument("--all", action="store_true")
    ap.add_argument("--rubrics", nargs="+", default=[str(C.RUBRICS_V1)])
    a = ap.parse_args()
    C.require_parquet()
    codes = list(C.load_rubrics(a.rubrics))
    for b in C.benchmarks_from(a):
        bdir = Path(a.root) / b
        if (bdir / "LOCK").exists():
            print(f"{b}: LOCK present (still running, or crashed): skipped")
            continue
        missing = C.finalize(bdir, codes)
        print(f"{b}: {'finished -> ' + str(bdir / C.FINAL) if not missing else f'{missing} (text, rubric) pairs still missing'}")


if __name__ == "__main__":
    main()
