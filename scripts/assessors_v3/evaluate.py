"""Evaluate one prior (with T0, T1, T2 and the baselines) under evaluation protocol v3.

    python scripts/assessors_v3/evaluate.py --prior R1+R2+S1+I0+I2 [--regimes primary secondary] [--reps 5]
        [--tune] [--device auto|cpu|gpu|cuda] [--n-jobs 8] [--quick] [--name NAME]

For every repetition (D4) and regime (section 4), and every cell of that regime:
  1. the cell's evaluation draws (seed 0) and tuning draws (seed 100), spread evenly over the cells (D5);
  2. the prior, trained on the cell's training pairs (prior.py), predicts every instance of the drawn pairs;
     the logits are cached (RESULTS/<name>/cache), so a rerun or a new test-time model does not refit it;
  3. with --tune, the test-time scales are chosen on the tuning draws of all cells (D7, TAU_GRID), else config.TAU;
  4. every predictor is scored on the evaluation draws (simulate.py).
Outputs in RESULTS/<name>/: records_rep<r>_<regime>.parquet (one row per draw, pair, budget, predictor),
importance.csv (prior feature gains per cell), tau.json, checks.json (the draw checks of section 5), meta.json.
Everything is resumable: finished (repetition, regime) records and cached cells are kept. Summaries and plots:
report.py.
"""

import argparse
import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
from common import LowMemory, check_start_ram, guard_memory, log, n_jobs_default, write_json  # noqa: E402
from components import parse  # noqa: E402
from data import Data  # noqa: E402
from prior import fit_prior, lgb_device  # noqa: E402
from protocol import Repetition, sibling_groups  # noqa: E402
from simulate import PREDICTORS, draw_scores, run_draws  # noqa: E402


def cell_name(cell):
    return f"{cell[0]}x{'all' if cell[1] is None else cell[1]}"


def build_cell(d, R, cell, spec, n_eval, n_tune, cache_dir, n_jobs, device):
    """Draws of one cell and the arrays the simulator needs (prior logits per pair). Cached on disk."""
    path = cache_dir / f"rep{R.rep}_{cell_name(cell)}_d{n_eval}_t{n_tune}.pkl"   # the draws depend on both counts
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    draws = R.draws(cell, n_eval, C.EVAL_SEED)
    tdraws = R.draws(cell, n_tune, C.TUNE_SEED)
    groups = sorted({g for x in draws + tdraws for g in x})
    bundle = {"cell": cell, "draws": draws, "tune_draws": tdraws, "pairs": {}, "start": None, "importance": None}
    if groups:
        info = [R.pair(b, s) for b, s in groups]
        test_rows = np.concatenate([np.concatenate([p["acq_rows"], p["eval_rows"]]) for p in info])
        pr = fit_prior(d, R, cell, spec, test_rows, n_jobs=n_jobs, device=device, seed=R.rep)
        z = pd.Series(pr["z"], index=test_rows)
        z = z[~z.index.duplicated()]
        P = d.P
        for (b, s), p in zip(groups, info):
            a, e = p["acq_rows"], p["eval_rows"]
            bundle["pairs"][(b, s)] = {
                "bench": b, "subj": s,
                "acq_inst": P["inst"].to_numpy()[a], "revealed": p["revealed"], "acq_z": z.loc[a].to_numpy(),
                "eval_inst": P["inst"].to_numpy()[e], "eval_z": z.loc[e].to_numpy(),
                "n": P["n"].to_numpy()[e].astype(float), "s": P["s"].to_numpy()[e].astype(float),
                "q": P["q"].to_numpy()[e].astype(float)}
        bundle["start"], bundle["importance"] = pr["start"], pr["importance"]
        log(f"  cell {cell_name(cell)}: {len(draws)} draws, {len(groups)} pairs, prior on {pr['train_rows']:,} "
            f"training rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(bundle, fh)
    path.with_suffix(".tmp").replace(path)
    return bundle


def score_cells(bundles, key, predictors, tau1, tau2, n_jobs):
    """Run the simulator on every cell's draws (key = 'draws' or 'tune_draws'), in parallel over cells."""
    jobs = [b for b in bundles if b[key]]
    outs = Parallel(n_jobs=n_jobs)(delayed(run_draws)(b[key], b["pairs"], b["start"], predictors, tau1, tau2)
                                   for b in jobs)
    for b, o in zip(jobs, outs):
        o["cell"] = cell_name(b["cell"])
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def tune(bundles, n_jobs):
    """D7: the TAU_GRID setting with the best mean Brier ALC on the tuning draws, separately for T1 and T2 (B3 uses
    T1's). One simulator pass per setting."""
    best = {}
    for name in ("T1", "T2"):
        res = []
        for vals in itertools.product(*C.TAU_GRID.values()):
            tau = {**C.TAU, **dict(zip(C.TAU_GRID, vals))}
            rec = score_cells(bundles, "tune_draws", [name], tau, tau, n_jobs)
            alc = draw_scores(rec, keys=("cell", "draw"))["brier_alc"].mean()
            res.append((alc, tau))
        alc, best[name] = min(res, key=lambda r: r[0])
        log(f"  tuned {name}: {json.dumps({k: best[name][k] for k in C.TAU_GRID})} (tuning ALC {alc:.4f})")
    return best["T1"], best["T2"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prior", default="R1+R2+S1+I0+I2", help="components joined by '+' (components.py)")
    ap.add_argument("--name", default=None, help="results folder (default: the prior spec)")
    ap.add_argument("--regimes", nargs="+", default=["primary", "secondary"], choices=["primary", "secondary"])
    ap.add_argument("--reps", type=int, default=C.REPS, help="repetitions 0..reps-1 (D4)")
    ap.add_argument("--draws", type=int, default=C.EVAL_DRAWS, help="evaluation draws per repetition and regime")
    ap.add_argument("--tune-draws", type=int, default=C.TUNE_DRAWS)
    ap.add_argument("--tune", action="store_true", help="choose the test-time scales on the tuning draws (D7)")
    ap.add_argument("--predictors", nargs="+", default=list(PREDICTORS), choices=list(PREDICTORS))
    ap.add_argument("--benchmarks", nargs="*", help="restrict the data (quick tests)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu", "cuda"], help="LightGBM device")
    ap.add_argument("--n-jobs", type=int, default=n_jobs_default())
    ap.add_argument("--quick", action="store_true", help="1 repetition, 100 + 50 draws: a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.draws, a.tune_draws = 1, 100, 50
    parse(a.prior)                                              # fail early on a bad spec
    check_start_ram()
    name = a.name or a.prior.replace("+", "_")
    out = C.RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    device = lgb_device(a.device, a.n_jobs)
    write_json({"args": vars(a), "device": device, "tau": C.TAU, "tau_grid": C.TAU_GRID, "lgbm": C.LGBM_PARAMS,
                "train_pairs_per_benchmark": C.TRAIN_PAIRS_PER_BENCHMARK, "data": str(C.DATA)}, out / "meta.json")
    log(f"prior {a.prior}, regimes {a.regimes}, {a.reps} repetitions, device {device}, {a.n_jobs} jobs -> {out}")
    d = Data(benchmarks=a.benchmarks)
    siblings = sibling_groups(d)
    checks = json.loads((out / "checks.json").read_text()) if (out / "checks.json").exists() else {}
    taus = json.loads((out / "tau.json").read_text()) if (out / "tau.json").exists() else {}
    try:
        for rep in range(a.reps):
            R = None
            for regime in a.regimes:
                rec_path = out / f"records_rep{rep}_{regime}.parquet"
                if rec_path.exists():
                    log(f"rep {rep} {regime}: done (records exist)")
                    continue
                R = R or Repetition(d, rep, siblings)
                cells = R.cells(regime)
                n_eval = max(1, a.draws // len(cells))
                n_tune = max(1, a.tune_draws // len(cells)) if a.tune else 0
                log(f"rep {rep} {regime}: {len(cells)} cells, {n_eval} evaluation draws per cell")
                bundles = []
                for cell in cells:
                    guard_memory()
                    bundles.append(build_cell(d, R, cell, a.prior, n_eval, n_tune, out / "cache" / regime,
                                              a.n_jobs, device))
                all_draws = [x for b in bundles for x in b["draws"]]
                checks[f"rep{rep}_{regime}"] = R.check(all_draws)
                log(f"  checks: {json.dumps(checks[f'rep{rep}_{regime}'])}")
                tau1, tau2 = tune(bundles, a.n_jobs) if a.tune else (C.TAU, C.TAU)
                taus[f"rep{rep}_{regime}"] = {"T1": tau1, "T2": tau2}
                rec = score_cells(bundles, "draws", a.predictors, tau1, tau2, a.n_jobs)
                if rec.empty:
                    log(f"rep {rep} {regime}: no draws (no evaluable benchmark with eligible subjects)")
                    continue
                rec.insert(0, "regime", regime)
                rec.insert(0, "rep", rep)
                rec["benchmark"] = d.bench_names[rec["bench"].to_numpy()]
                rec["subject_id"] = d.subj_names[rec["subj"].to_numpy()]
                rec.drop(columns=["bench", "subj"]).to_parquet(rec_path, index=False)
                imp = pd.DataFrame([{"rep": rep, "regime": regime, "cell": cell_name(b["cell"]), **b["importance"]}
                                    for b in bundles if b["importance"] is not None and len(b["importance"])])
                if len(imp):
                    imp.to_csv(out / "importance.csv", mode="a", header=not (out / "importance.csv").exists(),
                               index=False)
                write_json(checks, out / "checks.json")
                write_json(taus, out / "tau.json")
                s = draw_scores(rec).groupby("predictor")[["brier_alc", "ece_alc"]].mean().sort_values("brier_alc")
                log(f"rep {rep} {regime}: Brier ALC / ECE ALC\n{s.round(4).to_string()}")
    except LowMemory as err:
        sys.exit(f"stopped: {err}. Finished parts are kept; rerun the same command to resume.")
    log(f"done -> {out}  (summaries and plots: python scripts/assessors_v3/report.py {name})")


if __name__ == "__main__":
    main()
