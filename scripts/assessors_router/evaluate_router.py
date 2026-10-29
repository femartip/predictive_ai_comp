"""Evaluate the routed assessors under evaluation protocol v3 (README section 5), then write the report.

    poetry run python scripts/assessors_router/evaluate_router.py [--tune] [--reps 5] [--regimes primary secondary]
        [--k 3] [--parts v3_assessor tier_assessors general_assessor general_assessor_oracle] [--name NAME] [--quick]
        [--data data/datasets/assessor_data/BUILD]

Run it inside tmux on this server (the full protocol takes hours). The data is an assessor_data build (default: the
newest), loaded by bank.py. Protocol v3 unchanged: repetitions, cells, draws, splits and acquisition orders only depend
on the data and the repetition's seed, so every predictor of a run is scored on the same draws.
For every repetition, regime and cell:
  1. the cell's evaluation draws (seed 0) and tuning draws (seed 100);
  2. from the cell's training rows only: model performance, tiers and the router (strength.route);
  3. the assessors (assessors.py) predict every instance of the drawn pairs. Each is cached on its own in
     <run>/cache/<regime>/rep<r>_<cell>/, so a later run with more assessors fits only the new ones;
  4. with --tune, each predictor's test-time scales are chosen on the tuning draws (TAU_GRID), else config.TAU;
  5. every predictor whose assessor was fitted is scored on the evaluation draws (streaming.py).
Outputs in results/assessors_router/<run>/ (default run name: general): records_rep<r>_<regime>.parquet (as v3);
benchmarks.csv and models.csv (the loaded data's benchmarks, and the model table given to S1);
routes_rep<r>_<regime>.parquet (one row per cell and test subject: known model, router and true tiers, model
performance); importance.csv; tau.json; checks.json; meta.json; report/ (report_router.py). Resumable.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")    # parallelism is across cells; LightGBM sets its own threads
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as S  # noqa: E402  (first: points assessors_v3 at the repository's paths)
import assessors as A  # noqa: E402
import config as C  # noqa: E402
import report_router  # noqa: E402
from bank import BankData, builds  # noqa: E402
from common import LowMemory, check_start_ram, guard_memory, log, write_json  # noqa: E402
from prior import lgb_device  # noqa: E402
from protocol import Repetition, sibling_groups  # noqa: E402
from simulate import draw_scores  # noqa: E402
from strength import oracle_measures, route  # noqa: E402
from streaming import BASELINES, PART, run_draws  # noqa: E402


def cell_name(cell):
    return f"{cell[0]}x{'all' if cell[1] is None else cell[1]}"


def cached(path, fn):
    """Load path if it exists, else compute fn() and write it atomically."""
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    obj = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(obj, fh)
    path.with_suffix(".tmp").replace(path)
    return obj


def build_cell(d, R, cell, k, parts, n_eval, n_tune, cdir, oracle, n_jobs, device):
    """Draws, routing and assessors of one cell, assembled into the per-pair arrays streaming.py needs."""
    tag = f"d{n_eval}_t{n_tune}"           # the drawn pairs (hence the test rows) depend on both counts
    dr = cached(cdir / f"draws_{tag}.pkl", lambda: {"draws": R.draws(cell, n_eval, C.EVAL_SEED),
                                                    "tune_draws": R.draws(cell, n_tune, C.TUNE_SEED)})
    bundle = {"cell": cell, **dr, "pairs": {}, "importance": [], "routes": None}
    groups = sorted({g for x in dr["draws"] + dr["tune_draws"] for g in x})
    if not groups:
        return bundle
    info = [R.pair(b, s) for b, s in groups]
    test_rows = np.unique(np.concatenate([np.concatenate([p["acq_rows"], p["eval_rows"]]) for p in info]))
    rt = cached(cdir / f"routing_k{k}_{tag}_cv3.pkl",         # cv3: CV training values, completeness
                lambda: route(d, R, cell, [s for _, s in groups], oracle, k, n_jobs))
    jobs = {"v3_assessor": lambda: A.fit_v3(d, R, cell, test_rows, n_jobs, device),
            "tier_assessors": lambda: A.fit_tiers(d, R, cell, rt, test_rows, n_jobs, device),
            "general_assessor": lambda: A.fit_general(d, R, cell, rt, test_rows, n_jobs, device, "used"),
            "general_assessor_oracle": lambda: A.fit_general(d, R, cell, rt, test_rows, n_jobs, device, "oracle")}
    fits = {}
    for part in parts:
        # _inf: subject features from inference fields only
        version = {"tier_assessors": f"_k{k}", "general_assessor": "_cv2_inf_kept",   # _kept: trees and features
                   "general_assessor_oracle": "_cv2_inf"}
        fits[part] = cached(cdir / f"{part}{version.get(part, '')}_{tag}.pkl", jobs[part])
    if "tier_assessors" in fits and any(f is None for f in fits["tier_assessors"]) and "v3_assessor" not in fits:
        fits["v3_assessor"] = cached(cdir / f"v3_assessor_{tag}.pkl", jobs["v3_assessor"])   # for an empty tier

    P = d.P
    inst, n, s, q = (P[c].to_numpy() for c in ("inst", "n", "s", "q"))
    pos = pd.Index(test_rows)
    tests = rt["test"].set_index("subj")
    for (b, sj), p in zip(groups, info):
        a, e = p["acq_rows"], p["eval_rows"]
        ia, ie = pos.get_indexer(a), pos.get_indexer(e)
        t = tests.loc[sj]
        pair = {"bench": b, "subj": sj, "acq_inst": inst[a], "revealed": p["revealed"], "eval_inst": inst[e],
                "n": n[e].astype(float), "s": s[e].astype(float), "q": q[e].astype(float), "r1": rt["r1"],
                "tier_router": int(t["tier_router"]), "tier_oracle": int(t["tier_oracle"])}
        for part in ("v3_assessor", "general_assessor", "general_assessor_oracle"):
            if part in fits:
                f = fits[part]
                pair[part] = {"acq": f["z"][ia], "eval": f["z"][ie], "start": f["start"]}
        if "tier_assessors" in fits:
            fs = [f if f is not None else fits["v3_assessor"] for f in fits["tier_assessors"]]
            pair["tier_assessors"] = {"acq": np.stack([f["z"][ia] for f in fs]),
                                      "eval": np.stack([f["z"][ie] for f in fs]),
                                      "starts": np.array([f["start"] for f in fs])}
        bundle["pairs"][(b, sj)] = pair
    if "general_assessor" in fits:      # B0 as predict() builds it, per draw and budget (assessors.evidence_logits)
        by_group = dict(zip(groups, info))
        bundle["evidence"] = {key: [A.evidence_logits(d, fits["general_assessor"], pos, draw, by_group, C.BUDGETS)
                                    for draw in dr[key]] for key in ("draws", "tune_draws")}
    for part, f in fits.items():
        for j, fj in (enumerate(f) if part == "tier_assessors" else [(None, f)]):
            if fj is not None and len(fj["importance"]):
                bundle["importance"].append(fj["importance"].rename_axis("feature").rename("gain").reset_index()
                                            .assign(assessor=part if j is None else f"tier{j}"))
    bundle["routes"] = rt["test"].assign(cell=cell_name(cell))
    log(f"  cell {cell_name(cell)}: {len(dr['draws'])} draws, {len(groups)} pairs, "
        f"{int(rt['test']['known'].sum())}/{len(rt['test'])} test subjects known")
    return bundle


def score_cells(bundles, key, predictors, taus, n_jobs):
    """Run the simulator on every cell's draws (key = 'draws' or 'tune_draws'), in parallel over cells."""
    jobs = [b for b in bundles if b[key]]
    outs = Parallel(n_jobs=n_jobs)(delayed(run_draws)(b[key], b["pairs"], predictors, taus,
                                                      evidence=b.get("evidence", {}).get(key)) for b in jobs)
    for b, o in zip(jobs, outs):
        o["cell"] = cell_name(b["cell"])
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def tune(bundles, predictors, n_jobs):
    """D7: per predictor, the TAU_GRID setting with the best mean Brier ALC on the tuning draws. B3 (T1, no slope)
    ignores alpha, so only its other scales are searched."""
    taus = {}
    for name in predictors:
        if name in ("B1", "B2", "oracle"):
            continue
        grid = {g: v for g, v in C.TAU_GRID.items() if not (name == "B3" and g == "alpha")}
        res = []
        for vals in itertools.product(*grid.values()):
            tau = {**C.TAU, **dict(zip(grid, vals))}
            rec = score_cells(bundles, "tune_draws", [name], {name: tau}, n_jobs)
            res.append((draw_scores(rec, keys=("cell", "draw"))["brier_alc"].mean(), tau))
        alc, taus[name] = min(res, key=lambda r: r[0])
        log(f"  tuned {name}: {json.dumps({g: taus[name][g] for g in grid})} (tuning ALC {alc:.4f})")
    return taus


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--k", type=int, default=S.K_TIERS, help="performance tiers")
    ap.add_argument("--parts", nargs="+", choices=A.PARTS, help="assessors to fit (default: all but tier_assessors)",
                    default=[p for p in A.PARTS if p != "tier_assessors"])
    ap.add_argument("--name", default="general", help="run folder in results/assessors_router")
    ap.add_argument("--regimes", nargs="+", default=["primary", "secondary"], choices=["primary", "secondary"])
    ap.add_argument("--reps", type=int, default=C.REPS, help="repetitions 0..reps-1 (D4)")
    ap.add_argument("--draws", type=int, default=C.EVAL_DRAWS, help="evaluation draws per repetition and regime")
    ap.add_argument("--tune-draws", type=int, default=C.TUNE_DRAWS)
    ap.add_argument("--tune", action="store_true", help="choose the test-time scales on the tuning draws (D7)")
    ap.add_argument("--n-jobs", type=int, default=S.N_JOBS)
    ap.add_argument("--data", type=Path, default=None, help="assessor_data build (default: the newest)")
    ap.add_argument("--quick", action="store_true", help="1 repetition, 100 + 50 draws: a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.reps, a.draws, a.tune_draws = 1, 100, 50
    os.nice(10)                                                   # the server is shared
    check_start_ram()
    a.data = a.data or builds()[-1]
    predictors = list(BASELINES) + [p for p in PART if PART[p] in a.parts]
    name = a.name
    out = S.RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    device = lgb_device(S.DEVICE, a.n_jobs)
    write_json({"args": vars(a), "device": device, "prior": S.PRIOR, "predictors": predictors, "tau": C.TAU,
                "tau_grid": C.TAU_GRID, "lgbm": C.LGBM_PARAMS, "router": S.ROUTER_PARAMS,
                "strength_shrink": S.STRENGTH_SHRINK, "min_tier_pairs": S.MIN_TIER_PAIRS,
                "train_pairs_per_benchmark": C.TRAIN_PAIRS_PER_BENCHMARK, "data": str(a.data)}, out / "meta.json")
    log(f"routed assessors, K={a.k}, assessors {a.parts}, regimes {a.regimes}, {a.reps} repetitions, "
        f"{a.n_jobs} jobs -> {out}")
    d = BankData(a.data, out / "models.csv")
    d.benchmarks.assign(benchmark=d.bench_names).to_csv(out / "benchmarks.csv", index=False)
    siblings = sibling_groups(d)
    oracle = oracle_measures(d)
    checks = json.loads((out / "checks.json").read_text()) if (out / "checks.json").exists() else {}
    taus_all = json.loads((out / "tau.json").read_text()) if (out / "tau.json").exists() else {}
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
                    bundles.append(build_cell(d, R, cell, a.k, a.parts, n_eval, n_tune,
                                              out / "cache" / regime / f"rep{rep}_{cell_name(cell)}", oracle,
                                              a.n_jobs, device))
                checks[f"rep{rep}_{regime}"] = R.check([x for b in bundles for x in b["draws"]])
                log(f"  checks: {json.dumps(checks[f'rep{rep}_{regime}'])}")
                taus = tune(bundles, predictors, a.n_jobs) if a.tune else {p: C.TAU for p in predictors}
                taus_all[f"rep{rep}_{regime}"] = taus
                rec = score_cells(bundles, "draws", predictors, taus, a.n_jobs)
                if rec.empty:
                    log(f"rep {rep} {regime}: no draws (no evaluable benchmark with eligible subjects)")
                    continue
                rec.insert(0, "regime", regime)
                rec.insert(0, "rep", rep)
                rec["benchmark"] = d.bench_names[rec["bench"].to_numpy()]
                rec["subject_id"] = d.subj_names[rec["subj"].to_numpy()]
                routes = pd.concat([b["routes"] for b in bundles if b["routes"] is not None], ignore_index=True)
                routes.insert(0, "regime", regime)
                routes.insert(0, "rep", rep)
                routes["subject_id"] = d.subj_names[routes["subj"].to_numpy()]
                routes["model_name"] = d.model_names[routes["model"].to_numpy()]
                routes.drop(columns=["subj", "model"]).to_parquet(out / f"routes_rep{rep}_{regime}.parquet",
                                                                  index=False)
                imp = [t.assign(rep=rep, regime=regime, cell=cell_name(b["cell"]))
                       for b in bundles for t in b["importance"]]
                if imp:
                    pd.concat(imp, ignore_index=True).to_csv(out / "importance.csv", mode="a",
                                                             header=not (out / "importance.csv").exists(), index=False)
                write_json(checks, out / "checks.json")
                write_json(taus_all, out / "tau.json")
                rec.drop(columns=["bench", "subj"]).to_parquet(rec_path, index=False)   # last: marks this part done
                s = draw_scores(rec).groupby("predictor")[["brier_alc", "ece_alc"]].mean().sort_values("brier_alc")
                log(f"rep {rep} {regime}: Brier ALC / ECE ALC\n{s.round(4).to_string()}")
    except LowMemory as err:
        sys.exit(f"stopped: {err}. Finished parts are kept; rerun the same command to resume.")
    report_router.report(name)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
