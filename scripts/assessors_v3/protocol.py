"""Evaluation protocol v3 (evaluation_protocol_v3.md): partitions, fixed benchmark subsets, splits and draws.

One Repetition (D4) fixes, from its seed:
    bench_fold   benchmark code -> fold 0..K_BENCH-1 (D2: sibling groups together, balanced by evaluable pairs)
    model_fold   model code -> fold 0..K_MODEL-1 (D3: by normalized_name, balanced by evaluable pairs)
    subset       evaluable benchmark -> its fixed instance subset, of a formative size (D1)
    eligible     evaluable benchmark -> the subjects that answered the whole subset (the only ones drawn)
    pair(b, s)   the pair's persistent 50/50 split, acquisition order and revealed labels
and draws() produces the draws of a cell with the platform's greedy rule (D5).

Cells. Primary regime (unseen benchmark, unseen model): cell (k, j) = benchmark fold k x model fold j; its prior is
trained on pairs whose benchmark is not in fold k AND whose model is not in fold j (train_mask), and its draws use
only fold-k benchmarks with fold-j subjects. Secondary regime (unseen benchmark, known model): cell (k, None); train
outside fold k, draw fold-k benchmarks with any subject.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import log  # noqa: E402
from config import (DRAW_ITEM_CAP, FORMATIVE_EVAL_ITEMS, K_BENCH, K_MODEL, MIN_ITEMS, SIBLING_OVERLAP,  # noqa: E402
                    sibling_group)


# ----------------------------------------------------------------------------- sibling groups (D2)

def sibling_groups(d):
    """Benchmark code -> sibling group id. Joined by the name rules (config.sibling_group) and by shared item texts:
    two benchmarks sharing >= SIBLING_OVERLAP of the smaller one's items (exact match of the normalised text) are
    joined, transitively (union-find). Only annotated items are compared (at most 600 per benchmark), a sample of
    each benchmark's items."""
    nb = len(d.bench_names)
    parent = list(range(nb))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    by_name = {}
    for b, name in enumerate(d.bench_names):
        g = sibling_group(name)
        if g in by_name:
            union(b, by_name[g])
        by_name.setdefault(g, b)
    item_bench = d.inst_bench[np.unique(d.inst_item, return_index=True)[1]]   # benchmark of each item code
    h = pd.DataFrame({"hash": d.items["text_hash"].to_numpy(), "bench": item_bench}).drop_duplicates()
    size = h.groupby("bench").size()
    shared = h.merge(h, on="hash").query("bench_x < bench_y").groupby(["bench_x", "bench_y"]).size()
    for (a, b), n in shared.items():
        if n >= SIBLING_OVERLAP * min(size[a], size[b]):
            union(int(a), int(b))
    return np.array([find(b) for b in range(nb)])


def _balanced(groups, weight, k, rng, fallback_weight=None):
    """Assign group ids to k folds: groups in random order (heavier first, with noise), each to the currently
    lightest fold. Groups of zero weight (e.g. training-only benchmarks) are spread by fallback_weight instead."""
    gw = pd.Series(weight).groupby(groups).sum()
    fw = pd.Series(fallback_weight if fallback_weight is not None else np.ones(len(groups))).groupby(groups).sum()
    fold_of, load, load2 = {}, np.zeros(k), np.zeros(k)
    pos = gw[gw > 0]
    for g in (pos * rng.uniform(0.5, 1.5, len(pos))).sort_values(ascending=False).index:
        f = int(np.argmin(load + rng.uniform(0, 1e-6, k)))
        fold_of[g] = f
        load[f] += gw[g]
    zero = fw[gw <= 0]
    for g in (zero * rng.uniform(0.5, 1.5, len(zero))).sort_values(ascending=False).index:
        f = int(np.argmin(load2 + rng.uniform(0, 1e-6, k)))
        fold_of[g] = f
        load2[f] += fw[g]
    return np.array([fold_of[g] for g in groups])


# ----------------------------------------------------------------------------- one repetition

class Repetition:
    def __init__(self, d, rep, siblings=None):
        self.d, self.rep = d, rep
        rng = np.random.default_rng([rep, 7])
        P = d.P
        ev = set(d.evaluable.tolist())
        # eligible pairs (>= 80 instances) of evaluable benchmarks: the weights of the fold balance
        per = P.groupby(["bench", "subj"]).size()
        per = per[(per >= MIN_ITEMS) & per.index.get_level_values("bench").isin(ev)]
        bench_pairs = per.groupby(level="bench").size().reindex(range(len(d.bench_names)), fill_value=0)
        model_pairs = pd.Series(d.subj_model[per.index.get_level_values("subj")]).value_counts() \
            .reindex(range(len(d.model_names)), fill_value=0)
        bench_rows = P.groupby("bench").size().reindex(range(len(d.bench_names)), fill_value=0)
        model_rows = P.groupby("model").size().reindex(range(len(d.model_names)), fill_value=0)
        self.siblings = siblings = sibling_groups(d) if siblings is None else siblings
        self.bench_fold = _balanced(siblings, bench_pairs.to_numpy(), K_BENCH, rng, bench_rows.to_numpy())
        self.model_fold = _balanced(np.arange(len(d.model_names)), model_pairs.to_numpy(), K_MODEL, rng,
                                    model_rows.to_numpy())
        # fixed subsets (D1)
        self.rows = {}         # (bench, subj) -> row indices in d.P, evaluable benchmarks only
        self.subset, self.eligible, self.skipped = {}, {}, []
        sizes, probs = zip(*FORMATIVE_EVAL_ITEMS.items())
        probs = np.array(probs, float) / sum(probs)
        Pev = P[P["bench"].isin(ev)]
        self.subjects_of = {}  # bench -> its subjects
        for (b, s), idx in Pev.groupby(["bench", "subj"]).indices.items():
            self.rows[(b, s)] = Pev.index.to_numpy()[idx]
            self.subjects_of.setdefault(b, []).append(s)
        for b in sorted(ev):
            m = 2 * int(rng.choice(sizes, p=probs)) + int(rng.random() < 0.5)
            sub, elig = self._subset(b, m, rng)
            if sub is None:
                self.skipped.append(d.bench_names[b])
                continue
            self.subset[b], self.eligible[b] = sub, elig
        self._pairs = {}
        log(f"rep {rep}: {len(self.subset)} benchmarks with a fixed subset "
            f"(median {int(np.median([len(v) for v in self.subset.values()])) if self.subset else 0} instances), "
            f"{sum(len(v) for v in self.eligible.values())} eligible pairs"
            + (f"; {len(self.skipped)} without enough complete subjects" if self.skipped else ""))

    def _subset(self, b, m, rng):
        """A random subset of m instances of benchmark b answered completely by as many subjects as possible.
        Most benchmarks are complete matrices (every subject answers everything) and this is a plain random
        subset. Otherwise: subjects by decreasing coverage (random among ties), intersecting their instance sets while
        the intersection keeps >= m instances; the subset is drawn from that intersection. A benchmark smaller than m
        keeps all its common instances; one whose three best-covered subjects share < MIN_ITEMS instances is skipped
        in this repetition (training only)."""
        inst = self.d.P["inst"].to_numpy()
        sets = {s: set(inst[self.rows[(b, s)]].tolist()) for s in self.subjects_of[b]}
        order = sorted(sets, key=lambda s: (-len(sets[s]), rng.random()))
        common = set(sets[order[0]])
        for k, s in enumerate(order[1:], start=2):
            nxt = common & sets[s]
            if len(nxt) < m and k > 3:
                break
            common = nxt
            if len(common) < MIN_ITEMS:
                return None, None
        pool = np.array(sorted(common), dtype=np.int64)
        sub = np.sort(rng.choice(pool, size=min(m, len(pool)), replace=False))
        subset = set(sub.tolist())
        elig = np.array([s for s in order if subset <= sets[s]], dtype=np.int64)
        return sub, elig

    # ------------------------------------------------------------------------- pairs

    def pair(self, b, s):
        """The pair's persistent split (D4): acquisition instances in their fixed order, their revealed labels, and
        the evaluation instances with their run statistics (n, s, q). Odd count: the extra instance goes to
        acquisition. A revealed label is one Bernoulli draw with the instance's mean score (exact for one 0/1 run)."""
        key = (b, s)
        if key not in self._pairs:
            rng = np.random.default_rng([self.rep, int(b), int(s), 11])
            P = self.d.P
            rows = self.rows[key]
            sub = self.subset[b]
            r = rows[np.isin(P["inst"].to_numpy()[rows], sub)]
            r = r[rng.permutation(len(r))]
            n_acq = (len(r) + 1) // 2
            acq, ev = r[:n_acq], r[n_acq:]
            mean_acq = P["s"].to_numpy()[acq] / P["n"].to_numpy()[acq]
            self._pairs[key] = {
                "bench": int(b), "subj": int(s),
                "acq_rows": acq, "revealed": (rng.random(len(acq)) < mean_acq).astype(float),
                "eval_rows": ev,
            }
        return self._pairs[key]

    # ------------------------------------------------------------------------- draws (D5)

    def cells(self, regime):
        if regime == "primary":
            return [(k, j) for k in range(K_BENCH) for j in range(K_MODEL)]
        return [(k, None) for k in range(K_BENCH)]

    def train_mask(self, cell):
        """Pairs a cell's prior may be trained on."""
        k, j = cell
        P = self.d.P
        m = self.bench_fold[P["bench"].to_numpy()] != k
        if j is not None:
            m &= self.model_fold[P["model"].to_numpy()] != j
        return m

    def candidates(self, cell):
        """(benchmark, subject, size) groups a cell's draws can use."""
        k, j = cell
        out = []
        for b, elig in self.eligible.items():
            if self.bench_fold[b] != k:
                continue
            for s in elig:
                if j is None or self.model_fold[self.d.subj_model[s]] == j:
                    out.append((int(b), int(s), len(self.subset[b])))
        return out

    def draws(self, cell, n_draws, seed):
        """The platform's rule: complete groups are added while they fit under DRAW_ITEM_CAP items, each time one
        whose benchmark, then whose subject, is least represented in the draw so far (random tie-breaking); stop
        when none fits. The cap counts the summed group sizes (as in night_experiments/mixed.py): counting distinct
        items, a second subject on the same benchmark would be free, and draws would take every subject of a
        benchmark, unlike the 8-9 pairs on 7 benchmarks of the formative feedback."""
        groups = self.candidates(cell)
        k, j = cell
        out = []
        for t in range(n_draws if groups else 0):
            rng = np.random.default_rng([seed, self.rep, k, K_MODEL if j is None else j, t])
            left = list(rng.permutation(len(groups)))
            total, nb, ns, draw = 0, {}, {}, []
            while True:
                fits = [i for i in left if total + groups[i][2] <= DRAW_ITEM_CAP]
                if not fits:
                    break
                i = min(fits, key=lambda g: (nb.get(groups[g][0], 0), ns.get(groups[g][1], 0)))  # first = random
                b, s, size = groups[i]
                left.remove(i)
                total += size
                nb[b] = nb.get(b, 0) + 1
                ns[s] = ns.get(s, 0) + 1
                draw.append((b, s))
            out.append(draw)
        return out

    def check(self, draws):
        """The draw checks of section 5, to compare with the formative table (section 1)."""
        if not draws:
            return {}
        pairs = [len(x) for x in draws]
        benches = [len({b for b, _ in x}) for x in draws]
        items = [sum(len(self.subset[b]) for b, _ in x) for x in draws]
        ev = [len(self.pair(b, s)["eval_rows"]) for x in draws for b, s in x]
        q = lambda v: f"{np.min(v)}-{np.max(v)} (median {np.median(v):g})"   # noqa: E731
        return {"draws": len(draws), "pairs per draw": q(pairs), "benchmarks per draw": q(benches),
                "items per draw": q(items), "evaluation instances per pair": q(ev)}
