"""LightGBM models without LightGBM: the raw score of a model saved as text (booster.save_model), computed with numpy.

The platform does not install lightgbm, so assessor.py and assessor_input.py read the model files with this instead.
It follows LightGBM's tree evaluation (Tree::Decision, NumericalDecision and CategoricalDecision in tree.h):
- a node's decision_type: bit 0 categorical, bit 1 default left, bits 2-3 the missing type (0 none, 1 zero, 2 NaN);
- numerical: NaN counts as 0 unless the missing type is NaN; a missing value (zero with missing type zero, NaN with
  missing type NaN) goes to the default side, otherwise x <= threshold goes left;
- categorical: NaN or a negative category goes right; otherwise left if the category is in the node's bitset;
- children: a negative child c is the leaf ~c; the score is the sum of the trees' leaf values (shrinkage is already in
  them). Inputs are cast to float32 first, as LightGBM's predictor does with the frames we give it.
Pandas categorical columns are coded as LightGBM codes them: the position in the category list the model file stores
(pandas_categorical); a value not in it is missing.
"""

import json
from pathlib import Path

import numpy as np

K_ZERO = 1e-35


class Model:
    def __init__(self, path):
        lines = Path(path).read_text(encoding='utf-8').splitlines()
        head, self.trees, tree = {}, [], None
        for line in lines:
            if line.startswith('Tree='):
                tree = {}
                self.trees.append(tree)
            elif line.startswith('end of trees'):
                tree = None
            elif line.startswith('pandas_categorical:'):
                self.categories = json.loads(line.split(':', 1)[1]) or []
            elif '=' in line:
                k, v = line.split('=', 1)
                (tree if tree is not None else head)[k] = v
        assert head.get('num_tree_per_iteration', '1') == '1', 'only single-output models'
        self.feature_names = head['feature_names'].split()
        self.trees = [self._parse(t) for t in self.trees]
        cat_features = sorted({int(f) for t in self.trees for f, d in zip(t['feature'], t['decision'])
                               if d & 1})
        assert len(cat_features) == len(self.categories), 'categorical features and pandas_categorical differ'
        self.cat_codes = {self.feature_names[f]: {c: i for i, c in enumerate(cats)}
                          for f, cats in zip(cat_features, self.categories)}

    @staticmethod
    def _parse(t):
        arr = lambda k, dt: np.array(t[k].split(), dtype=dt) if t.get(k) else np.array([], dtype=dt)  # noqa: E731
        out = {'leaf': arr('leaf_value', float), 'feature': arr('split_feature', int),
               'threshold': arr('threshold', float), 'decision': arr('decision_type', int),
               'left': arr('left_child', int), 'right': arr('right_child', int)}
        if int(t.get('num_cat', 0)) > 0:
            out['cat_bounds'] = arr('cat_boundaries', int)
            out['cat_bits'] = arr('cat_threshold', np.int64).astype(np.uint32)
        return out

    def matrix(self, rows):
        """rows: list of {feature name: value} -> float matrix in the model's feature order, categories coded."""
        X = np.full((len(rows), len(self.feature_names)), np.nan)
        for j, name in enumerate(self.feature_names):
            codes = self.cat_codes.get(name)
            for i, r in enumerate(rows):
                v = r.get(name)
                if codes is not None:
                    X[i, j] = codes.get(v, np.nan)
                elif v is not None:
                    X[i, j] = float(v)
        return X.astype(np.float32).astype(float)

    def raw_score(self, rows):
        X = self.matrix(rows)
        n = len(X)
        score = np.zeros(n)
        for t in self.trees:
            if len(t['feature']) == 0:                       # a single leaf
                score += t['leaf'][0]
                continue
            node = np.zeros(n, int)
            active = np.ones(n, bool)
            while active.any():
                i = np.flatnonzero(active)
                k = node[i]
                x = X[i, t['feature'][k]]
                d = t['decision'][k]
                go_left = np.empty(len(i), bool)
                cat = (d & 1).astype(bool)
                num = ~cat
                if num.any():
                    xn, dn, th = x[num].copy(), d[num], t['threshold'][k[num]]
                    missing_type = (dn >> 2) & 3
                    xn[np.isnan(xn) & (missing_type != 2)] = 0.0
                    missing = ((missing_type == 1) & (np.abs(xn) <= K_ZERO)) | ((missing_type == 2) & np.isnan(xn))
                    default_left = (dn & 2).astype(bool)
                    go_left[num] = np.where(missing, default_left, xn <= th)
                if cat.any():
                    xc, kc = x[cat], k[cat]
                    res = np.zeros(len(xc), bool)
                    ok = ~np.isnan(xc)
                    ok[ok] = xc[ok] >= 0
                    for j in np.flatnonzero(ok):
                        c = int(xc[j])
                        idx = int(t['threshold'][kc[j]])
                        lo, hi = t['cat_bounds'][idx], t['cat_bounds'][idx + 1]
                        word = c // 32
                        res[j] = word < hi - lo and bool((t['cat_bits'][lo + word] >> (c % 32)) & 1)
                    go_left[cat] = res
                child = np.where(go_left, t['left'][k], t['right'][k])
                leaf = child < 0
                score[i[leaf]] += t['leaf'][~child[leaf]]
                node[i] = np.where(leaf, 0, child)
                active[i[leaf]] = False
        return score
