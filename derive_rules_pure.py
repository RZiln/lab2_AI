#!/usr/bin/env python3
"""
derive_rules_pure.py - derive classification rules from a CSV file with
JRip (RIPPER), ID3, CN2 and PART, implemented from scratch with only
numpy + pandas (no Weka, Orange or scikit-learn).

    python derive_rules_pure.py data.csv --algo all
    python derive_rules_pure.py data.csv --algo jrip part --class-col species
    python derive_rules_pure.py data.csv --algo cn2 --cn2-variant unordered --out-dir rules_out

Each algorithm prints the learned rules and a stratified k-fold
cross-validated accuracy (use --cv 0 to skip it).

Notes on fidelity
-----------------
* JRip : RIPPER as in Cohen (1995) / Weka's JRip - classes processed from
         rarest to most frequent, IREP* grow (FOIL gain) / prune, MDL stopping
         and the rule-optimisation passes (replacement / revision). The MDL bit
         counts follow Weka's formulas closely but are not guaranteed to be
         bit-identical, so rule sets can differ slightly from Weka's.
* PART : partial C4.5 trees (gain ratio, pessimistic-error pruning) from which
         the leaf covering most instances becomes a rule (Frank & Witten 1998).
         Missing values: numeric -> training median, categorical -> category '?'.
* CN2  : beam search with Laplace accuracy (or entropy), likelihood-ratio
         significance test, ordered (1989) or unordered (1991) variant.
* ID3  : classic information-gain tree turned into rules; numeric columns are
         discretised into quantile bins first.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

ALGOS = ["jrip", "id3", "cn2", "part"]


# ==========================================================================
# Small numeric helpers
# ==========================================================================
def entropy_rows(counts):
    """Shannon entropy (bits) along the last axis of a count array."""
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, counts / n, 0.0)
        lp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return -(p * lp).sum(axis=-1)


def _gammaq(a, x):
    """Regularised upper incomplete gamma function Q(a, x)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-13:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-13:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x, dof):
    return _gammaq(dof / 2.0, x / 2.0)


def chi2_critical(alpha, dof):
    """x such that chi2_sf(x, dof) == alpha (bisection)."""
    lo, hi = 0.0, 1.0
    while chi2_sf(hi, dof) > alpha:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if chi2_sf(mid, dof) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def add_errs(n, e, cf):
    """C4.5 / Weka pessimistic extra errors (upper confidence bound)."""
    if cf > 0.5:
        return 0.0
    if e < 1:
        base = n * (1 - math.pow(cf, 1.0 / n))
        if e == 0:
            return base
        return base + e * (add_errs(n, 1, cf) - base)
    if e + 0.5 >= n:
        return max(n - e, 0.0)
    z = NormalDist().inv_cdf(1 - cf)
    f = (e + 0.5) / n
    r = (f + z * z / (2 * n) + z * math.sqrt(f / n - f * f / n + z * z / (4 * n * n))) / (1 + z * z / n)
    return r * n - e


# ==========================================================================
# Data loading
# ==========================================================================
def load_data(path, sep, class_col, drop_cols):
    df = pd.read_csv(path, sep=sep, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if class_col is None:
        class_col = df.columns[-1]
    if class_col not in df.columns:
        sys.exit(f"Class column '{class_col}' not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[class_col]).reset_index(drop=True)
    df[class_col] = df[class_col].astype(str)
    return df, class_col


def split_columns(df, target, force_categorical=()):
    num, cat = [], []
    for c in df.columns:
        if c == target:
            continue
        is_num = pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        (num if (is_num and c not in force_categorical) else cat).append(c)
    return num, cat


# ==========================================================================
# Conditions and rules (shared by JRip, CN2, PART)
# ==========================================================================
@dataclass(frozen=True)
class Cond:
    attr: str
    op: str          # '==', '<=', '>'
    val: object

    def mask(self, df, cache=None):
        key = (self.attr, self.op == "==")
        arr = cache.get(key) if cache is not None else None
        if arr is None:
            col = df[self.attr]
            if self.op == "==":
                arr = (col.isna().to_numpy(), col.astype(str).to_numpy())
            else:
                arr = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
            if cache is not None:
                cache[key] = arr
        if self.op == "==":
            isna, s = arr
            if self.val == "?":
                return isna | (s == "?")
            return (~isna) & (s == self.val)
        with np.errstate(invalid="ignore"):
            return (arr <= self.val) if self.op == "<=" else (arr > self.val)

    def text(self):
        if self.op == "==":
            return f"{self.attr} = {self.val}"
        return f"{self.attr} {self.op} {self.val:g}"


def make_conditions(df, num_cols, cat_cols, max_thr):
    """Candidate conditions: thresholds for numeric, equality for categorical."""
    conds = []
    for c in num_cols:
        vals = np.unique(pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(float))
        if len(vals) < 2:
            continue
        mids = (vals[:-1] + vals[1:]) / 2
        if len(mids) > max_thr:
            sel = np.unique(np.linspace(0, len(mids) - 1, max_thr).round().astype(int))
            mids = mids[sel]
        for t in mids:
            conds.append(Cond(c, "<=", float(t)))
            conds.append(Cond(c, ">", float(t)))
    for c in cat_cols:
        for v in sorted(df[c].dropna().astype(str).unique()):
            conds.append(Cond(c, "==", v))
    return conds


def cond_matrix(df, conds):
    cache = {}
    if not conds:
        return np.zeros((0, len(df)), dtype=bool)
    return np.vstack([c.mask(df, cache) for c in conds])


@dataclass
class Rule:
    conds: list
    label: str
    n: int = 0
    err: int = 0
    dist: object = None

    def mask(self, df, cache=None):
        m = np.ones(len(df), dtype=bool)
        for c in self.conds:
            m &= c.mask(df, cache)
        return m

    def text(self, target, classes=None):
        cond = " AND ".join(c.text() for c in self.conds) if self.conds else "TRUE"
        extra = ""
        if self.dist is not None and classes is not None:
            extra = "  dist=" + str({c: int(d) for c, d in zip(classes, self.dist) if d})
        return f"IF {cond} THEN {target} = {self.label}   [covers {self.n}, errors {self.err}]{extra}"


class RuleListModel:
    """Ordered decision list with a default class."""

    rules: list
    default: str
    default_n: int = 0
    default_err: int = 0
    target: str = "class"

    def predict_list(self, df):
        out = np.full(len(df), self.default, dtype=object)
        todo = np.ones(len(df), dtype=bool)
        cache = {}
        for r in self.rules:
            m = r.mask(df, cache) & todo
            out[m] = r.label
            todo &= ~m
        return out

    predict = predict_list

    def rules_text(self):
        classes = getattr(self, "classes", None)
        lines = [r.text(self.target, classes if self.show_dist else None) for r in self.rules]
        lines.append(f"ELSE {self.target} = {self.default}   [covers {self.default_n}, errors {self.default_err}]")
        return "\n".join(lines)

    show_dist = False


# ==========================================================================
# JRip (RIPPER)
# ==========================================================================
def _subset_dl(t, k, p):
    p = min(max(p, 0.0), 1 - 1e-12)
    rt = -k * math.log2(p) if p > 0 else 0.0
    rt -= (t - k) * math.log2(1 - p)
    return rt


def _data_dl(exp_fp_over_err, cover, uncover, fp, fn):
    total = math.log2(cover + uncover + 1.0)
    if cover > uncover:
        exp_err = exp_fp_over_err * (fp + fn)
        cover_bits = _subset_dl(cover, fp, exp_err / cover)
        uncover_bits = _subset_dl(uncover, fn, fn / uncover) if uncover > 0 else 0.0
    else:
        exp_err = (1.0 - exp_fp_over_err) * (fp + fn)
        cover_bits = _subset_dl(cover, fp, fp / cover) if cover > 0 else 0.0
        uncover_bits = _subset_dl(uncover, fn, exp_err / uncover) if uncover > 0 else 0.0
    return total + cover_bits + uncover_bits


class RIPPER(RuleListModel):
    name = "JRip"

    def __init__(self, folds=3, min_no=2.0, optimizations=2, seed=1, max_thr=20,
                 check_err=True, mdl_surplus=64.0):
        self.folds, self.min_no, self.optimizations = folds, min_no, optimizations
        self.seed, self.max_thr = seed, max_thr
        self.check_err, self.mdl_surplus = check_err, mdl_surplus

    # ---- helpers ---------------------------------------------------------
    def _cover(self, idx):
        key = tuple(idx)
        m = self._cache.get(key)
        if m is None:
            m = np.ones(self.N, dtype=bool)
            for i in key:
                m = m & self.M[i]
            self._cache[key] = m
        return m

    def _union(self, rules):
        m = np.zeros(self.N, dtype=bool)
        for r in rules:
            m |= self._cover(r)
        return m

    def _theory_dl(self, rule):
        k = len(rule)
        if k == 0:
            return 0.0
        tdl = math.log2(k)
        if k > 1:
            tdl += 2.0 * math.log2(tdl)
        tdl -= math.lgamma(k + 1) / math.log(2)          # order of conditions is irrelevant
        for i in rule:
            tdl += math.log2(max(self.n_attrs, 1)) + math.log2(max(self.choices[self.conds[i].attr], 1))
        return 0.5 * max(tdl, 0.0)

    def _total_dl(self, rules, stage, pos):
        cov = self._union(rules) & stage
        cover = int(cov.sum())
        uncover = int(stage.sum()) - cover
        fp = int((cov & ~pos).sum())
        fn = int((stage & ~cov & pos).sum())
        return sum(self._theory_dl(r) for r in rules) + _data_dl(0.5, cover, uncover, fp, fn)

    def _split(self, rem, pos):
        grow = np.zeros(self.N, dtype=bool)
        prune = np.zeros(self.N, dtype=bool)
        for sel in (rem & pos, rem & ~pos):
            idx = np.flatnonzero(sel)
            self.rng.shuffle(idx)
            k = int(round(len(idx) / self.folds))
            prune[idx[:k]] = True
            grow[idx[k:]] = True
        return grow, prune

    def _grow(self, grow, pos, start=()):
        conds = list(start)
        cov = grow.copy()
        for i in conds:
            cov &= self.M[i]
        used = set(conds)
        while True:
            pm, nm = cov & pos, cov & ~pos
            p0, n0 = int(pm.sum()), int(nm.sum())
            if p0 == 0 or n0 == 0:
                break
            p1 = self.Mf @ pm.astype(np.float32)
            n1 = self.Mf @ nm.astype(np.float32)
            tot = p1 + n1
            ratio = np.divide(p1, tot, out=np.zeros_like(p1), where=tot > 0)
            lr = np.log2(np.where(ratio > 0, ratio, 1.0))
            gain = p1 * (lr - math.log2(p0 / (p0 + n0)))
            ok = (p1 >= self.min_no) & (p1 > 0)
            if used:
                ok[list(used)] = False
            gain = np.where(ok, gain, -np.inf)
            best = int(np.argmax(gain))
            if not np.isfinite(gain[best]) or gain[best] <= 1e-12:
                break
            conds.append(best)
            used.add(best)
            cov &= self.M[best]
        return conds

    def _prune(self, conds, prune, pos):
        if not conds or not prune.any():
            return conds
        best_v, best_len = -np.inf, len(conds)
        cov = prune.copy()
        for length, i in enumerate(conds, 1):
            cov = cov & self.M[i]
            p, n = int((cov & pos).sum()), int((cov & ~pos).sum())
            v = (p - n) / (p + n) if p + n > 0 else -1.0
            if v > best_v:
                best_v, best_len = v, length
        return conds[:best_len]

    # ---- IREP* with MDL stopping -----------------------------------------
    def _irep(self, stage, pos, prefix=()):
        prefix = list(prefix)
        rules = []
        rem = stage & ~self._union(prefix)
        best_dl = self._total_dl(prefix, stage, pos)
        best_k = 0
        while (rem & pos).any():
            grow, prune = self._split(rem, pos)
            conds = self._grow(grow, pos)
            if not conds:
                break
            conds = self._prune(conds, prune, pos)
            if not conds:
                break
            m = self._cover(conds)
            if self.check_err:
                pm = prune & m
                tot = int(pm.sum())
                if tot > 0 and (pm & ~pos).sum() / tot >= 0.5:
                    break
            if not (m & rem & pos).any():
                break
            rules.append(tuple(conds))
            dl = self._total_dl(prefix + rules, stage, pos)
            if dl < best_dl:
                best_dl, best_k = dl, len(rules)
            if dl > best_dl + self.mdl_surplus:
                break
            rem &= ~m
        return rules[:best_k]

    def _optimize(self, rules, stage, pos):
        new_rules = []
        covered = np.zeros(self.N, dtype=bool)
        for i, r in enumerate(rules):
            rem = stage & ~covered
            grow, prune = self._split(rem, pos)
            replace = tuple(self._prune(self._grow(grow, pos), prune, pos))
            revision = tuple(self._prune(self._grow(grow, pos, start=r), prune, pos))
            cands = [r] + [c for c in (replace, revision) if c and c != r]
            tail = rules[i + 1:]
            best = min(cands, key=lambda c: self._total_dl(new_rules + [c] + tail, stage, pos))
            new_rules.append(best)
            covered |= self._cover(best)
        return new_rules

    def _reduce(self, rules, stage, pos):
        rules = list(rules)
        for i in range(len(rules) - 1, -1, -1):
            without = rules[:i] + rules[i + 1:]
            if self._total_dl(without, stage, pos) <= self._total_dl(rules, stage, pos):
                rules = without
        return rules

    def _rules_for_class(self, stage, pos):
        rules = self._irep(stage, pos)
        for _ in range(self.optimizations):
            rules = self._optimize(rules, stage, pos)
            rules += self._irep(stage, pos, prefix=rules)
            rules = self._reduce(rules, stage, pos)
        return rules

    # ---- public ------------------------------------------------------------
    def fit(self, df, target, num_cols, cat_cols):
        self.target = target
        y = df[target].astype(str).to_numpy()
        self.N = len(df)
        counts = Counter(y)
        order = sorted(counts, key=lambda c: (counts[c], c))      # rarest first
        self.conds = make_conditions(df, num_cols, cat_cols, self.max_thr)
        self.M = cond_matrix(df, self.conds)
        self.Mf = self.M.astype(np.float32)
        self.rng = np.random.default_rng(self.seed)
        self.n_attrs = len(num_cols) + len(cat_cols)
        self.choices = Counter(c.attr for c in self.conds)
        self._cache = {}

        stage = np.ones(self.N, dtype=bool)
        rules = []
        for cls in order[:-1]:
            pos = y == cls
            if not (stage & pos).any():
                continue
            cls_rules = self._rules_for_class(stage, pos)
            for r in cls_rules:                       # sequential coverage statistics
                m = self._cover(r) & stage
                p, n = int((m & pos).sum()), int((m & ~pos).sum())
                rules.append(Rule(_simplify([self.conds[i] for i in r]), cls, p + n, n))
                stage = stage & ~m
        self.default = order[-1]
        self.default_n = int(stage.sum())
        self.default_err = int((stage & (y != self.default)).sum())
        self.rules = rules
        for attr in ("M", "Mf", "_cache"):                # free memory
            setattr(self, attr, None)
        return self


# ==========================================================================
# CN2
# ==========================================================================
class CN2(RuleListModel):
    name = "CN2"

    def __init__(self, variant="ordered", beam=5, max_len=5, min_cover=1, alpha=0.05,
                 evaluator="laplace", max_thr=20):
        self.variant, self.beam, self.max_len = variant, beam, max_len
        self.min_cover, self.alpha, self.evaluator, self.max_thr = min_cover, alpha, evaluator, max_thr
        self.show_dist = variant == "unordered"

    def _evaluate(self, counts, n, target):
        k = counts.shape[1]
        if self.evaluator == "laplace":
            acc = (counts + 1.0) / (n[:, None] + k)
            val = acc.max(1) if target is None else acc[:, target]
        else:
            val = -entropy_rows(counts)
        return val, counts.argmax(1)

    def _find(self, active, target):
        Y, M, Mf, conds = self.Y, self.M, self.Mf, self.conds
        prior = Y[active].sum(0)
        ptot = prior.sum()
        if ptot == 0 or self.K < 2:
            return None
        prior_p = prior / ptot
        best, best_val = None, -np.inf
        star = [((), active)]
        seen = {frozenset()}
        for _ in range(self.max_len):
            cands = []
            for cs, mask in star:
                counts = Mf @ (Y * mask[:, None].astype(np.float32))
                n = counts.sum(1)
                valid = (n >= self.min_cover) & (n < mask.sum())
                if cs:
                    valid[list(cs)] = False
                for a in {conds[i].attr for i in cs if conds[i].op == "=="}:
                    valid &= ~self.eq_mask[a]
                if not valid.any():
                    continue
                val, lab = self._evaluate(counts, n, target)
                ok = valid.copy()
                if self.alpha < 1.0:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        exp = n[:, None] * prior_p[None, :]
                        t = np.where(counts > 0, counts * np.log(counts / np.maximum(exp, 1e-12)), 0.0).sum(1) * 2
                    ok &= t >= self.crit
                if target is not None:
                    ok &= lab == target
                if ok.any():
                    j = int(np.argmax(np.where(ok, val, -np.inf)))
                    if val[j] > best_val:
                        best_val, best = val[j], (cs + (j,), mask & M[j])
                idx = np.flatnonzero(valid)
                top = idx[np.argsort(-val[idx], kind="stable")][: self.beam]
                for j in top:
                    cands.append((float(val[j]), cs + (int(j),), mask))
            cands.sort(key=lambda t: -t[0])
            new_star = []
            for _, cs_new, pm in cands:
                key = frozenset(cs_new)
                if key in seen:
                    continue
                seen.add(key)
                new_star.append((cs_new, pm & M[cs_new[-1]]))
                if len(new_star) == self.beam:
                    break
            if not new_star:
                break
            star = new_star
        return best

    def fit(self, df, target, num_cols, cat_cols):
        self.target = target
        y = df[target].astype(str).to_numpy()
        self.classes = sorted(set(y))
        self.K = len(self.classes)
        cidx = {c: i for i, c in enumerate(self.classes)}
        yi = np.array([cidx[v] for v in y])
        N = len(df)
        self.Y = np.zeros((N, self.K), dtype=np.float32)
        self.Y[np.arange(N), yi] = 1.0
        self.conds = make_conditions(df, num_cols, cat_cols, self.max_thr)
        self.M = cond_matrix(df, self.conds)
        self.Mf = self.M.astype(np.float32)
        self.eq_mask = {}
        for a in set(c.attr for c in self.conds):
            self.eq_mask[a] = np.array([c.attr == a and c.op == "==" for c in self.conds])
        self.crit = chi2_critical(self.alpha, max(self.K - 1, 1)) if self.alpha < 1.0 else 0.0

        rules = []
        if self.variant == "ordered":
            active = np.ones(N, dtype=bool)
            while active.any():
                found = self._find(active, None)
                if found is None:
                    break
                cs, cov = found
                dist = self.Y[cov].sum(0)
                lab = int(dist.argmax())
                rules.append(Rule(_simplify([self.conds[i] for i in cs]), self.classes[lab],
                                  int(cov.sum()), int(cov.sum() - dist[lab]), dist))
                active &= ~cov
            base = self.Y[active].sum(0) if active.any() else self.Y.sum(0)
            d = int(base.argmax())
            self.default = self.classes[d]
            self.default_n = int(active.sum())
            self.default_err = int(active.sum() - (self.Y[active][:, d].sum() if active.any() else 0))
        else:
            for k in range(self.K):
                pos = self.Y[:, k] > 0
                remaining = pos.copy()
                while remaining.any():
                    found = self._find((~pos) | remaining, k)
                    if found is None:
                        break
                    cs, _ = found
                    cov = np.ones(N, dtype=bool)
                    for i in cs:
                        cov &= self.M[i]
                    if not (cov & remaining).any():
                        break
                    dist = self.Y[cov].sum(0)
                    rules.append(Rule(_simplify([self.conds[i] for i in cs]), self.classes[k],
                                      int(cov.sum()), int(cov.sum() - dist[k]), dist))
                    remaining &= ~cov
            tot = self.Y.sum(0)
            d = int(tot.argmax())
            self.default = self.classes[d]
            self.default_n, self.default_err = N, int(N - tot[d])
        self.rules = rules
        self.M = self.Mf = self.Y = None
        return self

    def predict(self, df):
        if self.variant == "ordered":
            return self.predict_list(df)
        scores = np.zeros((len(df), self.K))
        fired = np.zeros(len(df), dtype=bool)
        cache = {}
        for r in self.rules:
            m = r.mask(df, cache)
            scores[m] += r.dist
            fired |= m
        out = np.array(self.classes, dtype=object)[scores.argmax(1)]
        out[~fired] = self.default
        return out


# ==========================================================================
# PART
# ==========================================================================
def _simplify(path):
    """Merge redundant numeric conditions (<= keeps min, > keeps max)."""
    le, gt, rest = {}, {}, []
    for c in path:
        if c.op == "<=":
            le[c.attr] = min(le.get(c.attr, math.inf), c.val)
        elif c.op == ">":
            gt[c.attr] = max(gt.get(c.attr, -math.inf), c.val)
        else:
            rest.append(c)
    out, seen = [], set()
    for c in path:
        if c.op == "==":
            out.append(c)
        elif (c.attr, c.op) not in seen:
            seen.add((c.attr, c.op))
            out.append(Cond(c.attr, c.op, le[c.attr] if c.op == "<=" else gt[c.attr]))
    return out


class PART(RuleListModel):
    name = "PART"

    def __init__(self, min_obj=2, confidence=0.25, unpruned=False, mdl_correction=True):
        self.min_obj, self.cf = min_obj, confidence
        self.unpruned, self.mdl_correction = unpruned, mdl_correction

    # ---- tree -----------------------------------------------------------
    def _leaf(self, idx, expanded=True):
        return {"idx": idx, "counts": np.bincount(self.y[idx], minlength=self.K).astype(float),
                "leaf": True, "split": None, "children": None, "labels": None,
                "expanded": expanded}

    def _leaf_err(self, node):
        n = len(node["idx"])
        e = n - node["counts"].max()
        return e + add_errs(n, e, self.cf)

    def _best_split(self, idx, counts):
        n, K = len(idx), self.K
        base = float(entropy_rows(counts))
        Y1 = self.Y1[idx]
        cands = []
        for a, arr in self.num.items():
            v = arr[idx]
            order = np.argsort(v, kind="stable")
            vs = v[order]
            pos = np.flatnonzero(vs[:-1] < vs[1:])
            if pos.size == 0:
                continue
            pos = pos[(pos + 1 >= self.min_obj) & (n - pos - 1 >= self.min_obj)]
            if pos.size == 0:
                continue
            cum = np.cumsum(Y1[order], axis=0)
            left = cum[pos]
            right = counts - left
            nl = (pos + 1).astype(float)
            nr = n - nl
            gains = base - (nl * entropy_rows(left) + nr * entropy_rows(right)) / n
            j = int(np.argmax(gains))
            g = float(gains[j])
            if self.mdl_correction:
                g -= math.log2(len(pos)) / n
            if g <= 1e-9:
                continue
            si = float(entropy_rows(np.array([nl[j], nr[j]])))
            cands.append((g, g / si if si > 0 else 0.0,
                          {"kind": "num", "attr": a, "thr": float((vs[pos[j]] + vs[pos[j] + 1]) / 2)}))
        for a, codes_all in self.cat.items():
            L = len(self.levels[a])
            tab = np.bincount(codes_all[idx] * K + self.y[idx], minlength=L * K).reshape(L, K).astype(float)
            sizes = tab.sum(1)
            if (sizes >= self.min_obj).sum() < 2:
                continue
            present = sizes > 0
            g = base - float((sizes[present] * entropy_rows(tab[present])).sum()) / n
            if g <= 1e-9:
                continue
            si = float(entropy_rows(sizes[present]))
            cands.append((g, g / si if si > 0 else 0.0, {"kind": "cat", "attr": a}))
        if not cands:
            return None
        avg = sum(c[0] for c in cands) / len(cands)
        return max((c for c in cands if c[0] >= avg - 1e-9), key=lambda c: c[1])[2]

    def _subsets(self, idx, sp):
        a = sp["attr"]
        if sp["kind"] == "num":
            v = self.num[a][idx]
            return [(Cond(a, "<=", sp["thr"]), idx[v <= sp["thr"]]),
                    (Cond(a, ">", sp["thr"]), idx[v > sp["thr"]])]
        codes = self.cat[a][idx]
        return [(Cond(a, "==", self.levels[a][c]), idx[codes == c]) for c in np.unique(codes)]

    def _build(self, idx, force_leaf):
        node = self._leaf(idx)
        n = len(idx)
        if force_leaf or n < 2 * self.min_obj or node["counts"].max() == n:
            return node
        sp = self._best_split(idx, node["counts"])
        if sp is None:
            return node
        subs = self._subsets(idx, sp)
        children = [None] * len(subs)
        order = sorted(range(len(subs)), key=lambda j: float(entropy_rows(np.bincount(self.y[subs[j][1]], minlength=self.K))))
        all_leaves = True
        for j in order:
            children[j] = self._build(subs[j][1], False)
            if not children[j]["leaf"]:
                all_leaves = False
                break
        for j, ch in enumerate(children):
            if ch is None:           # never expanded: not a candidate for rule building
                children[j] = self._leaf(subs[j][1], expanded=False)
        node.update(leaf=False, split=sp, children=children, labels=[c for c, _ in subs])
        if all_leaves and not self.unpruned:
            if self._leaf_err(node) <= sum(self._leaf_err(c) for c in children) + 0.1:
                node.update(leaf=True, split=None, children=None, labels=None)
        return node

    def _best_leaf(self, node, path=()):
        if node["leaf"]:
            return (node, list(path)) if node["expanded"] else (None, None)
        best = (None, None)
        for ch, cond in zip(node["children"], node["labels"]):
            leaf, p = self._best_leaf(ch, path + (cond,))
            if leaf is not None and (best[0] is None or len(leaf["idx"]) > len(best[0]["idx"])):
                best = (leaf, p)
        return best

    # ---- public -------------------------------------------------------------
    def fit(self, df, target, num_cols, cat_cols):
        self.target = target
        ys = df[target].astype(str).to_numpy()
        self.classes = sorted(set(ys))
        self.K = len(self.classes)
        self.y = np.array([self.classes.index(v) for v in ys])
        N = len(df)
        self.Y1 = np.zeros((N, self.K))
        self.Y1[np.arange(N), self.y] = 1.0
        self.num, self.cat, self.levels = {}, {}, {}
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            med = s.median()
            self.num[c] = s.fillna(0.0 if pd.isna(med) else med).to_numpy(float)
        for c in cat_cols:
            s = df[c].astype(object).where(df[c].notna(), "?").astype(str)
            self.levels[c] = sorted(s.unique())
            lut = {v: i for i, v in enumerate(self.levels[c])}
            self.cat[c] = s.map(lut).to_numpy(int)

        remaining = np.arange(N)
        rules = []
        self.default, self.default_n, self.default_err = self.classes[int(np.bincount(self.y).argmax())], 0, 0
        while remaining.size:
            root = self._build(remaining, False)
            if root["leaf"]:
                d = int(root["counts"].argmax())
                self.default = self.classes[d]
                self.default_n = len(remaining)
                self.default_err = int(len(remaining) - root["counts"][d])
                break
            leaf, path = self._best_leaf(root)
            lab = int(leaf["counts"].argmax())
            n = len(leaf["idx"])
            rules.append(Rule(_simplify(path), self.classes[lab], n, int(n - leaf["counts"][lab])))
            remaining = np.setdiff1d(remaining, leaf["idx"])
        self.rules = rules
        self.num = self.cat = self.Y1 = None
        return self


# ==========================================================================
# ID3 (information gain tree -> rules)
# ==========================================================================
class ID3:
    name = "ID3"

    def __init__(self, bins=4, max_depth=None, min_gain=1e-9):
        self.bins, self.max_depth, self.min_gain = bins, max_depth, min_gain

    def _fit_bins(self, df, numeric_cols):
        self.edges = {}
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                self.edges[c] = np.array([])
                continue
            q = np.quantile(vals, np.linspace(0, 1, self.bins + 1)[1:-1])
            self.edges[c] = np.unique(q)

    @staticmethod
    def _label(edges, i):
        if len(edges) == 0:
            return "any"
        if i == 0:
            return f"<= {edges[0]:g}"
        if i == len(edges):
            return f"> {edges[-1]:g}"
        return f"({edges[i - 1]:g}, {edges[i]:g}]"

    def _transform(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.features:
            if c in self.edges:
                e = self.edges[c]
                col = pd.to_numeric(df[c], errors="coerce")
                idx = np.digitize(col.to_numpy(), e, right=True)
                labels = pd.Series([self._label(e, i) for i in idx], index=df.index)
                labels[col.isna()] = "?"
                out[c] = labels
            else:
                out[c] = df[c].astype(object).where(df[c].notna(), "?").astype(str)
        return out

    def fit(self, df, target, num_cols, cat_cols):
        self.target = target
        self.features = list(num_cols) + list(cat_cols)
        self._fit_bins(df, num_cols)
        X = self._transform(df).reset_index(drop=True)
        y = df[target].astype(str).reset_index(drop=True)
        self.tree = self._build(X, y, list(self.features), 0)
        return self

    def _build(self, X, y, feats, depth):
        counts = Counter(y)
        majority = counts.most_common(1)[0][0]
        node = {"counts": counts, "majority": majority}
        if len(counts) == 1 or not feats or (self.max_depth and depth >= self.max_depth):
            node["leaf"] = majority
            return node
        base = float(entropy_rows(np.array(list(counts.values()))))
        best, best_gain = None, self.min_gain
        for f in feats:
            rem = 0.0
            for _, idx in X.groupby(f, sort=False).groups.items():
                sub = np.array(list(Counter(y.loc[idx]).values()))
                rem += len(idx) / len(y) * float(entropy_rows(sub))
            if base - rem > best_gain:
                best, best_gain = f, base - rem
        if best is None:
            node["leaf"] = majority
            return node
        node["attr"] = best
        node["children"] = {}
        rest = [f for f in feats if f != best]
        for val, idx in X.groupby(best, sort=False).groups.items():
            node["children"][val] = self._build(X.loc[idx], y.loc[idx], rest, depth + 1)
        return node

    def predict(self, df):
        X = self._transform(df)
        preds = []
        for _, row in X.iterrows():
            node = self.tree
            while "leaf" not in node:
                child = node["children"].get(row[node["attr"]])
                if child is None:
                    break
                node = child
            preds.append(node.get("leaf", node["majority"]))
        return np.array(preds, dtype=object)

    def rules_text(self):
        out = []

        def walk(node, conds):
            if "leaf" in node:
                n = sum(node["counts"].values())
                ok = node["counts"][node["leaf"]]
                cond = " AND ".join(conds) if conds else "TRUE"
                out.append(f"IF {cond} THEN {self.target} = {node['leaf']}   [covers {n}, errors {n - ok}]")
                return
            for val, child in node["children"].items():
                walk(child, conds + [f"{node['attr']} = {val}"])

        walk(self.tree, [])
        return "\n".join(out)


# ==========================================================================
# Cross-validation (stratified, no sklearn)
# ==========================================================================
def stratified_folds(y, k, seed):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    offset = 0
    for cls in sorted(set(y)):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[(j + offset) % k].append(i)
        offset += len(idx)
    return [np.array(sorted(f)) for f in folds]


def cv_accuracy(make_model, df, target, num_cols, cat_cols, k, seed):
    y = df[target].astype(str).to_numpy()
    k = min(k, min(Counter(y).values()))
    if k < 2:
        return float("nan")
    correct = 0
    for test in stratified_folds(y, k, seed):
        mask = np.ones(len(df), dtype=bool)
        mask[test] = False
        model = make_model().fit(df[mask].reset_index(drop=True), target, num_cols, cat_cols)
        pred = model.predict(df.iloc[test].reset_index(drop=True))
        correct += int((pred == y[test]).sum())
    return 100.0 * correct / len(df)


# ==========================================================================
# Main
# ==========================================================================
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="input CSV file")
    p.add_argument("--algo", nargs="+", default=["all"], choices=ALGOS + ["all"])
    p.add_argument("--class-col", help="class column (default: last column)")
    p.add_argument("--sep", default=",")
    p.add_argument("--drop", nargs="*", default=[], help="columns to ignore (e.g. IDs)")
    p.add_argument("--categorical", nargs="*", default=[], help="numeric-looking columns to treat as categorical")
    p.add_argument("--cv", type=int, default=10, help="CV folds (0 = skip)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", help="save rules to <out-dir>/<algo>_rules.txt")
    p.add_argument("--max-thresholds", type=int, default=20,
                   help="max numeric split points per column for JRip/CN2")
    g = p.add_argument_group("JRip")
    g.add_argument("--jrip-folds", type=int, default=3)
    g.add_argument("--jrip-min-weight", type=float, default=2.0)
    g.add_argument("--jrip-optimizations", type=int, default=2)
    g = p.add_argument_group("PART")
    g.add_argument("--part-min-obj", type=int, default=2)
    g.add_argument("--part-confidence", type=float, default=0.25)
    g.add_argument("--part-unpruned", action="store_true")
    g = p.add_argument_group("ID3")
    g.add_argument("--id3-bins", type=int, default=4)
    g.add_argument("--id3-max-depth", type=int, default=None)
    g = p.add_argument_group("CN2")
    g.add_argument("--cn2-variant", choices=["ordered", "unordered"], default="ordered")
    g.add_argument("--cn2-beam-width", type=int, default=5)
    g.add_argument("--cn2-min-covered", type=int, default=1)
    g.add_argument("--cn2-max-rule-length", type=int, default=5)
    g.add_argument("--cn2-alpha", type=float, default=0.05, help="significance level (1.0 = no test)")
    g.add_argument("--cn2-evaluator", choices=["laplace", "entropy"], default="laplace")
    a = p.parse_args()

    makers = {
        "jrip": lambda: RIPPER(a.jrip_folds, a.jrip_min_weight, a.jrip_optimizations, a.seed, a.max_thresholds),
        "id3": lambda: ID3(a.id3_bins, a.id3_max_depth),
        "cn2": lambda: CN2(a.cn2_variant, a.cn2_beam_width, a.cn2_max_rule_length, a.cn2_min_covered,
                           a.cn2_alpha, a.cn2_evaluator, a.max_thresholds),
        "part": lambda: PART(a.part_min_obj, a.part_confidence, a.part_unpruned),
    }
    algos = ALGOS if "all" in a.algo else a.algo
    df, target = load_data(a.csv, a.sep, a.class_col, a.drop)
    num_cols, cat_cols = split_columns(df, target, set(a.categorical))
    print(f"Loaded {len(df)} rows, {len(num_cols)} numeric + {len(cat_cols)} categorical "
          f"features, class='{target}' ({df[target].nunique()} classes)\n")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)

    for algo in algos:
        print("=" * 72, algo.upper(), "=" * 72, sep="\n")
        model = makers[algo]().fit(df, target, num_cols, cat_cols)
        rules = model.rules_text()
        print(rules)
        acc = float("nan")
        if a.cv:
            acc = cv_accuracy(makers[algo], df, target, num_cols, cat_cols, a.cv, a.seed)
            print(f"\n{a.cv}-fold CV accuracy: {acc:.2f}%")
        print()
        if a.out_dir:
            with open(os.path.join(a.out_dir, f"{algo}_rules.txt"), "w", encoding="utf-8") as f:
                f.write(rules + (f"\n\n# {a.cv}-fold CV accuracy: {acc:.2f}%\n" if a.cv else "\n"))


if __name__ == "__main__":
    main()
