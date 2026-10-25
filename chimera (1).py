"""
chimera.py
----------
Hierarchy-based scalarization of several objectives into ONE merit value
(to be minimized), after

    Hase, Roch, Aspuru-Guzik, "Chimera: enabling hierarchy based
    multi-objective optimization for self-driving laboratories",
    Chem. Sci., 2018, 9, 7642.

Objectives are listed in PRIORITY ORDER (index 0 = most important). Each
objective k has a threshold ("tolerance") that decides whether a point is
good enough on that objective:

  * absolute=True : the threshold is the number you give, in the
                    objective's own units and sense (e.g. goal='max',
                    tolerance=95 means "value must be >= 95").
  * absolute=False: the threshold is a fraction of the observed range of
                    that objective, measured only among the points that
                    already passed every higher-priority objective
                    (tolerance 0.2 = "may be 20% worse than the best").

SCORING (what this file guarantees)
-----------------------------------
Every point is classified by the FIRST objective it fails:

    fails objective 0           -> worst group
    passes 0, fails objective 1 -> next group
    ...
    passes every objective      -> best group

so a point that fails a higher-priority objective can NEVER outrank a
point that fails only lower-priority ones (strict hierarchy). Inside a
group, points are ordered by how far they miss the threshold of the
objective that decided their group. Points that pass everything are
ordered by the top-priority objective (smaller is better after sign
flip).

The merit lies in [0, 1] (0 = best). The groups occupy equal slices of
that interval: with m objectives, "passes everything" is [0, 1/(m+1)],
"first failure at objective m-1" is the next slice, ... and "first
failure at objective 0" is the last slice.

NOTE: this replaces an earlier shift-based implementation whose
arithmetic could rank a point that failed objective 0 above one that
passed it when only a handful of points existed (a point sitting exactly
on a threshold was treated as passing in one place and failing in
another). The strict group structure above removes that class of error.
"""

import numpy as np


class Chimera:
    """
    Parameters
    ----------
    goals : list of {'min', 'max'}
    tolerances : list of float   (meaning depends on `absolutes`, see above)
    absolutes : list of bool, optional (default all False)
    """

    _EPS = 1e-12

    def __init__(self, goals, tolerances, absolutes=None):
        self.n_obj = len(goals)
        if self.n_obj == 0:
            raise ValueError("need at least one objective")
        if len(tolerances) != self.n_obj:
            raise ValueError("tolerances must have the same length as goals")
        self.goals = list(goals)
        for g in self.goals:
            if g not in ("min", "max"):
                raise ValueError("goal must be 'min' or 'max'")
        self.tolerances = np.array(tolerances, dtype=float)
        if not np.all(np.isfinite(self.tolerances)):
            raise ValueError("tolerances must be finite numbers")
        self.absolutes = list(absolutes) if absolutes is not None else [False] * self.n_obj
        if len(self.absolutes) != self.n_obj:
            raise ValueError("absolutes must have the same length as goals")
        for tol, ab in zip(self.tolerances, self.absolutes):
            if not ab and not (0.0 <= tol <= 1.0):
                raise ValueError("a relative tolerance must lie in [0, 1]")
        self._signs = np.array([1.0 if g == "min" else -1.0 for g in self.goals])
        self.last_ = None   # diagnostics of the most recent scalarize() call

    # ------------------------------------------------------------------
    def scalarize(self, objs):
        """
        Raw merit in [0, 1] for each row of `objs` (shape (n, n_obj), original
        units and sense, columns in priority order). Lower is always better.
        """
        objs = np.asarray(objs, dtype=float)
        if objs.ndim != 2 or objs.shape[1] != self.n_obj:
            raise ValueError(f"expected an array of shape (n, {self.n_obj}), got {objs.shape}")
        if not np.all(np.isfinite(objs)):
            raise ValueError("objective values must be finite (no NaN/inf)")
        n, m = objs.shape
        eps = self._EPS
        signs = self._signs

        F = objs * signs                                   # everything: smaller is better
        fmin, fmax = F.min(axis=0), F.max(axis=0)
        span = np.where(fmax > fmin, fmax - fmin, 1.0)
        Fn = (F - fmin) / span                             # each column mapped to [0, 1]

        # thresholds, objective by objective, in normalized units
        thr = np.empty(m)
        domain = np.ones(n, dtype=bool)                    # points that passed all tiers so far
        for k in range(m):
            col = Fn[:, k]
            if self.absolutes[k]:
                thr[k] = (self.tolerances[k] * signs[k] - fmin[k]) / span[k]
            else:
                lo, hi = col[domain].min(), col[domain].max()
                thr[k] = lo + self.tolerances[k] * (hi - lo)
            passed = domain & (col <= thr[k] + eps)
            if passed.any():
                domain = passed

        fails = Fn > (thr[None, :] + eps)                  # (n, m) True where objective failed
        any_fail = fails.any(axis=1)
        first_fail = np.where(any_fail, fails.argmax(axis=1), m)

        w = 1.0 / (m + 1)
        merit = np.empty(n)

        bad = np.where(any_fail)[0]
        if bad.size:
            f = first_fail[bad]
            room = np.maximum(1.0 - thr[f], eps)
            excess = np.clip((Fn[bad, f] - thr[f]) / room, 0.0, 1.0)
            merit[bad] = (m - f + excess) * w              # f=0 -> top slice, f=m-1 -> second slice

        good = np.where(~any_fail)[0]
        if good.size:
            denom = max(thr[0], eps)
            merit[good] = np.clip(Fn[good, 0] / denom, 0.0, 1.0) * w   # bottom slice [0, w]

        # thresholds back in the user's original units, for reporting
        thr_original = (thr * span + fmin) * signs
        self.last_ = {
            "thresholds": thr_original,
            "first_fail": first_fail,           # m means "passes everything"
            "fails": fails,
        }
        return merit

    # ------------------------------------------------------------------
    def scalarize_and_transform(self, objs, transform="sqrt"):
        """
        Scalarize, stretch to exactly [0, 1], then apply a monotone transform
        that sharpens the region near the best point (Gryffin's default is sqrt).
        Returns the target the surrogate is trained on (lower = better).
        """
        chi = self.scalarize(objs)
        lo, hi = chi.min(), chi.max()
        norm = (chi - lo) / (hi - lo) if hi > lo else np.zeros_like(chi)
        if transform == "sqrt":
            return np.sqrt(norm)
        if transform == "cbrt":
            return np.cbrt(norm)
        if transform == "square":
            return norm ** 2
        if transform is None or transform == "identity":
            return norm
        raise ValueError(f"unknown transform '{transform}'")
