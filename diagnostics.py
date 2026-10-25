"""
diagnostics.py
--------------
Leave-one-out check of the kernel surrogate on YOUR data: hide one batch,
predict its merit from the others, compare with what it really scored.

Why: the kernel width is set by tau = alpha_coeff * n^2 (default
alpha_coeff = 12). With few batches that makes each kernel very narrow, so
the surrogate says little about points that are not right next to an
experiment. This table shows which alpha_coeff predicts held-out batches best,
and whether the surrogate beats the trivial "always predict the average"
guess at all.

Caveat: the merit of every batch comes from a min-max scale that includes the
held-out batch, so this is a diagnostic, not a strict out-of-sample estimate.
"""

import numpy as np

from surrogate import KernelDensitySurrogate


def _ranks(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def _spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


DEFAULT_GRID = (12.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)


def _verdict(rmse, base_rmse, spearman):
    if rmse < 0.85 * base_rmse and spearman == spearman and spearman >= 0.3:
        return "useful"
    if rmse < base_rmse:
        return "marginal"
    return "worse than predicting the mean"


def choose_alpha_coeff(U, merit, grid=DEFAULT_GRID):
    """
    Pick the kernel-width coefficient from leave-one-out error: take every
    width whose RMSE is within 2% of the best one and use the WIDEST of them
    (smallest coefficient). When widths perform about equally, the smoother
    surrogate is the safer choice and the pick does not flip from one round
    to the next. Returns (coeff, info), info being the leave_one_out() result.
    With fewer than 6 unique conditions there is too little to validate on,
    so the Phoenics/Gryffin default (12) is returned.
    """
    if len(merit) < 6:
        return 12.0, None
    info = leave_one_out(U, merit, alpha_coeffs=grid, verbose=False)
    best_rmse = min(r["rmse"] for r in info["results"])
    near_best = [r for r in info["results"] if r["rmse"] <= best_rmse * 1.02]
    chosen = min(near_best, key=lambda r: r["alpha_coeff"])
    return chosen["alpha_coeff"], info


def leave_one_out(U, merit, alpha_coeffs=DEFAULT_GRID, verbose=True):
    """
    U      : (n, d) encoded inputs (unit scale)
    merit  : (n,)   scalar merit (lower = better)
    Returns a list of dicts, one per alpha_coeff, plus a 'baseline' entry.
    """
    U = np.asarray(U, dtype=float)
    merit = np.asarray(merit, dtype=float)
    n = len(merit)
    if n < 4:
        raise ValueError("need at least 4 unique conditions for a leave-one-out check")

    base_pred = np.array([np.delete(merit, i).mean() for i in range(n)])
    base_rmse = float(np.sqrt(np.mean((base_pred - merit) ** 2)))

    results = []
    for coef in alpha_coeffs:
        pred = np.empty(n)
        for i in range(n):
            keep = np.arange(n) != i
            s = KernelDensitySurrogate(alpha_coeff=coef).fit(U[keep], merit[keep])
            pred[i] = s.predict(U[i:i + 1])[0]
        results.append({
            "alpha_coeff": coef,
            "kernel_sd": float(1.0 / np.sqrt(coef * (n - 1) ** 2)),
            "rmse": float(np.sqrt(np.mean((pred - merit) ** 2))),
            "spearman": _spearman(pred, merit),
        })

    if verbose:
        print(f"Leave-one-out on {n} unique conditions "
              f"(baseline 'predict the mean' RMSE = {base_rmse:.3f})")
        print(f"  {'alpha_coeff':>11}  {'kernel sd':>9}  {'RMSE':>6}  {'rank corr':>9}  verdict")
        for r in results:
            print(f"  {r['alpha_coeff']:>11g}  {r['kernel_sd']:>9.3f}  {r['rmse']:>6.3f}  "
                  f"{r['spearman']:>9.2f}  {_verdict(r['rmse'], base_rmse, r['spearman'])}")
    return {"baseline_rmse": base_rmse, "results": results}
