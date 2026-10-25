"""
tests.py -- run with:  python3 tests.py
Each test prints PASS or raises. No test framework needed.
"""

import os
import tempfile
import numpy as np

from chimera import Chimera
from targets import TargetSet
from data_processing import (ParameterSpace, prepare_observations, pooled_replicate_sd,
                             load_observations_csv, save_observations_csv)
from surrogate import KernelDensitySurrogate
from optimizer import SimpleGryffin

RESULTS = []


def test(fn):
    try:
        fn()
        print(f"PASS  {fn.__name__}")
        RESULTS.append(True)
    except Exception:
        print(f"FAIL  {fn.__name__}")
        RESULTS.append(False)
        raise
    return fn


# ----------------------------------------------------------------------
# Chimera
# ----------------------------------------------------------------------
@test
def chimera_hierarchy_is_strict():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, m = int(rng.integers(3, 40)), int(rng.integers(1, 7))
        Y = rng.normal(size=(n, m)) * rng.uniform(1, 50, size=m) + rng.uniform(-20, 20, size=m)
        goals = list(rng.choice(["min", "max"], size=m))
        absolutes = list(rng.random(m) < 0.5)
        tols = [float(np.median(Y[:, k])) if a else float(rng.uniform(0.05, 0.6))
                for k, a in enumerate(absolutes)]
        ch = Chimera(goals, tols, absolutes)
        merit = ch.scalarize(Y)
        assert merit.min() >= 0 and merit.max() <= 1
        ff = ch.last_["first_fail"]
        for i in range(n):
            for j in range(n):
                if ff[i] > ff[j]:          # i fails later in the hierarchy than j
                    assert merit[i] < merit[j], "hierarchy violated"


@test
def chimera_absolute_limits_pass_fail():
    # monomer >= 95 (max), hmw <= 3 (min)
    Y = np.array([[96.0, 2.0], [95.0, 3.0], [94.9, 1.0], [97.0, 3.1]])
    ch = Chimera(["max", "min"], [95.0, 3.0], [True, True])
    m = ch.scalarize(Y)
    ff = ch.last_["first_fail"]
    assert list(ff) == [2, 2, 0, 1], ff          # rows 0,1 pass everything (boundary inclusive)
    assert m[0] < m[3] < m[2]                    # passes all < fails hmw only < fails monomer
    assert m[1] < m[3]
    assert m[0] <= m[1]                          # among passers: higher monomer first


@test
def chimera_degenerate_inputs():
    ch = Chimera(["min"], [0.2])
    assert np.all(ch.scalarize(np.array([[1.0], [1.0], [1.0]])) == 0)
    assert ch.scalarize_and_transform(np.array([[1.0], [1.0]])).tolist() == [0.0, 0.0]
    for bad in (np.array([[np.nan], [1.0]]), np.array([1.0, 2.0])):
        try:
            ch.scalarize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad input accepted")


# ----------------------------------------------------------------------
# Targets with acceptance limits
# ----------------------------------------------------------------------
@test
def targets_translation_and_status():
    ts = TargetSet([
        {"name": "monomer", "lower": 95.0},
        {"name": "hmw", "upper": 3.0},
        {"name": "hm", "lower": 2.0, "upper": 8.0},
    ])
    rows = [{"monomer": 96, "hmw": 2, "hm": 5},
            {"monomer": 95, "hmw": 3, "hm": 2},       # all exactly on a limit -> pass
            {"monomer": 94, "hmw": 4, "hm": 9}]
    st = ts.limit_status(rows)
    assert st.tolist() == [[True] * 3, [True] * 3, [False] * 3]
    cols = ts.matrix(rows)
    assert cols[:, 2].tolist() == [0.0, 3.0, 4.0]       # |y - 5|
    ch = ts.chimera()
    assert ch.goals == ["max", "min", "min"]
    assert list(ch.tolerances) == [95.0, 3.0, 3.0]       # half-width of [2, 8]


@test
def window_centre_and_flat_accept_the_same_values():
    y = np.linspace(-2, 12, 141)
    rows = [{"hm": float(v)} for v in y]
    for margin in (0.0, 0.5, 1.0):
        passes = {}
        for mode in ("centre", "flat"):
            ts = TargetSet([{"name": "hm", "lower": 2.0, "upper": 8.0,
                             "margin": margin, "window_mode": mode}])
            ch = ts.chimera()
            ch.scalarize(ts.matrix(rows))
            passes[mode] = ch.last_["first_fail"] == 1       # 1 == passes the only target
        assert np.array_equal(passes["centre"], passes["flat"])
        expected = (y >= 2 + margin - 1e-9) & (y <= 8 - margin + 1e-9)
        assert np.array_equal(passes["centre"], expected)


@test
def targets_validation_errors():
    bad = [
        [],
        [{"name": "a", "lower": 5, "upper": 1}],
        [{"name": "a", "lower": 1, "upper": 4, "margin": 2}],        # window disappears
        [{"name": "a", "lower": 1}, {"name": "a", "upper": 2}],       # duplicate name
        [{"name": "a"}],                                              # no limits, no goal
        [{"name": "a", "lower": 1, "margin": -1}],
    ]
    for spec in bad:
        try:
            TargetSet(spec)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid targets: {spec}")


@test
def targets_without_limits_still_work():
    ts = TargetSet([{"name": "yield", "goal": "max", "tolerance": 0.2},
                    {"name": "cost", "goal": "min", "tolerance": 0.2}])
    Y = ts.matrix([{"yield": 80, "cost": 5}, {"yield": 60, "cost": 3}, {"yield": 79, "cost": 9}])
    m = ts.chimera().scalarize(Y)
    assert m[0] < m[2] and m[1] > m[0]


# ----------------------------------------------------------------------
# Data preparation, replicates
# ----------------------------------------------------------------------
def _space2():
    return ParameterSpace([{"name": "x1", "low": 0, "high": 10}, {"name": "x2", "low": 0, "high": 1}])


@test
def replicates_are_averaged_and_counted():
    sp = _space2()
    rows = [{"x1": 5.0, "x2": 0.5, "y": 10.0}, {"x1": 5.0, "x2": 0.5, "y": 12.0},
            {"x1": 5.0, "x2": 0.5, "y": 14.0}, {"x1": 1.0, "x2": 0.1, "y": 3.0},
            {"x1": 2.0, "x2": 0.2, "y": 4.0}, {"x1": 2.0, "x2": 0.2, "y": 6.0}]
    out, rep = prepare_observations(rows, sp, ["y"], verbose=False)
    assert len(out) == 3 and rep["n_unique"] == 3 and rep["n_replicated_conditions"] == 2
    by_x1 = {r["x1"]: r for r in out}
    assert by_x1[5.0]["y"] == 12.0 and by_x1[5.0]["_n_rep"] == 3
    assert by_x1[2.0]["y"] == 5.0 and by_x1[1.0]["_n_rep"] == 1
    # pooled SD: group A deviations (-2,0,2) -> 8 ; group B (-1,1) -> 2 ; dof 2 + 1 = 3
    assert abs(rep["replicate_sd"]["y"] - np.sqrt(10 / 3)) < 1e-12
    assert abs(pooled_replicate_sd(rows, sp, ["y"])["y"] - np.sqrt(10 / 3)) < 1e-12
    first, _ = prepare_observations(rows, sp, ["y"], replicates="first", verbose=False)
    assert {r["x1"]: r["y"] for r in first}[5.0] == 10.0


@test
def bad_rows_are_dropped_and_reported():
    sp = _space2()
    rows = [{"x1": 1.0, "x2": 0.1, "y": 1.0},
            {"x1": 2.0, "x2": 0.2},                          # target missing
            {"x1": "abc", "x2": 0.2, "y": 1.0},              # not a number
            {"x1": 3.0, "x2": 0.3, "y": float("nan")},       # NaN target
            {"x1": 99.0, "x2": 0.3, "y": 2.0}]               # outside bounds: kept, reported
    out, rep = prepare_observations(rows, sp, ["y"], verbose=False)
    assert len(out) == 2 and rep["dropped_missing"] == 1 and rep["dropped_invalid"] == 2
    assert rep["rows_outside_bounds"] == 1


@test
def csv_roundtrip():
    rows = [{"a": 1.5, "b": "x"}, {"a": 2.0, "b": "y"}]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.csv")
        save_observations_csv(rows, p)
        back = load_observations_csv(p)
    assert back == rows


# ----------------------------------------------------------------------
# Surrogate
# ----------------------------------------------------------------------
@test
def surrogate_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    X, y = rng.uniform(size=(5, 4)), rng.uniform(size=5)
    s = KernelDensitySurrogate(alpha_coeff=0.5).fit(X, y)
    x0 = X[0] + 0.02 * rng.normal(size=4)
    for lam in (-1.0, 0.0, 0.7):
        _, g = s.value_and_grad_acquisition(x0, lam)
        num = np.zeros(4)
        for i in range(4):
            e = np.zeros(4)
            e[i] = 1e-6
            num[i] = (s.value_and_grad_acquisition(x0 + e, lam)[0]
                      - s.value_and_grad_acquisition(x0 - e, lam)[0]) / 2e-6
        assert np.max(np.abs(g - num)) < 1e-7


@test
def surrogate_predict_is_stable_far_from_data():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 0.2, size=(20, 7))
    y = rng.uniform(0.3, 0.9, size=20)
    s = KernelDensitySurrogate().fit(X, y)             # very narrow kernels
    far = np.full((1, 7), 1.0)
    p = s.predict(far)[0]
    assert np.isfinite(p) and y.min() - 1e-12 <= p <= y.max() + 1e-12   # not collapsing to 0
    assert np.allclose(s.predict(X[:3]), y[:3], atol=1e-9)              # reproduces observed values


# ----------------------------------------------------------------------
# End to end
# ----------------------------------------------------------------------
PARAMS = [{"name": f"x{i + 1}", "low": 0.0, "high": 10.0} for i in range(5)]
TARGETS = [
    {"name": "monomer", "lower": 95.0},
    {"name": "hmw", "upper": 3.0},
    {"name": "high_mannose", "lower": 2.0, "upper": 8.0, "margin": 0.5},
    {"name": "acidic", "upper": 25.0},
]


def _experiment(p, rng):
    x = np.array([p[f"x{i + 1}"] for i in range(5)]) / 10
    d2 = float(np.sum((x - 0.4) ** 2))
    return {"monomer": 99 - 8 * d2 + rng.normal(0, 0.1),
            "hmw": 1.5 + 5 * d2 + rng.normal(0, 0.05),
            "high_mannose": 5 + 12 * (x[0] - 0.4) + rng.normal(0, 0.2),
            "acidic": 20 + 20 * d2 + rng.normal(0, 0.2)}


def _make_history(n=12, seed=0, replicate_first=2):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        p = {f"x{j + 1}": round(float(rng.uniform(0, 10)), 3) for j in range(5)}
        for _ in range(replicate_first if i == 0 else 1):
            rows.append({**p, **_experiment(p, rng)})
    return rows


@test
def end_to_end_with_replicates_and_limits():
    hist = _make_history()
    opt = SimpleGryffin(PARAMS, TARGETS, {"random_seed": 3, "verbose": False,
                                          "num_random_samples": 500, "adam_steps": 20,
                                          "num_restarts": 4})
    props = opt.recommend(hist, num_proposals=3)
    assert 1 <= len(props) <= 3
    for p in props:
        assert set(p) == {f"x{i + 1}" for i in range(5)}
        assert all(0.0 <= v <= 10.0 and np.isfinite(v) for v in p.values())
    assert opt.last_report_["n_unique"] == 12 and opt.last_report_["n_replicated_conditions"] == 1
    again = SimpleGryffin(PARAMS, TARGETS, {"random_seed": 3, "verbose": False,
                                            "num_random_samples": 500, "adam_steps": 20,
                                            "num_restarts": 4}).recommend(hist, num_proposals=3)
    assert again == props                                  # reproducible with a seed


@test
def cold_start_and_unusable_data_fall_back_to_random():
    opt = SimpleGryffin(PARAMS, TARGETS, {"random_seed": 0, "verbose": False})
    for obs in ([], [{"x1": 1.0}]):                        # nothing / only malformed rows
        props = opt.recommend(obs, num_proposals=4)
        assert len(props) == 4
        assert all(0 <= v <= 10 for p in props for v in p.values())


@test
def status_and_diagnose_run():
    hist = _make_history(n=10, replicate_first=3)
    opt = SimpleGryffin(PARAMS, TARGETS, {"random_seed": 0, "verbose": False})
    st = opt.status(hist, show=False)
    assert len(st) == 10 and st[0]["merit"] <= st[-1]["merit"]
    assert {"meets_all_limits", "failed_targets", "_n_rep"} <= set(st[0])
    res = opt.diagnose(hist)
    assert len(res["results"]) == 7 and np.isfinite(res["baseline_rmse"])


@test
def mixed_input_kinds_still_supported():
    params = [{"name": "t", "low": 20.0, "high": 150.0},
              {"name": "plates", "kind": "discrete_range", "low": 5, "high": 20},
              {"name": "media", "kind": "categorical", "options": ["A", "B", "C"]}]
    targets = [{"name": "monomer", "lower": 95.0}, {"name": "hmw", "upper": 3.0}]
    rng = np.random.default_rng(0)
    hist = []
    for _ in range(10):
        r = {"t": float(rng.uniform(20, 150)), "plates": int(rng.integers(5, 21)),
             "media": str(rng.choice(["A", "B", "C"]))}
        r["monomer"] = 96 + rng.normal(0, 1) + (1.5 if r["media"] == "B" else 0)
        r["hmw"] = 2.5 + rng.normal(0, 0.5)
        hist.append(r)
    opt = SimpleGryffin(params, targets, {"random_seed": 1, "verbose": False,
                                          "num_random_samples": 400, "adam_steps": 15,
                                          "num_restarts": 3})
    for p in opt.recommend(hist, num_proposals=2):
        assert p["media"] in ("A", "B", "C") and 5 <= p["plates"] <= 20
        assert float(p["plates"]).is_integer() and 20 <= p["t"] <= 150


@test
def configuration_errors_are_loud():
    for kwargs in (
        dict(parameters=PARAMS, targets=TARGETS, general={"typo_setting": 1}),
        dict(parameters=PARAMS, targets=[{"name": "x1", "lower": 1.0}]),   # clashes with an input
    ):
        try:
            SimpleGryffin(**kwargs)
        except ValueError:
            continue
        raise AssertionError("bad configuration accepted")


@test
def auto_kernel_width_widens_kernels_for_sparse_smooth_data():
    from diagnostics import choose_alpha_coeff, leave_one_out
    rng = np.random.default_rng(0)
    U = rng.uniform(size=(27, 7))
    d2 = ((U - 0.45) ** 2).sum(axis=1)
    d2 = (d2 - d2.min()) / (d2.max() - d2.min())            # smooth merit, 27 points in 7-D
    coeff, info = choose_alpha_coeff(U, d2)
    assert coeff < 12.0                                      # the paper default is too narrow here
    best = min(r["rmse"] for r in info["results"])
    assert best <= [r for r in info["results"] if r["alpha_coeff"] == 12.0][0]["rmse"]
    assert choose_alpha_coeff(U[:5], d2[:5])[0] == 12.0      # too little data -> default


@test
def auto_is_used_by_recommend_and_fixed_value_is_respected():
    hist = _make_history(n=12)
    kw = {"random_seed": 3, "verbose": False, "num_random_samples": 300,
          "adam_steps": 10, "num_restarts": 3}
    a = SimpleGryffin(PARAMS, TARGETS, kw)
    a.recommend(hist, num_proposals=1)
    assert a.last_alpha_coeff_ in (12.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01)
    f = SimpleGryffin(PARAMS, TARGETS, {**kw, "kernel_alpha_coeff": 5.0})
    f.recommend(hist, num_proposals=1)
    assert f.last_alpha_coeff_ == 5.0
    for bad in (0, -1, "wide"):
        try:
            SimpleGryffin(PARAMS, TARGETS, {"kernel_alpha_coeff": bad})
        except ValueError:
            continue
        raise AssertionError("bad kernel_alpha_coeff accepted")


if __name__ == "__main__":
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} passed")
