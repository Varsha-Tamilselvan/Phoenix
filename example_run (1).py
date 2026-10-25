"""
example_run.py
--------------
End-to-end demo on SYNTHETIC data shaped like your problem: 7 continuous
inputs, 9 quality attributes with acceptance limits (lower-only, upper-only
and one two-sided window), and replicate runs. Replace `run_experiment` with
your real data / lab call, and edit PARAMETERS and TARGETS.

    python3 example_run.py
"""

import numpy as np

from optimizer import SimpleGryffin
from data_processing import pooled_replicate_sd

# ----------------------------------------------------------------------
# 1) Inputs: use the ranges your DoE actually covered
# ----------------------------------------------------------------------
PARAMETERS = [{"name": f"x{i + 1}", "low": 0.0, "high": 10.0} for i in range(7)]

# ----------------------------------------------------------------------
# 2) Targets IN PRIORITY ORDER (first = most important), with their
#    acceptance limits. Names must match your data columns.
# ----------------------------------------------------------------------
TARGETS = [
    {"name": "monomer_pct",    "lower": 95.0},
    {"name": "hmw_pct",        "upper": 3.0},
    {"name": "lmw_pct",        "upper": 2.0},
    {"name": "total_acidic",   "upper": 25.0},
    {"name": "total_basic",    "upper": 20.0},
    {"name": "main_peak",      "lower": 55.0},
    {"name": "high_mannose",   "lower": 2.0, "upper": 8.0, "margin": 0.5},   # two-sided
    {"name": "galactosylation", "lower": 5.0},
    {"name": "fucosylation",   "lower": 80.0},
]

GENERAL = {
    "sampling_strategies": 2,     # proposals per call (lambda = +1 exploit ... -1 explore)
    "kernel_alpha_coeff": "auto", # kernel width picked by leave-one-out; or a fixed number (12 = paper)
    "random_seed": 7,
}

# ----------------------------------------------------------------------
# 3) Stand-in for the real experiment
# ----------------------------------------------------------------------
_rng = np.random.default_rng(0)


def run_experiment(p):
    x = np.array([p[f"x{i + 1}"] for i in range(7)]) / 10.0
    d2 = float(np.sum((x - 0.45) ** 2))
    n = lambda s: _rng.normal(0, s)
    return {
        "monomer_pct": 99.0 - 9 * d2 + n(0.15),
        "hmw_pct": 1.2 + 6 * d2 + n(0.08),
        "lmw_pct": 0.8 + 3 * d2 + n(0.05),
        "total_acidic": 19 + 22 * d2 + n(0.4),
        "total_basic": 14 + 12 * d2 + n(0.3),
        "main_peak": 62 - 25 * d2 + n(0.5),
        "high_mannose": 5.0 + 14 * (x[0] - 0.45) + n(0.3),
        "galactosylation": 9 - 12 * d2 + n(0.3),
        "fucosylation": 88 - 20 * d2 + n(0.5),
    }


def main():
    opt = SimpleGryffin(PARAMETERS, TARGETS, GENERAL)
    print("How your targets are interpreted:")
    print(opt.targets.explain(), "\n")

    # --- history: 27 unique batches, the first condition run 3 times -----------
    observations = []
    for i in range(27):
        p = {f"x{j + 1}": round(float(_rng.uniform(0, 10)), 3) for j in range(7)}
        for _ in range(3 if i == 0 else 1):
            observations.append({**p, **run_experiment(p)})
    print(f"history: {len(observations)} experiment rows\n")

    print("Current ranking (best first):")
    opt.status(observations)
    print()

    print("Is the surrogate useful on this data? (leave-one-out)")
    opt.diagnose(observations)
    sd = pooled_replicate_sd(observations, opt.space, opt.target_names)
    print("\nreplicate SD per target (use as 'margin' if you want a safety buffer):")
    print({k: (None if v is None else round(v, 3)) for k, v in sd.items()}, "\n")

    # --- optimization loop -------------------------------------------------------
    for rnd in range(1, 4):
        print(f"=== round {rnd} ===")
        for p in opt.recommend(observations):
            result = run_experiment(p)                 # <- your real experiment goes here
            observations.append({**p, **result})
            ok = opt.targets.limit_status([result]).all()
            print("  proposed", {k: round(v, 2) for k, v in p.items()},
                  "->", "MEETS all limits" if ok else "misses a limit")
    print("\nFinal ranking, top 5:")
    opt.status(observations)


if __name__ == "__main__":
    main()
