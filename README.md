# SimpleGryffin — final version with acceptance-limit targets

Pure NumPy implementation of the Phoenics / Gryffin / Chimera pipeline. Built for
continuous process inputs and quality attributes that have acceptance limits.

```
data_processing.py  inputs -> unit scale; validation; REPLICATE averaging; CSV helpers
targets.py          quality attributes + acceptance limits -> Chimera objectives
chimera.py          priority-ordered objectives -> ONE merit (0 = best), strict hierarchy
surrogate.py        Gaussian kernel-density model of the merit over the inputs
acquisition.py      search for the next experiment(s) (random pool -> Adam refinement)
diagnostics.py      leave-one-out check of the surrogate; automatic kernel width
optimizer.py        SimpleGryffin: recommend(), status(), diagnose()
example_run.py      7 inputs, 9 targets, replicates, 3 rounds (synthetic data)
tests.py            python3 tests.py   (19 tests)
```

## Quick start

```python
from optimizer import SimpleGryffin
from data_processing import load_observations_csv

PARAMETERS = [{"name": "x1", "low": 0.0, "high": 10.0}, ...]      # your DoE ranges

TARGETS = [                                    # ORDER = priority, first = most important
    {"name": "monomer_pct",  "lower": 95.0},                       # must be >= 95
    {"name": "hmw_pct",      "upper": 3.0},                        # must be <= 3
    {"name": "high_mannose", "lower": 2.0, "upper": 8.0},          # must lie in [2, 8]
    ...
]
opt = SimpleGryffin(PARAMETERS, TARGETS, {"random_seed": 1})
print(opt.targets.explain())                    # check how every target is read

obs = load_observations_csv("batches.csv")      # inputs + target columns, replicates allowed
opt.status(obs)                                 # ranking, which batches meet every limit
opt.diagnose(obs)                               # does the surrogate predict held-out batches?
proposals = opt.recommend(obs, num_proposals=2) # next experiments, in real units
# run them, append {inputs..., measured targets...} to obs, call recommend() again
```

## How limits are handled (targets.py)

| You give | Chimera sees | Passes when |
|---|---|---|
| `lower: L` | maximize y, absolute threshold L | y >= L |
| `upper: U` | minimize y, absolute threshold U | y <= U |
| `lower: L, upper: U` (window) | minimize abs(y - mid), threshold (U-L)/2 | L <= y <= U |
| `margin: m` | tightens every limit by m (window shrinks from both sides) | y >= L+m, y <= U-m |
| none of the above | `goal` + relative `tolerance`, as in the earlier version | |

`window_mode: 'flat'` uses "distance outside the window, must be 0" instead of
"distance from the centre"; both accept exactly the same values. Choose a margin
of about one replicate standard deviation if you want batches that survive normal
run-to-run variation (`pooled_replicate_sd` computes it).

## Replicates

Rows with identical inputs (rounded to `replicate_decimals`, default 6) are merged
into ONE condition by averaging each target, so `n` = number of unique conditions.
Merging matters because the kernel width depends on `n`.
`general["replicates"]` can also be `'first'` or `'keep'` (not recommended).

## Chimera scoring

Each condition is classified by the FIRST target it fails; failing a higher-priority
target always ranks worse than failing only lower-priority ones. Conditions that
pass everything are ordered by the top-priority target. Merit is in [0, 1].

## Kernel width and what to expect with ~27 conditions

Kernel standard deviation = 1 / sqrt(coeff * n^2) in unit-cube coordinates. The
paper default (coeff = 12) gives about 0.01 at n = 27: the surrogate only knows
about points essentially on top of an experiment. `kernel_alpha_coeff: "auto"`
(default) chooses the coefficient by leave-one-out error on your data every call;
set a number to fix it. Read `opt.diagnose(obs)`: verdict "useful" means the surrogate
predicts held-out conditions clearly better than the average; if every row says
"marginal" or "worse", the proposals are driven mostly by lambda (exploit/explore),
and more data will help more than any setting.

## Known limits

* No noise model: replicates are averaged, but the surrogate treats the average as exact.
* Merit is re-scaled by min-max over the current data, so it shifts when a new extreme arrives.
* The batch diversity filter can return fewer proposals than requested (it says so).
* Only continuous inputs were exercised on real-style data; discrete/categorical inputs
  are supported and covered by a smoke test.

Changes from the previous version: Chimera replaced (the old tier arithmetic could rank a
point that failed the top priority above one that passed it); duplicate rows are averaged
instead of silently dropped; stable surrogate prediction far from data; leave-one-out
diagnostics and automatic kernel width; limit-based targets.
