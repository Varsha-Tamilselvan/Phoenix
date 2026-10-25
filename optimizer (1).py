"""
optimizer.py
------------
Top-level `SimpleGryffin` class. One call to `recommend()` runs, in order:

    1. data_processing.prepare_observations  validate rows, merge replicates
    2. data_processing.ParameterSpace        inputs -> unit-scale encoded vectors
    3. targets.TargetSet + chimera.Chimera   quality attributes with acceptance
                                             limits -> ONE merit value per
                                             unique condition (0 = best)
    4. surrogate.KernelDensitySurrogate      kernel model of merit over inputs
    5. acquisition.optimize_acquisition      search for the best next
                                             experiment(s), one per lambda
    6. decode back to real units             list of dicts you can run

Usage
-----
    opt = SimpleGryffin(parameters, targets, general)
    observations = load_observations_csv("batches.csv")   # may contain replicates
    proposals = opt.recommend(observations, num_proposals=2)
    # run the experiments, append {inputs..., targets...} rows, call again.

`observations` is always the FULL history. The optimizer keeps no state
between calls other than the diagnostics of the last call.
"""

import numpy as np

from data_processing import ParameterSpace, prepare_observations
from targets import TargetSet
from surrogate import KernelDensitySurrogate
from acquisition import optimize_acquisition, select_diverse_batch
from diagnostics import leave_one_out, choose_alpha_coeff, DEFAULT_GRID


DEFAULT_GENERAL = {
    "sampling_strategies": 2,    # number of lambda values = proposals per call
    "num_random_samples": 2000,  # global random search pool size
    "num_restarts": 10,          # how many of those get refined
    "adam_lr": 0.05,
    "adam_steps": 60,
    "obj_transform": "sqrt",
    "min_rel_dist": 0.05,        # diversity floor between proposals in one batch
    # kernel width: tau = coeff * n^2 (smaller coeff = wider kernels).
    # 'auto' picks the coefficient with the lowest leave-one-out error on your
    # data at every call (needs >= 6 unique conditions, else the paper's 12).
    # Give a number (e.g. 12.0, the Phoenics/Gryffin default) to fix it.
    "kernel_alpha_coeff": "auto",
    "replicates": "average",     # 'average' | 'first' | 'keep'
    "replicate_decimals": 6,     # inputs equal to this many decimals = same condition
    "random_seed": None,
    "verbose": True,
}


class SimpleGryffin:
    def __init__(self, parameters, targets, general=None):
        """
        parameters : list of dict, e.g. {'name': 'x1', 'low': 0, 'high': 10}
                     (kind defaults to 'continuous'; 'discrete_range',
                     'discrete_set' and 'categorical' are also supported)
        targets    : list of dict IN PRIORITY ORDER, see targets.py, e.g.
                     {'name': 'monomer',       'lower': 95.0}
                     {'name': 'hmw',           'upper': 3.0}
                     {'name': 'high_mannose',  'lower': 2.0, 'upper': 8.0}
        general    : dict overriding DEFAULT_GENERAL
        """
        self.space = ParameterSpace(parameters)
        self.param_names = self.space.names
        self.targets = TargetSet(targets)
        self.target_names = self.targets.names
        self.chimera = self.targets.chimera()

        clash = set(self.param_names) & set(self.target_names)
        if clash:
            raise ValueError(f"names used both as input and target: {sorted(clash)}")

        self.general = dict(DEFAULT_GENERAL)
        if general:
            unknown = set(general) - set(DEFAULT_GENERAL)
            if unknown:
                raise ValueError(f"unknown general setting(s): {sorted(unknown)}")
            self.general.update(general)
        c = self.general["kernel_alpha_coeff"]
        if not (c == "auto" or (isinstance(c, (int, float)) and c > 0)):
            raise ValueError("kernel_alpha_coeff must be 'auto' or a positive number")

        self._rng = np.random.default_rng(self.general["random_seed"])
        self.last_rows_ = None
        self.last_merit_ = None
        self.last_report_ = None
        self.last_surrogate_ = None
        self.last_alpha_coeff_ = None

    # ------------------------------------------------------------------
    def _log(self, msg):
        if self.general["verbose"]:
            print(msg)

    def _bootstrap(self, num_proposals):
        """Random legal points, used when there is nothing usable to learn from."""
        U = self._rng.uniform(size=(num_proposals, self.space.encoded_dim))
        for blk in self.space.blocks():
            idx, cands = blk["indices"], blk["candidates"]
            U[:, idx] = cands[self._rng.integers(len(cands), size=num_proposals)]
        return [self.space.decode_vector(u) for u in U]

    def _prepare(self, observations):
        rows, report = prepare_observations(
            observations, self.space, self.target_names,
            replicates=self.general["replicates"],
            decimals=self.general["replicate_decimals"],
            verbose=self.general["verbose"])
        self.last_rows_, self.last_report_ = rows, report
        return rows, report

    def _merit(self, rows):
        cols = self.targets.matrix(rows)
        return self.chimera.scalarize_and_transform(cols, transform=self.general["obj_transform"])

    # ------------------------------------------------------------------
    def recommend(self, observations, num_proposals=None):
        """
        Propose the next experiment(s).

        observations : list of dict (inputs + measured targets). Replicates are
                       fine: rows with identical inputs are merged by averaging.
        Returns a list of dicts holding the input values only, in real units.
        """
        n_prop = num_proposals or self.general["sampling_strategies"]
        rows, report = self._prepare(observations)
        if not rows:
            self._log(f"[recommend] no usable observations -> {n_prop} random point(s)")
            return self._bootstrap(n_prop)

        U = self.space.encode_observations(rows)
        merit = self._merit(rows)
        self.last_merit_ = merit
        if np.ptp(merit) == 0:
            self._log("[recommend] WARNING: every condition has the same merit, so the "
                      "surrogate is flat and proposals are driven by lambda alone.")

        coeff = self.general["kernel_alpha_coeff"]
        if coeff == "auto":
            coeff, info = choose_alpha_coeff(U, merit)
            if info is not None:
                best = next(r for r in info["results"] if r["alpha_coeff"] == coeff)
                self._log(f"[recommend] kernel width chosen by leave-one-out: alpha_coeff={coeff:g} "
                          f"(kernel sd {best['kernel_sd']:.3f}, RMSE {best['rmse']:.3f} vs "
                          f"{info['baseline_rmse']:.3f} for predicting the mean)")
                if best["rmse"] >= info["baseline_rmse"]:
                    self._log("[recommend] WARNING: no kernel width predicts held-out conditions "
                              "better than the average -- the surrogate has little to go on yet; "
                              "proposals will be driven mostly by lambda (explore/exploit).")
        self.last_alpha_coeff_ = float(coeff)
        surrogate = KernelDensitySurrogate(alpha_coeff=float(coeff)).fit(U, merit)
        self.last_surrogate_ = surrogate

        blocks = self.space.blocks()
        lambdas = np.linspace(1.0, -1.0, n_prop) if n_prop > 1 else np.array([1.0])
        cand_x, cand_val = [], []
        for lam in lambdas:
            x, val = optimize_acquisition(
                surrogate, lam, self.space.encoded_dim,
                num_random_samples=self.general["num_random_samples"],
                num_restarts=self.general["num_restarts"],
                adam_lr=self.general["adam_lr"],
                adam_steps=self.general["adam_steps"],
                blocks=blocks, rng=self._rng)
            cand_x.append(x)
            cand_val.append(val)

        kept = select_diverse_batch(cand_x, cand_val, min_rel_dist=self.general["min_rel_dist"])
        proposals = [self.space.decode_vector(cand_x[i]) for i in kept]

        ok = self.targets.limit_status(rows).all(axis=1)
        best = int(np.argmin(merit))
        self._log(f"[recommend] {len(rows)} unique conditions, {int(ok.sum())} meet every limit; "
                  f"best so far: { {t: round(rows[best][t], 3) for t in self.target_names} }")
        if len(proposals) < n_prop:
            self._log(f"[recommend] note: {n_prop - len(proposals)} proposal(s) dropped as too "
                      f"close to another one in this batch (min_rel_dist).")
        return proposals

    # ------------------------------------------------------------------
    def status(self, observations, show=True):
        """
        Table of your unique conditions, best first: merit, whether each meets
        every acceptance limit, and which targets fail. Returns the rows.
        """
        rows, _ = self._prepare(observations)
        if not rows:
            return []
        merit = self._merit(rows)
        within = self.targets.limit_status(rows)
        out = []
        for i in np.argsort(merit):
            r = dict(rows[i])
            r["merit"] = float(merit[i])
            r["meets_all_limits"] = bool(within[i].all())
            r["failed_targets"] = [t for t, ok in zip(self.target_names, within[i]) if not ok]
            out.append(r)
        if show:
            print(f"{'rank':>4} {'merit':>6} {'n_rep':>5}  meets all limits?  failed targets")
            for k, r in enumerate(out, 1):
                print(f"{k:>4} {r['merit']:>6.3f} {r['_n_rep']:>5}  "
                      f"{'yes' if r['meets_all_limits'] else 'no ':<17}  "
                      f"{', '.join(r['failed_targets']) or '-'}")
        return out

    def diagnose(self, observations, alpha_coeffs=DEFAULT_GRID):
        """Leave-one-out check of the surrogate (see diagnostics.py)."""
        rows, _ = self._prepare(observations)
        U = self.space.encode_observations(rows)
        merit = self._merit(rows)
        return leave_one_out(U, merit, alpha_coeffs=alpha_coeffs, verbose=self.general["verbose"])

    def predict_merit(self, param_dicts):
        """Surrogate's predicted merit for given inputs, from the last recommend() call."""
        if self.last_surrogate_ is None:
            raise RuntimeError("call recommend() at least once before predict_merit()")
        return self.last_surrogate_.predict(self.space.encode_observations(param_dicts))
