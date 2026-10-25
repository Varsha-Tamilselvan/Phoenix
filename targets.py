"""
targets.py
----------
Turns quality attributes with ACCEPTANCE LIMITS into the one-sided,
priority-ordered objectives that Chimera understands.

Each target is a dict. The ORDER of the list is the priority order
(first = most important). Keys:

    name         column name in your observation rows
    lower        optional lower acceptance limit  (value must be >= lower)
    upper        optional upper acceptance limit  (value must be <= upper)
    margin       optional safety margin in the attribute's own units
                 (default 0). It tightens every limit you gave:
                 lower+margin, upper-margin. For a two-sided window it
                 shrinks the window from both sides. A sensible choice is
                 the replicate standard deviation (see
                 data_processing.pooled_replicate_sd).
    window_mode  only for two-sided targets:
                   'centre' (default): objective = |y - midpoint|,
                            must be <= half-width - margin
                   'flat'            : objective = distance OUTSIDE the
                            (margin-tightened) window, must be 0
                 Both accept exactly the same values of y. They differ only
                 in how values are ordered when this target is the first
                 (top-priority) one or when points fail it.
    goal, tolerance, absolute
                 only for a target WITHOUT limits (plain "make it bigger /
                 smaller" objective, as in the earlier version):
                 goal='min'|'max', tolerance = fraction of observed range
                 (default 0.2), absolute=False.

How each kind is translated (y = measured value):

    lower limit L          goal max   column y            pass if y >= L+margin
    upper limit U          goal min   column y            pass if y <= U-margin
    window [L, U] centre   goal min   column |y-(L+U)/2|  pass if <= (U-L)/2-margin
    window [L, U] flat     goal min   column max(L'-y,0,y-U')   pass if == 0
    no limits              goal/tolerance as given
"""

import numpy as np

from chimera import Chimera


class TargetSet:
    def __init__(self, targets):
        if not targets:
            raise ValueError("define at least one target")
        self.specs = []
        seen = set()
        for t in targets:
            name = t.get("name")
            if not name or not isinstance(name, str):
                raise ValueError(f"every target needs a string 'name': {t}")
            if name in seen:
                raise ValueError(f"duplicate target name '{name}'")
            seen.add(name)
            self.specs.append(self._parse(t))
        self.names = [s["name"] for s in self.specs]

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(t):
        name = t["name"]
        lower, upper = t.get("lower"), t.get("upper")
        margin = float(t.get("margin", 0.0))
        if margin < 0:
            raise ValueError(f"{name}: margin must be >= 0")
        for lim, label in ((lower, "lower"), (upper, "upper")):
            if lim is not None and not np.isfinite(float(lim)):
                raise ValueError(f"{name}: {label} limit must be a finite number")
        spec = {"name": name, "margin": margin, "lower": lower, "upper": upper}

        if lower is not None and upper is not None:
            lower, upper = float(lower), float(upper)
            if not lower < upper:
                raise ValueError(f"{name}: lower limit must be below upper limit")
            eff_lo, eff_hi = lower + margin, upper - margin
            if not eff_lo < eff_hi:
                raise ValueError(
                    f"{name}: margin {margin} is too large for the window [{lower}, {upper}] "
                    f"(it would leave no acceptable range)")
            mode = t.get("window_mode", "centre")
            if mode not in ("centre", "flat"):
                raise ValueError(f"{name}: window_mode must be 'centre' or 'flat'")
            spec.update(kind="window", mode=mode, eff_lo=eff_lo, eff_hi=eff_hi,
                        mid=(lower + upper) / 2.0, goal="min", absolute=True,
                        tol=(eff_hi - eff_lo) / 2.0 if mode == "centre" else 0.0)
        elif lower is not None:
            spec.update(kind="lower", goal="max", absolute=True, tol=float(lower) + margin)
        elif upper is not None:
            spec.update(kind="upper", goal="min", absolute=True, tol=float(upper) - margin)
        else:
            goal = t.get("goal")
            if goal not in ("min", "max"):
                raise ValueError(f"{name}: give a lower/upper limit, or goal='min'|'max'")
            spec.update(kind="goal", goal=goal, absolute=bool(t.get("absolute", False)),
                        tol=float(t.get("tolerance", 0.2)))
        return spec

    # ------------------------------------------------------------------
    def _column(self, spec, y):
        y = np.asarray(y, dtype=float)
        if spec["kind"] == "window":
            if spec["mode"] == "centre":
                return np.abs(y - spec["mid"])
            return np.maximum.reduce([spec["eff_lo"] - y, np.zeros_like(y), y - spec["eff_hi"]])
        return y

    def matrix(self, rows):
        """(n_rows, n_targets) array of the columns Chimera should see."""
        cols = []
        for s in self.specs:
            y = np.array([float(r[s["name"]]) for r in rows])
            cols.append(self._column(s, y))
        return np.column_stack(cols)

    def chimera(self):
        return Chimera([s["goal"] for s in self.specs],
                       [s["tol"] for s in self.specs],
                       [s["absolute"] for s in self.specs])

    # ------------------------------------------------------------------
    @staticmethod
    def _slack(v):
        return 1e-9 * max(1.0, abs(v))

    def limit_status(self, rows, use_margin=False):
        """
        Boolean array (n_rows, n_targets): does each row satisfy each target's
        acceptance limits? With use_margin=True the margin-tightened limits are
        used. Targets without limits are always True.
        """
        out = np.ones((len(rows), len(self.specs)), dtype=bool)
        for j, s in enumerate(self.specs):
            y = np.array([float(r[s["name"]]) for r in rows])
            if s["kind"] == "goal":
                continue
            m = s["margin"] if use_margin else 0.0
            ok = np.ones(len(y), dtype=bool)
            if s["lower"] is not None:
                lo = float(s["lower"]) + m
                ok &= y >= lo - self._slack(lo)
            if s["upper"] is not None:
                hi = float(s["upper"]) - m
                ok &= y <= hi + self._slack(hi)
            out[:, j] = ok
        return out

    def explain(self):
        """Plain-language description of how every target is interpreted."""
        lines = []
        for i, s in enumerate(self.specs, 1):
            k = s["kind"]
            if k == "lower":
                d = f"must be >= {float(s['lower'])} (+ margin {s['margin']} -> >= {s['tol']})"
            elif k == "upper":
                d = f"must be <= {float(s['upper'])} (- margin {s['margin']} -> <= {s['tol']})"
            elif k == "window":
                d = (f"must lie in [{float(s['lower'])}, {float(s['upper'])}]"
                     f" (margin {s['margin']} -> [{s['eff_lo']}, {s['eff_hi']}]); "
                     f"mode={s['mode']}")
            else:
                d = (f"no limits: {s['goal']}imize, "
                     f"{'absolute' if s['absolute'] else 'relative'} tolerance {s['tol']}")
            lines.append(f"  priority {i}: {s['name']:<22s} {d}")
        return "\n".join(lines)
