"""
data_processing.py
-------------------
Cleaning of raw experimental records and scaling/encoding of input
parameters into the internal representation the surrogate and
acquisition function operate on.

Each parameter has a 'kind' (default 'continuous'):

  - 'continuous'     : {'name', 'low', 'high'}
                       any real value in [low, high].
  - 'discrete_range' : {'name', 'low', 'high'}
                       any INTEGER in [low, high] (consecutive, step 1)
                       -- e.g. "number of plates: 5 to 20". Mirrors
                       Gryffin's own 'discrete' type, which internally
                       expands to options = arange(low, high + 1).
  - 'discrete_set'   : {'name', 'options'}
                       only the specific NUMERIC values listed are
                       legal, not necessarily consecutive -- e.g.
                       "concentration: one of 0.5, 1.0, 2.5". Distance
                       between two values is their numeric difference
                       (12 is closer to 10 than to 15).
  - 'categorical'    : {'name', 'options'}
                       only the specific LABELS listed are legal, and
                       they are UNORDERED -- e.g. "production media:
                       one of 'A', 'B', 'C'" (labels need not be
                       numbers; even if they are numbers, e.g. media
                       IDs 10/12/15, no numeric closeness is assumed
                       between them). Encoded as one-hot, so every
                       pair of distinct categories is equally distant
                       -- this is the same "naive"/descriptor-free
                       categorical kernel Gryffin itself falls back to
                       when no expert descriptors are supplied for a
                       categorical variable.

No external dependencies beyond numpy.
"""

import numpy as np


class ParameterSpace:
    """
    Holds every input parameter's definition and converts between
    real/labeled values (what your experiment understands) and the
    internal ENCODED representation (what the surrogate/acquisition
    operate on): continuous and numeric-discrete parameters contribute
    one encoded coordinate each, scaled to [0, 1]; categorical
    parameters contribute one encoded coordinate PER OPTION (a one-hot
    block), so the encoded space can be wider than the number of
    user-facing parameters.

    Example
    -------
        [{'name': 'temperature', 'kind': 'continuous',    'low': 20.0, 'high': 120.0},
         {'name': 'num_plates',  'kind': 'discrete_range', 'low': 5,    'high': 20},
         {'name': 'conc',        'kind': 'discrete_set',   'options': [0.5, 1.0, 2.5]},
         {'name': 'media',       'kind': 'categorical',    'options': ['A', 'B', 'C']}]
    """

    def __init__(self, param_specs):
        if len(param_specs) == 0:
            raise ValueError("param_specs must contain at least one parameter")

        self.names = [p["name"] for p in param_specs]
        self.kinds = [p.get("kind", "continuous") for p in param_specs]
        self.n_params = len(self.names)

        self.low = [None] * self.n_params          # continuous / discrete_range / discrete_set only
        self.high = [None] * self.n_params
        self.allowed_real = [None] * self.n_params  # sorted ndarray of legal values (discrete_range/discrete_set)
        self.categories = [None] * self.n_params     # list of labels (categorical only)

        offsets = []
        widths = []
        offset = 0
        for i, (p, kind) in enumerate(zip(param_specs, self.kinds)):
            if kind == "continuous":
                lo, hi = float(p["low"]), float(p["high"])
                if hi <= lo:
                    raise ValueError(f"{p['name']}: continuous needs high > low")
                self.low[i], self.high[i] = lo, hi
                width = 1

            elif kind == "discrete_range":
                lo, hi = int(p["low"]), int(p["high"])
                if hi <= lo:
                    raise ValueError(f"{p['name']}: discrete_range needs high > low")
                self.low[i], self.high[i] = float(lo), float(hi)
                self.allowed_real[i] = np.arange(lo, hi + 1, dtype=float)
                width = 1

            elif kind == "discrete_set":
                opts = np.array(sorted(float(v) for v in p["options"]))
                if len(opts) < 2:
                    raise ValueError(f"{p['name']}: discrete_set needs at least 2 options")
                self.low[i], self.high[i] = float(opts.min()), float(opts.max())
                self.allowed_real[i] = opts
                width = 1

            elif kind == "categorical":
                labels = list(p["options"])
                if len(labels) < 2:
                    raise ValueError(f"{p['name']}: categorical needs at least 2 options")
                if len(set(labels)) != len(labels):
                    raise ValueError(f"{p['name']}: categorical options must be unique")
                self.categories[i] = labels
                width = len(labels)

            else:
                raise ValueError(f"unknown parameter kind '{kind}' for '{p['name']}'")

            offsets.append(offset)
            widths.append(width)
            offset += width

        self.offsets = offsets
        self.widths = widths
        self.encoded_dim = offset  # total width of the internal representation
        self.discrete_dims = [i for i, k in enumerate(self.kinds) if k != "continuous"]
        self.categorical_dims = [i for i, k in enumerate(self.kinds) if k == "categorical"]

    # ------------------------------------------------------------------
    # encode: real/labeled dict  ->  internal vector (length encoded_dim)
    # ------------------------------------------------------------------
    def encode_dict(self, d):
        u = np.zeros(self.encoded_dim)
        for i, name in enumerate(self.names):
            if name not in d:
                raise KeyError(f"observation missing required parameter '{name}'")
            start = self.offsets[i]
            kind = self.kinds[i]

            if kind == "categorical":
                label = d[name]
                try:
                    cat_idx = self.categories[i].index(label)
                except ValueError:
                    raise ValueError(
                        f"'{label}' is not a legal option for categorical parameter "
                        f"'{name}' (allowed: {self.categories[i]})"
                    )
                u[start + cat_idx] = 1.0
            else:
                val = float(d[name])
                u[start] = (val - self.low[i]) / (self.high[i] - self.low[i])
        return u

    def encode_observations(self, observations):
        """observations: list of dicts -> ndarray (n, encoded_dim)"""
        return np.array([self.encode_dict(o) for o in observations])

    # ------------------------------------------------------------------
    # decode: internal vector  ->  real/labeled dict
    # ------------------------------------------------------------------
    def decode_vector(self, u):
        u = np.asarray(u, dtype=float)
        out = {}
        for i, name in enumerate(self.names):
            start = self.offsets[i]
            kind = self.kinds[i]

            if kind == "categorical":
                block = u[start:start + self.widths[i]]
                cat_idx = int(np.argmax(block))  # snap: nearest one-hot corner
                out[name] = self.categories[i][cat_idx]
            elif kind == "continuous":
                val = self.low[i] + np.clip(u[start], 0.0, 1.0) * (self.high[i] - self.low[i])
                out[name] = float(val)
            else:  # discrete_range / discrete_set -- snap to nearest legal value
                val = self.low[i] + np.clip(u[start], 0.0, 1.0) * (self.high[i] - self.low[i])
                opts = self.allowed_real[i]
                snapped = opts[np.argmin(np.abs(opts - val))]
                out[name] = int(snapped) if float(snapped).is_integer() else float(snapped)
        return out

    # ------------------------------------------------------------------
    # blocks: the finite-choice coordinate groups the acquisition
    # optimizer must treat as one atomic swap (never a gradient step)
    # ------------------------------------------------------------------
    def blocks(self):
        """
        Returns a list of (indices, candidates) for every NON-continuous
        parameter:
          - indices    : list of encoded-vector positions this parameter
                         occupies (length 1 for discrete_range/discrete_set,
                         length n_categories for categorical)
          - candidates : ndarray (n_options, len(indices)) -- each row is
                         one legal full assignment for those positions
                         (a single scaled value, or a one-hot row)

        Used by the acquisition optimizer exactly like Gryffin's
        NaiveDiscreteOptimizer/NaiveCategoricalOptimizer: try one
        random candidate row per block per iteration, keep only if it
        improves the acquisition value; never move these coordinates
        with a gradient step.
        """
        out = []
        for i in self.discrete_dims:
            start = self.offsets[i]
            width = self.widths[i]
            indices = list(range(start, start + width))
            if self.kinds[i] == "categorical":
                candidates = np.eye(width)  # one-hot rows
            else:
                opts = self.allowed_real[i]
                unit_vals = (opts - self.low[i]) / (self.high[i] - self.low[i])
                candidates = unit_vals.reshape(-1, 1)
            out.append({"indices": indices, "candidates": candidates})
        return out


def clean_observations(observations, space, objective_names,
                        drop_duplicates=True, verbose=True):
    """
    Validate and clean a list of observation dicts before they are fed
    to the optimizer.

    Parameters
    ----------
    observations : list of dict
    space : ParameterSpace
        Used to know each parameter's kind: numeric params are cast to
        float and checked for finiteness; categorical params are
        checked against their declared legal labels (no numeric cast).
    objective_names : list of str
    drop_duplicates : bool
        Remove rows whose full parameter assignment exactly repeats an
        earlier row.
    verbose : bool

    Returns
    -------
    cleaned : list of dict
    """
    all_keys = space.names + objective_names
    cleaned = []
    seen = []
    n_dropped_missing = 0
    n_dropped_bad_value = 0
    n_dropped_dup = 0

    for row in observations:
        if not all(k in row for k in all_keys):
            n_dropped_missing += 1
            continue

        ok = True
        for i, name in enumerate(space.names):
            if space.kinds[i] == "categorical":
                if row[name] not in space.categories[i]:
                    ok = False
                    break
            else:
                try:
                    v = float(row[name])
                except (TypeError, ValueError):
                    ok = False
                    break
                if not np.isfinite(v):
                    ok = False
                    break
        if not ok:
            n_dropped_bad_value += 1
            continue

        for name in objective_names:
            try:
                v = float(row[name])
            except (TypeError, ValueError):
                ok = False
                break
            if not np.isfinite(v):
                ok = False
                break
        if not ok:
            n_dropped_bad_value += 1
            continue

        if drop_duplicates:
            key = tuple(row[name] for name in space.names)
            if key in seen:
                n_dropped_dup += 1
                continue
            seen.append(key)

        cleaned.append(row)

    if verbose:
        print(
            f"[clean_observations] kept {len(cleaned)}/{len(observations)} rows "
            f"(missing_keys={n_dropped_missing}, invalid_value={n_dropped_bad_value}, "
            f"duplicates={n_dropped_dup})"
        )

    return cleaned


def objectives_to_array(observations, objective_names):
    """list of dicts -> ndarray (n, len(objective_names))"""
    return np.array([[float(o[name]) for name in objective_names] for o in observations])


# ======================================================================
# Replicate-aware preparation (this is what the optimizer uses)
# ======================================================================
def _condition_key(row, space, decimals):
    """Hashable key identifying an experimental condition (the inputs only)."""
    key = []
    for i, name in enumerate(space.names):
        if space.kinds[i] == "categorical":
            key.append(row[name])
        else:
            key.append(round(float(row[name]), decimals))
    return tuple(key)


def prepare_observations(observations, space, target_names, replicates="average",
                         decimals=6, verbose=True):
    """
    Validate raw experiment rows and collapse replicates.

    A replicate = rows whose INPUTS are identical (after rounding the numeric
    inputs to `decimals` places). Replicates carry no new spatial information,
    only information about noise, so they are merged into ONE row per unique
    condition before the surrogate sees them (otherwise identical kernels
    would stack up and the kernel width, which depends on the number of
    points, would shrink for no reason).

    Parameters
    ----------
    observations : list of dict   one dict per experiment (inputs + targets)
    space        : ParameterSpace
    target_names : list of str    measured columns to carry through
    replicates   : 'average' (default) -> mean of each target over replicates
                   'first'             -> keep only the first replicate
                   'keep'              -> do not merge (not recommended)
    decimals     : rounding used to decide that two inputs are "the same"

    Returns
    -------
    rows   : list of dict, one per unique condition: inputs, mean targets and
             '_n_rep' (how many experiments were merged)
    report : dict with counts and 'replicate_sd' (pooled standard deviation of
             each target over replicate groups; None where no replicates exist)
    """
    if replicates not in ("average", "first", "keep"):
        raise ValueError("replicates must be 'average', 'first' or 'keep'")
    need = list(space.names) + list(target_names)

    valid, n_missing, n_invalid, n_oob = [], 0, 0, 0
    for row in observations:
        if not all(k in row for k in need):
            n_missing += 1
            continue
        ok = True
        for i, name in enumerate(space.names):
            v = row[name]
            if space.kinds[i] == "categorical":
                if v not in space.categories[i]:
                    ok = False
                    break
            else:
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    ok = False
                    break
                if not np.isfinite(fv):
                    ok = False
                    break
        if ok:
            for name in target_names:
                try:
                    fv = float(row[name])
                except (TypeError, ValueError):
                    ok = False
                    break
                if not np.isfinite(fv):
                    ok = False
                    break
        if not ok:
            n_invalid += 1
            continue
        # inputs outside the declared bounds are kept (they are real data) but reported
        for i, name in enumerate(space.names):
            if space.kinds[i] in ("continuous", "discrete_range", "discrete_set"):
                v = float(row[name])
                pad = 1e-9 * max(1.0, abs(space.high[i] - space.low[i]))
                if v < space.low[i] - pad or v > space.high[i] + pad:
                    n_oob += 1
                    break
        valid.append(row)

    groups = {}
    order = []
    for idx, row in enumerate(valid):
        key = _condition_key(row, space, decimals) if replicates != "keep" else ("row", idx)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    rows = []
    ss = {t: 0.0 for t in target_names}   # pooled sum of squares around group means
    dof = 0
    for key in order:
        grp = groups[key]
        out = {name: grp[0][name] for name in space.names}
        if replicates == "first":
            use = grp[:1]
        else:
            use = grp
        for t in target_names:
            vals = np.array([float(g[t]) for g in use])
            out[t] = float(vals.mean())
            if len(vals) > 1:
                ss[t] += float(((vals - vals.mean()) ** 2).sum())
        if len(use) > 1:
            dof += len(use) - 1
        out["_n_rep"] = len(use) if replicates != "first" else len(grp)
        rows.append(out)

    sd = {t: (float(np.sqrt(ss[t] / dof)) if dof > 0 else None) for t in target_names}
    report = {
        "n_input": len(observations), "n_valid": len(valid), "n_unique": len(rows),
        "n_replicated_conditions": sum(1 for r in rows if r["_n_rep"] > 1),
        "dropped_missing": n_missing, "dropped_invalid": n_invalid,
        "rows_outside_bounds": n_oob, "replicate_sd": sd,
    }
    if verbose:
        msg = (f"[prepare_observations] {len(observations)} rows -> {len(valid)} valid -> "
               f"{len(rows)} unique conditions "
               f"({report['n_replicated_conditions']} with replicates)")
        if n_missing or n_invalid:
            msg += f"; dropped {n_missing} with missing keys, {n_invalid} with invalid values"
        print(msg)
        if n_oob:
            print(f"[prepare_observations] WARNING: {n_oob} row(s) have inputs outside the "
                  f"declared bounds; widen the bounds so the whole data range is inside them.")
    return rows, report


def pooled_replicate_sd(observations, space, target_names, decimals=6):
    """Pooled standard deviation of each target across replicate groups
    (None for targets with no replicates). Useful as a safety margin."""
    _, report = prepare_observations(observations, space, target_names,
                                     decimals=decimals, verbose=False)
    return report["replicate_sd"]


# ======================================================================
# Plain CSV helpers (standard library only)
# ======================================================================
def load_observations_csv(path):
    """Read a CSV with a header row into a list of dicts. Cells that parse as
    numbers become floats; empty cells become NaN (and are then reported as
    invalid by prepare_observations); everything else stays a string."""
    import csv
    rows = []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            row = {}
            for k, v in rec.items():
                if k is None:
                    continue
                k = k.strip()
                v = "" if v is None else v.strip()
                if v == "":
                    row[k] = float("nan")
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def save_observations_csv(rows, path):
    """Write a list of dicts to CSV (columns = union of keys, first-seen order)."""
    import csv
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
