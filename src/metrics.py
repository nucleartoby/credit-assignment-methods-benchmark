import numpy as np

CURVES = ("p_decisive", "expected_return", "credit_corr", "credit_slope", "spurious_frac", "spurious_rms")
ROLES = ("decision", "distractor", "neutral")
WASTE = ("share_offset", "share_neutral", "share_decision_noise", "share_distractor_noise")
FAILURES = tuple(f"{role}_{stat}" for role in ROLES[:2] for stat in ("corr", "retention", "noise", "signal")) + WASTE


def credit_metrics(credit, truth, groups, tol: float = 1e-8) -> dict:
    credit = np.asarray(credit, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    zero = np.abs(truth) < tol
    counts = np.bincount(groups)
    credit = credit - (np.bincount(groups, weights=credit) / np.maximum(counts, 1))[groups]
    truth = truth - (np.bincount(groups, weights=truth) / np.maximum(counts, 1))[groups]
    out = dict.fromkeys(("credit_corr", "credit_slope", "spurious_frac", "spurious_rms"), float("nan"))

    credit_ss, truth_ss = (credit**2).sum(), (truth**2).sum()
    if truth_ss > 1e-12:
        out["credit_slope"] = (credit * truth).sum() / truth_ss
        if credit_ss > 1e-12:
            out["credit_corr"] = (credit * truth).sum() / np.sqrt(credit_ss * truth_ss)
    if zero.any():
        out["spurious_rms"] = np.sqrt((credit[zero] ** 2).mean())
        if not zero.all() and credit_ss > 1e-12:
            out["spurious_frac"] = (credit[zero] ** 2).sum() / credit_ss
    return {k: float(v) for k, v in out.items()}


def failure_metrics(credit, truth, groups, roles) -> dict:
    credit = np.asarray(credit, dtype=np.float64)
    credit = credit - credit.mean()
    truth = np.asarray(truth, dtype=np.float64)
    counts = np.maximum(np.bincount(groups), 1)
    offset = (np.bincount(groups, weights=credit) / counts)[groups]
    within = credit - offset
    truth = truth - (np.bincount(groups, weights=truth) / counts)[groups]
    out = dict.fromkeys(FAILURES, float("nan"))

    total = (credit**2).sum()
    if total > 1e-12:
        out["share_offset"] = (offset**2).sum() / total
        out["share_neutral"] = (within[roles == ROLES.index("neutral")] ** 2).sum() / total
    for role, name in enumerate(ROLES[:2]):
        mask = roles == role
        c, t = within[mask], truth[mask]
        truth_ss = (t**2).sum()
        slope = (c * t).sum() / truth_ss if truth_ss > 1e-12 else float("nan")
        residual = c - np.nan_to_num(slope) * t
        out[f"{name}_retention"] = slope
        if mask.any():
            out[f"{name}_signal"] = np.sqrt(truth_ss / mask.sum())
            out[f"{name}_noise"] = np.sqrt(((residual + offset[mask]) ** 2).mean())
        if total > 1e-12:
            out[f"{name}_corr"] = slope * np.sqrt(truth_ss / total)
            out[f"share_{name}_noise"] = (residual**2).sum() / total
    return {k: float(v) for k, v in out.items()}


def run_summary(history: list[dict], solved: float = 0.9) -> dict:
    summary = dict(history[-1])
    for name in CURVES:
        values = [m[name] for m in history if np.isfinite(m.get(name, float("nan")))]
        if values:
            summary[f"avg_{name}"] = float(np.mean(values))
    if "p_decisive" in summary:
        reached = [m["episodes"] for m in history if m["p_decisive"] >= solved]
        summary["solved_at"] = float(reached[0]) if reached else float("nan")
        learning = [m for m in history if not reached or m["episodes"] < reached[0]]
        for name in FAILURES:
            values = [m[name] for m in learning if np.isfinite(m.get(name, float("nan")))]
            if values:
                summary[f"learn_{name}"] = float(np.mean(values))
    return summary
