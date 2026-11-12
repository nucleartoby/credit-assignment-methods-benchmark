import numpy as np

CURVES = ("p_decisive", "expected_return", "credit_corr", "credit_slope", "spurious_frac", "spurious_rms")


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


def run_summary(history: list[dict], solved: float = 0.9) -> dict:
    summary = dict(history[-1])
    for name in CURVES:
        values = [m[name] for m in history if np.isfinite(m.get(name, float("nan")))]
        if values:
            summary[f"avg_{name}"] = float(np.mean(values))
    if "p_decisive" in summary:
        reached = [m["episodes"] for m in history if m["p_decisive"] >= solved]
        summary["solved_at"] = float(reached[0]) if reached else float("nan")
    return summary
