import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .methods import METHODS
from .metrics import WASTE, failure_metrics
from .ppo import PPOConfig
from .sweep import summarize
from .tasks import DelayedEffect
from .verify import CRITICS, critic_values, sample

PROFILE = ("decision_corr", "decision_retention", "detect_episodes", "distractor_corr", "position", "mostly")
TRAINING = ("oracle_gap", "learn_decision_corr", "learn_decision_retention", "learn_distractor_corr", "learn_share_offset", "learn_share_neutral")


def verdict(row: dict, limit: float = float("inf"), z: float = 2.0) -> dict:
    shares = {"decision": row["decision_corr"] ** 2, "distractor": row["distractor_corr"] ** 2}
    shares.update({k.removeprefix("share_"): row[k] for k in WASTE})
    shares = {k: v for k, v in shares.items() if np.isfinite(v)}
    snr = row["decision_retention"] * row["decision_signal"] / max(row["decision_noise"], 1e-12)
    detect = (z / snr) ** 2 if snr > 0 else float("inf")
    return {"detect_episodes": float("nan") if np.isnan(snr) else detect if detect <= limit else float("inf"),
            "mostly": max(shares, key=shares.get) if shares else "none"}


def profile(task_kwargs: dict, variants: list[dict], critic: str = "exact", num_episodes: int = 20_000, seed: int = 0) -> list[dict]:
    task = DelayedEffect(num_episodes, seed=seed, **task_kwargs)
    probs = np.full((2 * (task.depth + 1), task.num_actions), 1.0 / task.num_actions)
    rollout, states = sample(task, probs, critic_values(task, probs, critic, seed), seed)
    truth = rollout.oracle_credit.numpy()
    groups, roles = task.observation_ids()[states], task.roles()[states]
    rows = []
    for variant in variants:
        cfg = PPOConfig(env_id="DelayedEffect-v0", gamma=1.0, **variant)
        credit = METHODS[cfg.method](rollout, cfg).numpy().astype(np.float64)
        row = failure_metrics(credit, truth, groups, roles)
        energy = np.bincount(states // 2, (credit - credit.mean()) ** 2)
        scale = energy.sum() * task.depth
        row["position"] = energy @ np.arange(task.depth + 1) / scale if scale > 1e-12 else float("nan")
        rows.append({"label": cfg.label or cfg.method, "critic": critic, **row, **verdict(row, num_episodes)})
    return rows


def diagnose(spec: dict, critics=("exact", "zero"), num_episodes: int = 20_000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for task_kwargs in spec["grid"]["env_kwargs"]:
        case = {f"env.{k}": v for k, v in task_kwargs.items()}
        for critic in critics:
            rows += [{**case, **row} for row in profile(task_kwargs, spec["grid"]["variant"], critic, num_episodes, seed)]
    return pd.DataFrame(rows)


def training(spec: dict, cases: list[str]) -> pd.DataFrame:
    summary = summarize(spec["out_dir"])[0]
    summary = summary.rename(columns=lambda c: c.removesuffix("_mean")).set_index(cases)
    reference = [v["label"] for v in spec["grid"]["variant"] if v["method"] == "oracle"]
    if reference and "learn_decision_corr" in summary:
        best = summary.loc[summary["label"] == reference[0], "avg_p_decisive"]
        gap = best.reindex(summary.index).to_numpy() - summary["avg_p_decisive"]
        summary["oracle_gap"] = gap.where(summary["learn_decision_corr"].notna())
    return summary.reset_index()


def show(table: pd.DataFrame, cases: list[str], metrics, title: str) -> None:
    for metric in metrics:
        wide = table.pivot(index=cases, columns="label", values=metric).dropna(how="all") if metric in table else pd.DataFrame()
        if not wide.empty:
            wide = wide.rename(index=lambda v: str(v)[0] if isinstance(v, (bool, np.bool_)) else v)
            wide.index.names = [name.removeprefix("env.") for name in wide.index.names]
            print(f"\n{title}: {metric}")
            print((wide if metric == "mostly" else wide.astype(float).round(0 if metric.endswith("episodes") else 3)).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="experiment file, e.g. experiments/baselines.json")
    parser.add_argument("--critics", nargs="+", default=["exact", "zero"], choices=CRITICS)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", help="read the runs from here instead of the experiment file's out_dir")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    spec["out_dir"] = args.out_dir or spec["out_dir"]
    tasks = spec["grid"]["env_kwargs"]
    cases = [f"env.{k}" for k in tasks[0] if len({json.dumps(task[k]) for task in tasks}) > 1]

    table = diagnose(spec, args.critics, args.episodes, args.seed)
    for critic in args.critics:
        show(table[table["critic"] == critic], cases, PROFILE, f"start-of-training policy, {critic} critic")
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "profile.csv", index=False)
    print(f"\nSaved to {out_dir}/profile.csv")
    if any(out_dir.glob("*/final.json")):
        runs = training(spec, cases)
        show(runs, cases, TRAINING, "training runs")
        if "learn_decision_corr" not in runs:
            print(f"\nThe runs in {out_dir} were made before these diagnostics were logged; rerun the sweep with --out-dir to get them")
