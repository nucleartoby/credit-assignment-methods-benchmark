import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from .settings import SETTINGS, offline, online
from .sweep import _slug, load_runs, sweep

WINDOW = 10
METRICS = ("credit_corr", "credit_slope", "spurious_frac", "spurious_rms", "decision_corr", "decision_retention", "decision_noise",
           "decision_signal", "distractor_corr", "distractor_retention", "share_offset", "share_neutral", "share_decision_noise",
           "share_distractor_noise", "value_error")
CONTROL = {"avg_credit_corr": "credit_corr", "avg_credit_slope": "credit_slope", "avg_spurious_frac": "spurious_frac",
           "avg_spurious_rms": "spurious_rms", "learn_decision_corr": "decision_corr", "learn_decision_retention": "decision_retention",
           "learn_decision_noise": "decision_noise", "learn_decision_signal": "decision_signal", "learn_distractor_corr": "distractor_corr",
           "learn_distractor_retention": "distractor_retention", "learn_share_offset": "share_offset", "learn_share_neutral": "share_neutral",
           "learn_share_decision_noise": "share_decision_noise", "learn_share_distractor_noise": "share_distractor_noise"}
PERFORMANCE = ("avg_p_decisive", "p_decisive", "solved_at", "solved")
CURVES = ("credit_corr", "decision_corr", "spurious_rms", "p_decisive", "value_error")
KEYS = ["setting", "condition", "task", "group", "label"]
LOWER_IS_BETTER = ("spurious",)
NO_DECISIVE = "no decisive action"


def tests(headline_episodes: int) -> dict:
    offline_head = f"{headline_episodes} episodes"
    return {"offline_accuracy": ("offline", offline_head, "credit_corr"), "offline_signal": ("offline", offline_head, "decision_corr"),
            "stationary_accuracy": ("stationary", "final", "credit_corr"), "stationary_signal": ("stationary", "final", "decision_corr"),
            "changing_accuracy": ("drift", "moving", "credit_corr"), "changing_signal": ("drift", "moving", "decision_corr"),
            "learning": ("control", "learning", "avg_p_decisive"), "spurious": ("stationary", "final", "spurious_rms")}


def task_name(t: dict) -> str:
    parts = [f"depth {t['depth']}"]
    parts += [f"density {t['density']:g}"] if t["density"] else []
    parts += [f"{t['num_actions']} actions"] if t["num_actions"] != 2 else []
    parts += [f"noise {t['reward_noise']:g}"] if t["reward_noise"] else []
    parts += [] if t["observe_flag"] else ["flag hidden"]
    parts += [] if t["decisive"] else [NO_DECISIVE]
    return ", ".join(parts)


def task_group(t: dict) -> str:
    changed = [t["density"] > 0, t["num_actions"] != 2, t["reward_noise"] > 0, not t["observe_flag"]]
    if not t["decisive"]:
        return NO_DECISIVE
    if sum(changed) > 1:
        return "combined"
    return next((g for g, c in zip(("density", "breadth", "stochasticity", "depth, flag hidden"), changed) if c), "depth, flag visible")


def name_tasks(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    env = list(spec["grid"]["env_kwargs"][0])
    tasks = [dict(zip(env, row)) for row in df[[f"env.{k}" for k in env]].itertuples(index=False)]
    return df.assign(task=[task_name(t) for t in tasks], group=[task_group(t) for t in tasks])


def labels(spec: dict, reference: bool = True) -> list[str]:
    return [v.get("label") or v["method"] for v in spec["grid"]["variant"] if reference or v["method"] != "oracle"]


def setting_jobs(spec: dict, out: Path) -> list[dict]:
    return [{"setting": setting, "task": task, "seed": seed, "path": str(out / f"{setting}__{_slug(task)}__seed-{seed}.csv"),
             "base": spec["base"], "variants": spec["grid"]["variant"], "settings": spec["settings"]}
            for setting in SETTINGS for task in spec["grid"]["env_kwargs"] for seed in spec["grid"]["seed"]]


def run_job(job: dict) -> str:
    torch.set_num_threads(1)
    s = job["settings"]
    if job["setting"] == "offline":
        rows = offline(job["task"], job["variants"], job["base"], job["seed"], s["offline_episodes"], s["p_end"])
    else:
        drift = s["drift_episodes"] if job["setting"] == "drift" else float("inf")
        rows = online(job["task"], job["variants"], job["base"], job["seed"], drift, s["p_end"])
    df = pd.DataFrame(rows).assign(setting=job["setting"], seed=job["seed"], **{f"env.{k}": v for k, v in job["task"].items()})
    df.to_csv(job["path"], index=False)
    return Path(job["path"]).name


def run_settings(spec: dict, out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    wanted = set(labels(spec))
    jobs = [j for j in setting_jobs(spec, out)
            if not Path(j["path"]).exists() or not wanted <= set(pd.read_csv(j["path"], usecols=["label"])["label"])]
    print(f"{len(jobs)} setting runs to do")
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(run_job, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            print(f"[{n}/{len(jobs)}] {future.result()}", flush=True)


def load_settings(spec: dict, out: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(j["path"]) for j in setting_jobs(spec, out)], ignore_index=True)
    df = name_tasks(df[df["label"].isin(labels(spec))], spec)
    last = df.groupby("setting")["update"].transform("max")
    stationary, drift = df["setting"] == "stationary", df["setting"] == "drift"
    df["condition"] = None
    df.loc[df["setting"] == "offline", "condition"] = df["episodes"].astype(int).astype(str) + " episodes"
    df.loc[stationary & (df["episodes"] <= spec["settings"]["drift_episodes"]), "condition"] = "early"
    df.loc[stationary & (df["update"] > last - WINDOW), "condition"] = "final"
    df.loc[drift & (df["moved"] < 1), "condition"] = "moving"
    df.loc[drift & (df["moved"] >= 1), "condition"] = "held"
    return df


def load_control(spec: dict, out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = load_runs(out_dir)[0]
    runs = name_tasks(runs[(runs["env_id"] == "DelayedEffect-v0") & runs["label"].isin(labels(spec))], spec)
    runs = runs.drop(columns=[c for c in CONTROL.values() if c in runs]).rename(columns=CONTROL)
    runs = runs.reindex(columns=[*runs.columns, *(c for c in CONTROL.values() if c not in runs)])
    runs["solved"] = (runs["p_decisive"] >= 0.9).astype(float).where(runs["group"] != NO_DECISIVE)
    curves = []
    for run in runs[["run_name", "task", "group", "label", "seed"]].to_dict("records"):
        for line in (Path(out_dir) / run["run_name"] / "metrics.jsonl").read_text().splitlines():
            m = json.loads(line)
            curves.append({**run, "update": m["update"], "episodes": m["episodes"], **{c: m.get(c) for c in CURVES}})
    curves = pd.DataFrame(curves).drop(columns="run_name").assign(setting="control")
    return runs.assign(setting="control", condition="learning"), curves.astype({c: float for c in CURVES})


def build(spec: dict, out: Path) -> dict[str, pd.DataFrame]:
    settings = load_settings(spec, out / "settings")
    control, control_curves = load_control(spec, spec["out_dir"])
    per_seed = settings.dropna(subset=["condition"]).groupby(KEYS + ["seed"], sort=False)[list(METRICS)].mean().reset_index()
    both = pd.concat([per_seed, control[KEYS + ["seed", *METRICS[:-1], *PERFORMANCE]]], ignore_index=True)
    grouped = both.groupby(KEYS, sort=False)
    tasks = grouped[[*METRICS, *PERFORMANCE]].agg(["mean", "std"])
    tasks.columns = [f"{metric}_{stat}" for metric, stat in tasks.columns]
    tasks.insert(0, "seeds", grouped.size())
    tasks = tasks.reset_index()

    streams = settings[settings["setting"].isin(["stationary", "drift"])]
    curves = pd.concat([streams[["setting", "task", "group", "label", "update", "episodes", *CURVES]], control_curves], ignore_index=True)
    curves = curves.groupby(["setting", "task", "group", "label", "update"], sort=False)[["episodes", *CURVES]].mean().round(4).reset_index()
    curves[["update", "episodes"]] = curves[["update", "episodes"]].astype(int)

    columns, groups = {}, []
    for name, (setting, condition, metric) in tests(spec["settings"]["headline_episodes"]).items():
        part = tasks[(tasks["setting"] == setting) & (tasks["condition"] == condition)]
        groups.append(part.groupby(["group", "label"], sort=False)[f"{metric}_mean"].mean().rename(name))
        columns[name] = part[(part["group"] == NO_DECISIVE) == (name == "spurious")].groupby("label")[f"{metric}_mean"].mean()
    comparison = pd.DataFrame(columns).reindex(labels(spec))
    ranks = comparison.loc[labels(spec, reference=False)].apply(lambda c: c.rank(ascending=c.name in LOWER_IS_BETTER))
    comparison["mean_rank"] = ranks.mean(axis=1)
    comparison = comparison.sort_values("mean_rank", na_position="last").rename_axis("label").reset_index()
    groups = pd.concat(groups, axis=1).reset_index()
    groups.loc[groups["group"] == NO_DECISIVE, "learning"] = float("nan")
    return {"comparison": comparison, "groups": groups, "tasks": tasks, "curves": curves}


def show(tables: dict[str, pd.DataFrame], spec: dict) -> None:
    tasks = tables["tasks"]
    decisive = tasks[tasks["group"] != NO_DECISIVE]
    for metric in ("credit_corr", "decision_corr"):
        wide = decisive.pivot_table(index="label", columns=["setting", "condition"], values=f"{metric}_mean", aggfunc="mean", sort=False)
        print(f"\n{metric} in each setting, mean over the tasks with a decisive action")
        print(wide.reindex(labels(spec)).round(3).to_string())
    print("\nComparison: accuracy is credit_corr, signal is decision_corr, learning is avg_p_decisive in the PPO loop,"
          " all averaged over the tasks with a decisive action; spurious is spurious_rms on the tasks without one (lower is better)."
          " Ranks leave out the oracle.")
    print(tables["comparison"].round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="experiment file with a settings block, e.g. experiments/baselines.json")
    parser.add_argument("--workers", type=int, default=1, help="number of runs to train in parallel")
    parser.add_argument("--out-dir", help="read and write the runs here instead of the experiment file's out_dir")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    spec["out_dir"] = args.out_dir or spec["out_dir"]
    out = Path(spec["out_dir"])

    sweep(spec, args.workers)
    run_settings(spec, out / "settings", args.workers)
    tables = build(spec, out)
    (out / "benchmark").mkdir(exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / "benchmark" / f"{name}.csv", index=False)
    show(tables, spec)
