import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from .ppo import PPOConfig, train


def _slug(value) -> str:
    if isinstance(value, dict):
        return "_".join(f"{k}-{_slug(v)}" for k, v in value.items()) or "default"
    return str(value)


def _run(kwargs: dict) -> str:
    torch.set_num_threads(1)
    train(PPOConfig(**kwargs))
    return kwargs["run_name"]


def sweep(spec_path: str, workers: int) -> None:
    spec = json.loads(Path(spec_path).read_text())
    out_dir = Path(spec["out_dir"])
    jobs, finished = [], 0
    for values in itertools.product(*spec["grid"].values()):
        combo = dict(zip(spec["grid"], values))
        name = "__".join(f"{k}-{_slug(v)}" for k, v in combo.items())
        if (out_dir / name / "final.json").exists():
            finished += 1
            continue
        jobs.append({**spec["base"], **combo, "out_dir": str(out_dir), "run_name": name, "log_interval": 0})

    print(f"{len(jobs)} runs to do, {finished} already finished")
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            print(f"[{n}/{len(jobs)}] {future.result()}", flush=True)


def summarize(out_dir: str) -> pd.DataFrame:
    rows, config_cols = [], set()
    for final in sorted(Path(out_dir).glob("*/final.json")):
        config = json.loads((final.parent / "config.json").read_text())
        config.update({f"env.{k}": v for k, v in config.pop("env_kwargs").items()})
        config_cols.update(config)
        rows.append({**json.loads(final.read_text()), **config})
    if not rows:
        raise SystemExit(f"No finished runs in {out_dir}")
    df = pd.DataFrame(rows)

    settings = [c for c in df.columns if c in config_cols and c not in ("method", "seed", "run_name")]
    df[settings] = df[settings].astype(object).where(df[settings].notna(), "default")
    group_cols = ["method"] + [c for c in settings if df[c].nunique() > 1]
    metric_cols = [c for c in df.columns if c.startswith("info_")] + ["mean_return"]

    grouped = df.groupby(group_cols, sort=False)
    summary = grouped[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric.removeprefix('info_')}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "seeds", grouped.size())
    summary = summary.reset_index().sort_values(
        group_cols, key=lambda col: col.map(lambda v: (isinstance(v, str), v)), ignore_index=True)
    summary.to_csv(Path(out_dir) / "summary.csv", index=False)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="experiment file, e.g. experiments/key_to_door.json")
    parser.add_argument("--workers", type=int, default=1, help="number of runs to train in parallel")
    args = parser.parse_args()
    sweep(args.spec, args.workers)
    out_dir = json.loads(Path(args.spec).read_text())["out_dir"]
    print(summarize(out_dir).round(3).to_string(index=False))
    print(f"\nSaved to {out_dir}/summary.csv")
