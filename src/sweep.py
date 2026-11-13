import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .ppo import PPOConfig, train

HEADLINE = ("p_decisive", "avg_p_decisive", "solved_at", "avg_credit_corr", "avg_credit_slope", "avg_spurious_frac",
            "avg_spurious_rms", "mean_return")


def _slug(value) -> str:
    if isinstance(value, dict):
        return "_".join(f"{k}-{_slug(v)}" for k, v in value.items()) or "default"
    return str(value)


def _run(kwargs: dict) -> str:
    torch.set_num_threads(1)
    train(PPOConfig(**kwargs))
    return kwargs["run_name"]


def sweep(spec: dict, workers: int, only: str = "") -> None:
    out_dir = Path(spec["out_dir"])
    config_fields = {f.name for f in fields(PPOConfig)}
    jobs, finished = [], 0
    for values in itertools.product(*spec["grid"].values()):
        combo = dict(zip(spec["grid"], values))
        name = "__".join(f"{k}-{_slug(v)}" for k, v in combo.items())
        if only not in name:
            continue
        if (out_dir / name / "final.json").exists():
            finished += 1
            continue
        kwargs = dict(spec["base"])
        for key, value in combo.items():
            kwargs.update({key: value} if key in config_fields else value)
        jobs.append({**kwargs, "out_dir": str(out_dir), "run_name": name, "log_interval": 0})

    print(f"{len(jobs)} runs to do, {finished} already finished")
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(_run, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            print(f"[{n}/{len(jobs)}] {future.result()}", flush=True)


def summarize(out_dir: str) -> tuple[pd.DataFrame, list[str]]:
    rows, config_cols = [], set()
    for final in sorted(Path(out_dir).glob("*/final.json")):
        config = json.loads((final.parent / "config.json").read_text())
        config.update({f"env.{k}": v for k, v in config.pop("env_kwargs").items()})
        config_cols.update(config)
        rows.append({**json.loads(final.read_text()), **config})
    if not rows:
        raise SystemExit(f"No finished runs in {out_dir}")
    df = pd.DataFrame(rows)

    variant = "label" if df["label"].astype(bool).all() else "method"
    settings = [c for c in df.columns if c in config_cols and c not in (variant, "seed", "run_name")]
    df[settings] = df[settings].astype(object).where(df[settings].notna(), "default")
    group_cols = [variant] + [c for c in settings if df.groupby(variant)[c].nunique().max() > 1]
    metric_cols = [c for c in df.columns if c.startswith("info_")] + [c for c in HEADLINE if c in df.columns]
    metric_cols += [c for c in df.columns if c.startswith("learn_")]

    grouped = df.groupby(group_cols, sort=False)
    summary = grouped[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{metric.removeprefix('info_')}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "seeds", grouped.size())
    summary = summary.reset_index().sort_values(
        group_cols, key=lambda col: col.map(lambda v: (isinstance(v, str), v)), ignore_index=True)
    summary.to_csv(Path(out_dir) / "summary.csv", index=False)
    return summary, group_cols


def scaling_slopes(summary: pd.DataFrame, group_cols: list[str], families: list[dict], out_dir: str) -> pd.DataFrame:
    variant_cols = [c for c in group_cols if not c.startswith("env.")]
    rows = []
    for family in families:
        base = family["base"]
        for axis, scale in family["axes"].items():
            others = [k for k in base if k != axis and f"env.{k}" in summary.columns]
            on_axis = summary[np.all([summary[f"env.{k}"] == base[k] for k in others], axis=0)]
            for variant, part in on_axis.groupby(variant_cols, sort=False):
                x = part[f"env.{axis}"].astype(float)
                x = np.log2(1 + x) if scale == "log" else x
                if x.nunique() < 2:
                    continue
                row = dict(zip(variant_cols, variant), family=family["name"], axis=axis, points=len(part))
                for metric in ("p_decisive_mean", "avg_p_decisive_mean"):
                    row[f"{metric.removesuffix('_mean')}_slope"] = np.polyfit(x, part[metric].astype(float), 1)[0]
                rows.append(row)
    slopes = pd.DataFrame(rows)
    slopes.to_csv(Path(out_dir) / "scaling.csv", index=False)
    return slopes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", help="experiment file, e.g. experiments/baselines.json")
    parser.add_argument("--workers", type=int, default=1, help="number of runs to train in parallel")
    parser.add_argument("--only", default="", help="run only the runs whose name contains this text, e.g. label-oracle")
    parser.add_argument("--out-dir", help="write the runs here instead of the experiment file's out_dir")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    spec["out_dir"] = args.out_dir or spec["out_dir"]

    sweep(spec, args.workers, args.only)
    summary, group_cols = summarize(spec["out_dir"])
    shown = group_cols + ["seeds"] + [c for c in summary.columns if c.endswith("_mean") and not c.startswith("learn_")]
    print(summary[shown].rename(columns=lambda c: c.removesuffix("_mean")).round(3).to_string(index=False))
    print(f"\nSaved to {spec['out_dir']}/summary.csv")
    if "scaling" in spec:
        slopes = scaling_slopes(summary, group_cols, spec["scaling"], spec["out_dir"])
        print("\nScaling slopes (change in the probability of the decisive action per unit of each axis)")
        print(slopes.round(3).to_string(index=False))
