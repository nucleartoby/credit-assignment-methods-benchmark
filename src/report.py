import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .benchmark import NO_DECISIVE, labels
from .sweep import load_runs

CONDITIONS = [("offline", "128 episodes", "128 ep"), ("offline", "1280 episodes", "1,280 ep"), ("offline", "10000 episodes", "10,000 ep"),
              ("stationary", "early", "first\n2,560 ep"), ("stationary", "final", "last\n10 updates"), ("drift", "moving", "policy\nmoving"),
              ("drift", "held", "policy\nheld"), ("control", "learning", "PPO\nrun")]
SETTINGS = [("Offline", 3), ("Stationary", 2), ("Drift", 2), ("Learn", 1)]
PANELS = [("decision_corr", "Signal (decision_corr)", "viridis", False), ("credit_corr", "Accuracy (credit_corr)", "viridis", False),
          ("spurious_rms", "Spurious credit (spurious_rms, lower is better)", "magma_r", True)]
LEARNING = [("avg_p_decisive", "avg\np_decisive", (0, 1)), ("p_decisive", "final\np_decisive", (0, 1)), ("solved", "seeds\nsolved", (0, 1)),
            ("solved_at", "solved at\n(episodes)", (10000, 0))]
NAMES = {"gae_0": "GAE, λ = 0", "gae_0.95": "GAE, λ = 0.95", "gae_1": "GAE, λ = 1", "mc": "Monte Carlo return", "oracle": "Oracle (true credit)"}
TESTS = ["offline_accuracy", "offline_signal", "stationary_accuracy", "stationary_signal", "changing_accuracy", "changing_signal", "learning", "spurious"]

# dashboard stuff

def cmap(name: str):
    c = plt.get_cmap(name).copy()
    c.set_bad("#ececec")
    return c


def heatmap(ax, values: np.ndarray, colors: np.ndarray, name: str, fmt) -> None:
    c = cmap(name)
    ax.imshow(np.ma.masked_invalid(colors), cmap=c, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(values):
        shade = colors[i, j]
        text = fmt(v) if np.isfinite(v) else "–"
        r, g, b, _ = c(shade) if np.isfinite(shade) else (0.93, 0.93, 0.93, 1)
        ax.text(j, i, text, ha="center", va="center", fontsize=7, color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")
    ax.set_yticks([])
    ax.tick_params(length=0)


def settings_header(ax) -> None:
    start = 0
    for name, width in SETTINGS:
        ax.text(start + width / 2 - 0.5, -1.9, name, ha="center", va="bottom", fontsize=8, fontweight="bold")
        if start:
            ax.axvline(start - 0.5, color="white", lw=2.5)
        start += width
    ax.set_xticks(range(len(CONDITIONS)), [c[2] for c in CONDITIONS], fontsize=7)
    ax.xaxis.tick_top()


def core_section(fig, grid, tasks: pd.DataFrame, label: str) -> None:
    mine = tasks[tasks["label"] == label]
    names = list(dict.fromkeys(tasks["task"]))
    groups = tasks.drop_duplicates("task").set_index("task")["group"]
    cell = {(r.setting, r.condition, r.task): r for r in mine.itertuples()}
    sub = grid.subgridspec(1, 5, width_ratios=[1.35, 3, 3, 3, 1.9], wspace=0.06)
    ax = fig.add_subplot(sub[0])
    ax.axis("off")
    ax.set_ylim(len(names) - 0.5, -0.5)
    breaks = [i - 0.5 for i in range(1, len(names)) if groups[names[i]] != groups[names[i - 1]]]
    for i, t in enumerate(names):
        ax.text(1, i, t, ha="right", va="center", fontsize=7.5, transform=ax.get_yaxis_transform())
    for k, (metric, title, name, lower) in enumerate(PANELS):
        values = np.array([[getattr(cell.get((s, c, t)), f"{metric}_mean", np.nan) for s, c, _ in CONDITIONS] for t in names], float)
        if lower:
            top = np.nanmax(tasks[f"{metric}_mean"]) or 1.0
            colors = values / top
        else:
            colors = np.clip(values, 0, 1)
        a = fig.add_subplot(sub[k + 1])
        heatmap(a, values, colors, name, lambda v: f"{v:.2f}")
        settings_header(a)
        for b in breaks:
            a.axhline(b, color="white", lw=1.5)
        a.set_title(title, fontsize=9, pad=50)
    values = np.array([[np.nan if groups[t] == NO_DECISIVE else getattr(cell.get(("control", "learning", t)), f"{m}_mean", np.nan)
                        for m, _, _ in LEARNING] for t in names], float)
    colors = np.column_stack([(values[:, j] - lo) / (hi - lo) for j, (_, _, (lo, hi)) in enumerate(LEARNING)])
    a = fig.add_subplot(sub[4])
    heatmap(a, values, colors, "viridis", lambda v: f"{v:,.0f}" if v > 1.5 else f"{v:.2f}")
    a.set_xticks(range(len(LEARNING)), [l for _, l, _ in LEARNING], fontsize=7)
    a.xaxis.tick_top()
    for b in breaks:
        a.axhline(b, color="white", lw=1.5)
    a.set_title("Performance in the PPO run\n(online, changing: learning)", fontsize=9, pad=36)


def game_runs(spec: dict, label: str) -> pd.DataFrame:
    out = Path(spec["out_dir"])
    runs = load_runs(str(out))[0] if any(out.glob("*/final.json")) else pd.DataFrame(columns=["label"])
    return runs[runs["label"] == label]


def game_table(ax, spec: dict, label: str, name: str, slots: int) -> None:
    ax.axis("off")
    variants = [v.get("label") or v["method"] for v in spec["grid"]["variant"]]
    seeds = len(spec["grid"]["seed"])
    head = f"{name} · online, changing: learning (PPO, {spec['base'].get('total_timesteps', 0):,} steps) · no oracle, so performance only"
    ax.set_title(head, fontsize=9, loc="left", fontweight="bold")
    if label not in variants:
        ax.text(0, 0.6, f"{label} is not in this experiment's grid.", fontsize=8, transform=ax.transAxes)
        return
    runs = game_runs(spec, label)
    if runs.empty:
        total = len(spec["grid"]["env_kwargs"]) * seeds
        ax.text(0, 0.6, f"Not run yet: 0 of {total} runs for {label}. Run python -m src.sweep {spec['file']}.", fontsize=8, transform=ax.transAxes)
        return
    keys = list(spec["grid"]["env_kwargs"][0])
    scores = list(dict.fromkeys(l["score"] for l in spec.get("ladders", [])))
    metrics = scores + sorted(c.removeprefix("info_") for c in runs if c.startswith("info_") and c.removeprefix("info_") not in scores)
    metrics += ["mean_return"]
    runs = runs.rename(columns=lambda c: c.removeprefix("info_"))
    rows, colors = [], []
    for task in spec["grid"]["env_kwargs"]:
        part = runs[np.all([runs[f"env.{k}"] == v for k, v in task.items()], axis=0)]
        cells, shades = [], []
        for m in metrics:
            x = part[m].astype(float) if m in part else pd.Series(dtype=float)
            cells.append("–" if x.dropna().empty else f"{x.mean():.2f} ± {x.std():.2f}" if len(x.dropna()) > 1 else f"{x.mean():.2f}")
            good = x.mean() if m in scores and not x.dropna().empty else np.nan
            shades.append(plt.get_cmap("viridis")(good * 0.85 + 0.15) if np.isfinite(good) and 0 <= good <= 1 else "white")
        rows.append([", ".join(f"{k}={v}" for k, v in task.items()), *cells, f"{len(part)}/{seeds}"])
        colors.append(["white", *shades, "white"])
    table = ax.table(cellText=rows, cellColours=colors, colLabels=["task", *metrics, "seeds"], loc="upper left", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(range(len(metrics) + 2))
    for (i, j), c in table.get_celld().items():
        c.set_height(1 / (slots + 1.6))
        if i == 0:
            c.set_text_props(fontweight="bold")
        elif 0 < j <= len(metrics):
            r, g, b, _ = matplotlib.colors.to_rgba(c.get_facecolor())
            c.get_text().set_color("white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else "black")


def report(label: str, core: dict, tasks: pd.DataFrame, comparison: pd.DataFrame, games: list[dict], path: Path) -> None:
    slots = [len(spec["grid"]["env_kwargs"]) if not game_runs(spec, label).empty else 0 for spec in games]
    pairs = [list(zip(games, slots))[i:i + 2] for i in range(0, len(games), 2)]
    heights = [1.5, 8.6] + [max(n for _, n in pair) * 0.24 + 0.75 for pair in pairs]
    fig = plt.figure(figsize=(22, sum(heights) * 1.12))
    grid = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.12 * 10 / sum(heights) * 2)
    head = fig.add_subplot(grid[0])
    head.axis("off")
    head.text(0, 1, f"{label} — {NAMES.get(label, label)}", fontsize=16, fontweight="bold", va="top", transform=head.transAxes)
    row = comparison.set_index("label").loc[label] if label in set(comparison["label"]) else None
    if row is not None:
        rank = "reference, not ranked" if pd.isna(row["mean_rank"]) else f"mean rank {row['mean_rank']:.2f} of {comparison['mean_rank'].notna().sum()}"
        line = "  ·  ".join(f"{t.replace('_', ' ')} {row[t]:.3f}" for t in TESTS)
        head.text(0, 0.72, f"Core comparison ({rank}):  {line}", fontsize=9, va="top", color="#333", transform=head.transAxes)
    head.text(0, 0.52, "Core DelayedEffect tasks, mean over 10 seeds, scored against the exact true credit. Offline = fixed dataset; Stationary = online, policy fixed; "
              "Drift = online, policy changing on a schedule; Learn = online, policy learned from this method's credit (PPO). Grey = not defined there.",
              fontsize=8, va="top", color="#555", transform=head.transAxes)
    core_section(fig, grid[1], tasks, label)
    for k, pair in enumerate(pairs):
        sub = grid[k + 2].subgridspec(1, 2, wspace=0.08)
        for j, (spec, _) in enumerate(pair):
            game_table(fig.add_subplot(sub[j]), spec, label, spec["name"], max(n for _, n in pair) or 1)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", help="experiment files; the one with a settings block is the core benchmark")
    parser.add_argument("--core-dir", help="where src.benchmark wrote the core results, if not the core experiment's out_dir")
    parser.add_argument("--method", nargs="*", help="labels to report; all by default")
    parser.add_argument("--out-dir", default="runs/report")
    args = parser.parse_args()
    specs = [{**json.loads(Path(f).read_text()), "file": f, "name": Path(f).stem.replace("_", " ").capitalize()} for f in args.specs]
    core = next(s for s in specs if "settings" in s)
    games = [s for s in specs if s is not core]
    bench = Path(args.core_dir or core["out_dir"]) / "benchmark"
    tasks, comparison = pd.read_csv(bench / "tasks.csv"), pd.read_csv(bench / "comparison.csv")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for label in args.method or labels(core):
        report(label, core, tasks, comparison, games, out / f"{label}.png")
        print(f"Saved {out / label}.png")
