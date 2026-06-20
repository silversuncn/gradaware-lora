from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = ["gradaware_lora", "lora", "topheavy_lora", "bitfit", "full_ft"]
BASELINES = ["lora", "topheavy_lora", "bitfit", "full_ft"]
METHOD_LABELS = {
    "gradaware_lora": "GradAware-LoRA",
    "lora": "LoRA",
    "topheavy_lora": "TopHeavy-LoRA",
    "bitfit": "BitFit",
    "full_ft": "Full FT",
}
METHOD_COLORS = {
    "gradaware_lora": "#1b9e77",
    "lora": "#7570b3",
    "topheavy_lora": "#d95f02",
    "bitfit": "#66a61e",
    "full_ft": "#e7298a",
}
MODEL_MARKERS = {
    "distilbert-base-uncased": "o",
    "bert-base-uncased": "s",
    "roberta-base": "^",
}
MODEL_LABELS = {
    "distilbert-base-uncased": "DistilBERT",
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
}
TIMESTAMP_RE = re.compile(r"__(\d{8}_\d{6})$")

plt.style.use("seaborn-v0_8-whitegrid")
matplotlib.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.frameon": False,
        "font.size": 10,
    }
)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistical analysis and generate publication-quality figures.")
    parser.add_argument("--results", type=str, default="results.json")
    parser.add_argument("--output_json", type=str, default="statistical_analysis.json")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def native_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): native_value(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [native_value(item) for item in value]
    if isinstance(value, tuple):
        return [native_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def significance_label(p_value: float | None) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def ci95_of_mean(values: pd.Series | np.ndarray | list[float]) -> float:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    count = int(numeric.shape[0])
    if count <= 1:
        return 0.0
    sd = float(numeric.std(ddof=1))
    return float(stats.t.ppf(0.975, count - 1) * sd / math.sqrt(count))


def benjamini_hochberg(p_values: list[float | None]) -> list[float | None]:
    indexed = [(index, value) for index, value in enumerate(p_values) if value is not None and math.isfinite(value)]
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running_min = 1.0
    for rank_from_end, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        candidate = min(1.0, p_value * m / rank)
        running_min = min(running_min, candidate)
        adjusted[original_index] = float(running_min)
    return adjusted


def parse_run_timestamp(run_dir_value: str | None) -> datetime:
    if not run_dir_value:
        return datetime.min
    name = Path(str(run_dir_value)).name
    match = TIMESTAMP_RE.search(name)
    if not match:
        return datetime.min
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def resolve_run_dir(run_dir_value: str | None) -> Path | None:
    if not run_dir_value:
        return None
    run_dir = Path(run_dir_value)
    if run_dir.exists():
        return run_dir
    fallback = PROJECT_ROOT / "artifacts" / "final_runs" / Path(str(run_dir_value)).name
    if fallback.exists():
        return fallback
    return None


def parse_layer_index(text: str) -> int | None:
    match = re.search(r"layer\.(\d+)", text)
    if match:
        return int(match.group(1))
    if str(text).isdigit():
        return int(text)
    return None


def extract_layer_totals(method_notes: dict[str, Any], adapter_config: dict[str, Any]) -> dict[int, float]:
    layer_rank_budget = method_notes.get("layer_rank_budget") or {}
    if layer_rank_budget:
        totals: dict[int, float] = {}
        for key, value in layer_rank_budget.items():
            index = parse_layer_index(str(key))
            if index is not None:
                totals[index] = float(value)
        if totals:
            return totals

    layer_ranks = method_notes.get("layer_ranks") or {}
    if layer_ranks:
        totals = {}
        for key, payload in layer_ranks.items():
            index = parse_layer_index(str(key))
            if index is None or not isinstance(payload, dict):
                continue
            totals[index] = float(sum(float(v) for v in payload.values()))
        if totals:
            return totals

    rank_pattern = adapter_config.get("rank_pattern") or {}
    if rank_pattern:
        totals = {}
        for module_name, value in rank_pattern.items():
            index = parse_layer_index(str(module_name))
            if index is None:
                continue
            totals[index] = totals.get(index, 0.0) + float(value)
        if totals:
            return totals

    return {}


def infer_uniform_total_rank(adapter_config: dict[str, Any]) -> float | None:
    rank = adapter_config.get("r")
    target_modules = adapter_config.get("target_modules") or []
    if rank is None or not target_modules:
        return None
    return float(rank) * float(len(target_modules))


def deduplicate_results(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["task_name", "model_name", "method", "seed"]
    dataframe = raw_df.copy()
    dataframe["_run_timestamp"] = dataframe["run_dir"].map(parse_run_timestamp)
    dataframe["_run_name"] = dataframe["run_dir"].map(lambda x: Path(str(x)).name if pd.notna(x) else "")

    duplicate_sizes = dataframe.groupby(keys).size().reset_index(name="n_rows")
    duplicate_groups = duplicate_sizes[duplicate_sizes["n_rows"] > 1].sort_values(keys).reset_index(drop=True)

    dataframe = dataframe.sort_values(keys + ["_run_timestamp", "_run_name"])
    deduplicated = dataframe.groupby(keys, as_index=False).tail(1).copy().reset_index(drop=True)
    deduplicated = deduplicated.drop(columns=["_run_timestamp", "_run_name"])

    summary = {
        "rows_before": int(len(raw_df)),
        "rows_after": int(len(deduplicated)),
        "dropped_rows": int(len(raw_df) - len(deduplicated)),
        "n_duplicate_key_groups": int(len(duplicate_groups)),
        "duplicate_key_groups": duplicate_groups.to_dict(orient="records"),
        "deduplication_rule": "Kept the latest timestamped run_dir per (task_name, model_name, method, seed).",
    }
    return deduplicated, summary


def load_inputs(results_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    raw_results = read_json(results_path)
    raw_df = pd.DataFrame(raw_results)
    if raw_df.empty:
        raise ValueError(f"No rows found in {results_path}")

    results_df, dedup_summary = deduplicate_results(raw_df)

    metadata_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for row in results_df.to_dict(orient="records"):
        run_dir = resolve_run_dir(row.get("run_dir"))
        metrics_payload: dict[str, Any] = {}
        adapter_payload: dict[str, Any] = {}
        training_curve_files: list[str] = []
        metrics_path = None
        adapter_path = None

        if run_dir is not None:
            metrics_path = run_dir / "metrics.json"
            adapter_path = run_dir / "model" / "adapter_config.json"
            trainer_output_dir = run_dir / "trainer_output"
            if metrics_path.exists():
                metrics_payload = read_json(metrics_path)
            if adapter_path.exists():
                adapter_payload = read_json(adapter_path)
            if trainer_output_dir.exists():
                training_curve_files = sorted(
                    str(path.relative_to(run_dir))
                    for path in trainer_output_dir.rglob("*")
                    if path.is_file()
                )

        method_notes = metrics_payload.get("method_notes") or {}
        train_metrics = metrics_payload.get("train_metrics") or {}
        eval_metrics = metrics_payload.get("eval_metrics") or {}
        layer_totals = extract_layer_totals(method_notes, adapter_payload)
        uniform_total_rank = infer_uniform_total_rank(adapter_payload)
        probe_norms = method_notes.get("probe_gradient_norms") or {}

        metadata_rows.append(
            {
                "task_name": row.get("task_name"),
                "model_name": row.get("model_name"),
                "method": row.get("method"),
                "seed": row.get("seed"),
                "run_dir": str(run_dir) if run_dir is not None else row.get("run_dir"),
                "metrics_path": str(metrics_path) if metrics_path and metrics_path.exists() else None,
                "adapter_config_path": str(adapter_path) if adapter_path and adapter_path.exists() else None,
                "primary_metric": row.get("primary_metric", eval_metrics.get("eval_primary_metric")),
                "primary_metric_name": row.get("primary_metric_name") or metrics_payload.get("primary_metric_name"),
                "eval_accuracy": row.get("eval_accuracy", eval_metrics.get("eval_accuracy")),
                "eval_loss": row.get("eval_loss", eval_metrics.get("eval_loss")),
                "train_loss": train_metrics.get("train_loss"),
                "train_runtime": train_metrics.get("train_runtime"),
                "eval_runtime": eval_metrics.get("eval_runtime"),
                "trainable_parameters": row.get("trainable_parameters"),
                "trainable_percentage": row.get("trainable_percentage"),
                "uniform_total_rank": uniform_total_rank,
                "training_curve_available": bool(training_curve_files),
                "training_curve_file_count": int(len(training_curve_files)),
                "training_curve_files": training_curve_files,
            }
        )

        all_layer_indices = sorted(
            set(layer_totals.keys())
            | {parse_layer_index(str(key)) for key in probe_norms.keys() if parse_layer_index(str(key)) is not None}
        )
        for layer_index in all_layer_indices:
            layer_rows.append(
                {
                    "task_name": row.get("task_name"),
                    "model_name": row.get("model_name"),
                    "method": row.get("method"),
                    "seed": row.get("seed"),
                    "layer_index": int(layer_index),
                    "layer_total_rank": layer_totals.get(layer_index),
                    "probe_gradient_norm": float(probe_norms[str(layer_index)]) if str(layer_index) in probe_norms else probe_norms.get(layer_index),
                    "uniform_total_rank": uniform_total_rank,
                }
            )

    metadata_df = pd.DataFrame(metadata_rows)
    layer_df = pd.DataFrame(layer_rows)
    training_curve_summary = {
        "runs_total": int(len(metadata_df)),
        "runs_with_epoch_level_history": int(metadata_df["training_curve_available"].sum()),
        "runs_without_epoch_level_history": int((~metadata_df["training_curve_available"]).sum()),
        "available_file_examples": sorted(
            {
                item
                for files in metadata_df.loc[metadata_df["training_curve_available"], "training_curve_files"].tolist()
                for item in files
            }
        )[:20],
        "note": "No trainer_state/log_history files were found in current artifacts; only final train/eval metrics are available.",
    }
    return metadata_df.copy(), metadata_df, layer_df, dedup_summary, training_curve_summary


def aligned_pairs(dataframe: pd.DataFrame, baseline_method: str, group_cols: list[str]) -> pd.DataFrame:
    treatment = dataframe[dataframe["method"] == "gradaware_lora"][group_cols + ["seed", "primary_metric"]].rename(
        columns={"primary_metric": "gradaware_primary_metric"}
    )
    baseline = dataframe[dataframe["method"] == baseline_method][group_cols + ["seed", "primary_metric"]].rename(
        columns={"primary_metric": "baseline_primary_metric"}
    )
    merged = treatment.merge(baseline, on=group_cols + ["seed"], how="inner")
    return merged.sort_values(group_cols + ["seed"]).reset_index(drop=True)


def paired_statistics(treatment: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    treatment = np.asarray(treatment, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    differences = treatment - baseline
    n = int(differences.size)

    result: dict[str, Any] = {
        "n_pairs": n,
        "treatment_mean": float(np.mean(treatment)) if n else None,
        "baseline_mean": float(np.mean(baseline)) if n else None,
        "mean_paired_difference": float(np.mean(differences)) if n else None,
        "ci95": {"lower": None, "upper": None},
        "p_value_two_sided": None,
        "t_stat": None,
        "df": n - 1 if n else None,
        "cohens_d_paired": None,
        "difference_std": float(np.std(differences, ddof=1)) if n > 1 else None,
    }
    if n == 0:
        return result
    if n == 1:
        value = float(differences[0])
        result["ci95"] = {"lower": value, "upper": value}
        return result

    mean_diff = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    if sd == 0.0:
        result["ci95"] = {"lower": mean_diff, "upper": mean_diff}
        result["t_stat"] = None
        result["p_value_two_sided"] = 1.0 if mean_diff == 0.0 else 0.0
        result["cohens_d_paired"] = 0.0 if mean_diff == 0.0 else None
        return result

    se = sd / math.sqrt(n)
    t_stat = mean_diff / se
    t_critical = float(stats.t.ppf(0.975, n - 1))
    result["ci95"] = {
        "lower": float(mean_diff - t_critical * se),
        "upper": float(mean_diff + t_critical * se),
    }
    result["t_stat"] = float(t_stat)
    result["p_value_two_sided"] = float(2.0 * stats.t.sf(abs(t_stat), n - 1))
    result["cohens_d_paired"] = float(mean_diff / sd)
    return result


def bootstrap_mean_ci(treatment: np.ndarray, baseline: np.ndarray, n_resamples: int, seed: int) -> dict[str, Any]:
    treatment = np.asarray(treatment, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    differences = treatment - baseline
    n = int(differences.size)
    result = {
        "n_pairs": n,
        "n_resamples": int(n_resamples),
        "bootstrap_mean_difference": float(np.mean(differences)) if n else None,
        "ci95": {"lower": None, "upper": None},
    }
    if n == 0:
        return result
    if n == 1:
        value = float(differences[0])
        result["ci95"] = {"lower": value, "upper": value}
        return result

    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, n, size=(n_resamples, n))
    bootstrap_means = differences[sample_indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    result["ci95"] = {"lower": float(lower), "upper": float(upper)}
    return result


def compute_paired_tests(dataframe: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    per_group: dict[str, Any] = {}
    overall: dict[str, Any] = {}
    adjust_targets: list[dict[str, Any]] = []

    groups = dataframe[["task_name", "model_name"]].drop_duplicates().sort_values(["task_name", "model_name"])
    for _, group in groups.iterrows():
        task_name = group["task_name"]
        model_name = group["model_name"]
        key = f"{task_name}__{model_name}"
        subset = dataframe[(dataframe["task_name"] == task_name) & (dataframe["model_name"] == model_name)]
        per_group[key] = {
            "task_name": task_name,
            "model_name": model_name,
            "primary_metric_names": sorted(subset["primary_metric_name"].dropna().unique().tolist()),
            "comparisons": {},
        }
        for baseline in BASELINES:
            merged = aligned_pairs(subset, baseline, ["task_name", "model_name"])
            stats_result = paired_statistics(
                merged["gradaware_primary_metric"].to_numpy(dtype=float),
                merged["baseline_primary_metric"].to_numpy(dtype=float),
            )
            stats_result["aligned_seeds"] = merged["seed"].tolist()
            stats_result["significance_stars"] = significance_label(stats_result.get("p_value_two_sided"))
            stats_result["p_value_fdr_bh"] = None
            stats_result["significance_stars_fdr_bh"] = "n/a"
            per_group[key]["comparisons"][baseline] = stats_result
            adjust_targets.append(stats_result)

    adjusted = benjamini_hochberg([entry.get("p_value_two_sided") for entry in adjust_targets])
    for entry, adjusted_p in zip(adjust_targets, adjusted):
        entry["p_value_fdr_bh"] = adjusted_p
        entry["significance_stars_fdr_bh"] = significance_label(adjusted_p)

    overall_targets: list[dict[str, Any]] = []
    for baseline in BASELINES:
        merged = aligned_pairs(dataframe, baseline, ["task_name", "model_name"])
        stats_result = paired_statistics(
            merged["gradaware_primary_metric"].to_numpy(dtype=float),
            merged["baseline_primary_metric"].to_numpy(dtype=float),
        )
        stats_result["aligned_pairs"] = merged[["task_name", "model_name", "seed"]].to_dict(orient="records")
        stats_result["significance_stars"] = significance_label(stats_result.get("p_value_two_sided"))
        stats_result["p_value_fdr_bh"] = None
        stats_result["significance_stars_fdr_bh"] = "n/a"
        overall[baseline] = stats_result
        overall_targets.append(stats_result)

    overall_adjusted = benjamini_hochberg([entry.get("p_value_two_sided") for entry in overall_targets])
    for entry, adjusted_p in zip(overall_targets, overall_adjusted):
        entry["p_value_fdr_bh"] = adjusted_p
        entry["significance_stars_fdr_bh"] = significance_label(adjusted_p)

    return per_group, overall


def compute_bootstrap(dataframe: pd.DataFrame, n_resamples: int, seed: int) -> dict[str, Any]:
    overall_pairs = aligned_pairs(dataframe, "lora", ["task_name", "model_name"])
    result: dict[str, Any] = {
        "main_comparison": "gradaware_lora_vs_lora",
        "overall": bootstrap_mean_ci(
            overall_pairs["gradaware_primary_metric"].to_numpy(dtype=float),
            overall_pairs["baseline_primary_metric"].to_numpy(dtype=float),
            n_resamples=n_resamples,
            seed=seed,
        ),
        "by_dataset_model": {},
    }
    groups = dataframe[["task_name", "model_name"]].drop_duplicates().sort_values(["task_name", "model_name"])
    for _, group in groups.iterrows():
        task_name = group["task_name"]
        model_name = group["model_name"]
        key = f"{task_name}__{model_name}"
        subset = dataframe[(dataframe["task_name"] == task_name) & (dataframe["model_name"] == model_name)]
        merged = aligned_pairs(subset, "lora", ["task_name", "model_name"])
        result["by_dataset_model"][key] = {
            "task_name": task_name,
            "model_name": model_name,
            **bootstrap_mean_ci(
                merged["gradaware_primary_metric"].to_numpy(dtype=float),
                merged["baseline_primary_metric"].to_numpy(dtype=float),
                n_resamples=n_resamples,
                seed=seed,
            ),
        }
    return result


def short_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def balanced_cell_means(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby(["task_name", "model_name", "method"], as_index=False)
        .agg(
            primary_metric_mean=("primary_metric", "mean"),
            eval_accuracy_mean=("eval_accuracy", "mean"),
            trainable_percentage_mean=("trainable_percentage", "mean"),
            trainable_parameters_mean=("trainable_parameters", "mean"),
        )
    )


def compute_rank_table(dataframe: pd.DataFrame) -> pd.DataFrame:
    cell_means = balanced_cell_means(dataframe)
    pieces = []
    for (task_name, model_name), subset in cell_means.groupby(["task_name", "model_name"]):
        ranked = subset.copy()
        ranked["rank"] = ranked["primary_metric_mean"].rank(ascending=False, method="average")
        pieces.append(ranked)
    return pd.concat(pieces, ignore_index=True)


def save_figure(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_main_comparison_bar(dataframe: pd.DataFrame, overall_tests: dict[str, Any], output_path: Path) -> str:
    cell_means = balanced_cell_means(dataframe)
    summary = (
        cell_means.groupby("method")["primary_metric_mean"]
        .agg(["mean", "std", "count"])
        .reindex(METHOD_ORDER)
    )
    means = summary["mean"].fillna(0.0).to_numpy(dtype=float)
    cis = [ci95_of_mean(cell_means[cell_means["method"] == method]["primary_metric_mean"]) for method in METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x,
        means,
        yerr=cis,
        capsize=5,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        alpha=0.92,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method] for method in METHOD_ORDER], rotation=18, ha="right")
    ax.set_ylabel("Mean primary metric across dataset × model cells")
    ax.set_title("Main comparison with 95% CIs")
    ymax = float(np.max(means + np.array(cis))) if len(means) else 1.0
    ax.set_ylim(0.0, max(1.0, ymax + 0.12))
    for idx, method in enumerate(METHOD_ORDER):
        if method == "gradaware_lora":
            ax.text(idx, means[idx] + cis[idx] + 0.02, "ref", ha="center", va="bottom", fontsize=10, fontweight="bold")
            continue
        label = overall_tests[method].get("significance_stars", "n/a")
        ax.text(idx, means[idx] + cis[idx] + 0.02, label, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(
        0.02,
        -0.25,
        "Bars average seed-averaged task × model cells. Stars denote pooled paired t-tests against GradAware-LoRA.",
        transform=ax.transAxes,
        fontsize=9,
    )
    return save_figure(fig, output_path)


def plot_per_dataset_accuracy_heatmap(dataframe: pd.DataFrame, output_path: Path) -> str:
    pivot = (
        dataframe.groupby(["task_name", "method"], as_index=False)["eval_accuracy"]
        .mean()
        .pivot(index="task_name", columns="method", values="eval_accuracy")
        .reindex(index=sorted(dataframe["task_name"].dropna().unique().tolist()), columns=METHOD_ORDER)
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0.45, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([METHOD_LABELS[col] for col in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Per-dataset accuracy heatmap")
    for row_index in range(pivot.shape[0]):
        for col_index in range(pivot.shape[1]):
            value = pivot.iloc[row_index, col_index]
            label = "n/a" if pd.isna(value) else f"{value:.3f}"
            ax.text(col_index, row_index, label, ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.9, label="Mean eval accuracy")
    return save_figure(fig, output_path)


def plot_method_ranking_across_datasets(dataframe: pd.DataFrame, output_path: Path) -> str:
    rank_df = compute_rank_table(dataframe)
    summary = (
        rank_df.groupby("method")["rank"]
        .agg(["mean", "std", "count"])
        .reindex(METHOD_ORDER)
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    methods = summary.index.tolist()
    y = np.arange(len(methods))
    means = summary["mean"].to_numpy(dtype=float)
    cis = [ci95_of_mean(rank_df[rank_df["method"] == method]["rank"]) for method in methods]
    ax.barh(y, means, xerr=cis, color=[METHOD_COLORS[m] for m in methods], alpha=0.9, capsize=5)

    rng = np.random.default_rng(42)
    for idx, method in enumerate(methods):
        values = rank_df[rank_df["method"] == method]["rank"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(values, np.full_like(values, idx, dtype=float) + jitter, color="black", s=14, alpha=0.45)

    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_xlabel("Average rank across dataset × model cells (lower is better)")
    ax.set_title("Method ranking across datasets")
    ax.invert_yaxis()
    return save_figure(fig, output_path)


def plot_gradaware_rank_allocation(layer_df: pd.DataFrame, output_path: Path) -> str:
    models = sorted(layer_df["model_name"].dropna().unique().tolist())
    if not models:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No layer rank metadata available.", ha="center", va="center")
        ax.set_axis_off()
        return save_figure(fig, output_path)

    fig, axes = plt.subplots(1, len(models), figsize=(5.4 * len(models), 4.8), squeeze=False)
    axes = axes[0]
    for ax, model_name in zip(axes, models):
        subset = layer_df[layer_df["model_name"] == model_name].copy()
        observed_layers = sorted(subset["layer_index"].dropna().astype(int).unique().tolist())
        for method in ["gradaware_lora", "topheavy_lora"]:
            method_subset = subset[(subset["method"] == method) & subset["layer_total_rank"].notna()]
            if method_subset.empty:
                continue
            means = []
            cis = []
            for layer_index in observed_layers:
                values = method_subset[method_subset["layer_index"] == layer_index]["layer_total_rank"]
                means.append(float(pd.to_numeric(values, errors="coerce").mean()))
                cis.append(ci95_of_mean(values))
            ax.errorbar(
                observed_layers,
                means,
                yerr=cis,
                marker="o",
                linewidth=2,
                capsize=4,
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )

        uniform_values = subset[(subset["method"] == "lora") & subset["uniform_total_rank"].notna()]["uniform_total_rank"]
        if not uniform_values.empty:
            uniform_rank = float(uniform_values.mean())
            ax.plot(
                observed_layers,
                [uniform_rank for _ in observed_layers],
                linestyle=":",
                linewidth=2,
                color=METHOD_COLORS["lora"],
                label=f"LoRA uniform ({uniform_rank:.0f})",
            )

        ax.set_title(short_model_name(model_name))
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Total allocated rank")
        ax.set_xticks(observed_layers)
        ax.legend(fontsize=8)
    fig.suptitle("GradAware rank allocation visualization", y=1.02, fontsize=14, fontweight="bold")
    return save_figure(fig, output_path)


def plot_training_curves_placeholder(metadata_df: pd.DataFrame, training_curve_summary: dict[str, Any], output_path: Path) -> str:
    with_history = training_curve_summary["runs_with_epoch_level_history"]
    without_history = training_curve_summary["runs_without_epoch_level_history"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar([0, 1], [with_history, without_history], color=["#1b9e77", "#d95f02"], width=0.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Runs with epoch-level\nhistory", "Runs without epoch-level\nhistory"])
    ax.set_ylabel("Number of runs")
    ax.set_title("Training curves (loss over epochs)")
    ax.text(
        0.5,
        0.90,
        f"{with_history}/{len(metadata_df)} runs contain epoch-level logs",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.68,
        "Current artifacts store only final train/eval metrics.\nNo trainer_state.json or log_history was found, so true loss-vs-epoch curves cannot be reconstructed without rerunning with logging enabled.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f7f7f7", "edgecolor": "#bbbbbb"},
    )
    ax.set_ylim(0, max(1, without_history) * 1.18)
    return save_figure(fig, output_path)


def plot_parameter_efficiency_scatter(dataframe: pd.DataFrame, output_path: Path) -> str:
    cell_means = balanced_cell_means(dataframe)
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name in sorted(cell_means["model_name"].dropna().unique().tolist()):
        for method in METHOD_ORDER:
            subset = cell_means[(cell_means["model_name"] == model_name) & (cell_means["method"] == method)]
            if subset.empty:
                continue
            ax.scatter(
                subset["trainable_percentage_mean"],
                subset["primary_metric_mean"],
                s=95,
                alpha=0.85,
                color=METHOD_COLORS[method],
                marker=MODEL_MARKERS.get(model_name, "o"),
                edgecolor="black",
                linewidth=0.4,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (%) [log scale]")
    ax.set_ylabel("Mean primary metric")
    ax.set_title("Parameter efficiency scatter plot")

    method_handles = [
        Line2D([0], [0], marker="o", color="w", label=METHOD_LABELS[m], markerfacecolor=METHOD_COLORS[m], markersize=9)
        for m in METHOD_ORDER
    ]
    model_handles = [
        Line2D([0], [0], marker=MODEL_MARKERS[m], color="black", linestyle="None", label=short_model_name(m), markersize=8)
        for m in sorted(cell_means["model_name"].dropna().unique().tolist())
    ]
    legend1 = ax.legend(handles=method_handles, title="Method", loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=model_handles, title="Model", loc="lower center")
    return save_figure(fig, output_path)


def plot_statistical_significance_matrix(per_group_tests: dict[str, Any], output_path: Path) -> str:
    group_keys = sorted(per_group_tests.keys())
    matrix = np.full((len(group_keys), len(BASELINES)), np.nan)
    labels: list[list[str]] = [["" for _ in BASELINES] for _ in group_keys]
    for row_index, key in enumerate(group_keys):
        for col_index, baseline in enumerate(BASELINES):
            comparison = per_group_tests[key]["comparisons"][baseline]
            p_value = comparison.get("p_value_two_sided")
            diff = comparison.get("mean_paired_difference")
            if p_value is None:
                labels[row_index][col_index] = "n/a"
                continue
            sign = 1.0 if (diff is None or diff >= 0.0) else -1.0
            matrix[row_index, col_index] = sign * (-math.log10(max(p_value, 1e-12)))
            labels[row_index][col_index] = comparison.get("significance_stars", "n/a")

    fig, ax = plt.subplots(figsize=(9.8, max(5.5, 0.42 * len(group_keys))))
    vmax = max(2.0, float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 2.0)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(BASELINES)))
    ax.set_xticklabels([METHOD_LABELS[b] for b in BASELINES], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(group_keys)))
    ax.set_yticklabels([
        f"{per_group_tests[key]['task_name']} / {short_model_name(per_group_tests[key]['model_name'])}"
        for key in group_keys
    ])
    ax.set_title("Statistical significance matrix\nSigned -log10(p): positive means GradAware-LoRA > baseline")
    for row_index in range(len(group_keys)):
        for col_index in range(len(BASELINES)):
            display = labels[row_index][col_index]
            ax.text(col_index, row_index, display, ha="center", va="center", color="black", fontsize=8.5, fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.95, label="Signed -log10(raw p-value)")
    return save_figure(fig, output_path)


def plot_per_model_comparison(dataframe: pd.DataFrame, output_path: Path) -> str:
    task_means = (
        dataframe.groupby(["model_name", "task_name", "method"], as_index=False)["primary_metric"]
        .mean()
        .rename(columns={"primary_metric": "task_mean_metric"})
    )
    models = sorted(task_means["model_name"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.16
    x = np.arange(len(models))
    for offset_index, method in enumerate(METHOD_ORDER):
        means = []
        cis = []
        for model_name in models:
            values = task_means[(task_means["model_name"] == model_name) & (task_means["method"] == method)]["task_mean_metric"]
            means.append(float(pd.to_numeric(values, errors="coerce").mean()))
            cis.append(ci95_of_mean(values))
        positions = x + (offset_index - (len(METHOD_ORDER) - 1) / 2.0) * width
        ax.bar(
            positions,
            means,
            width=width,
            yerr=cis,
            capsize=4,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([short_model_name(model_name) for model_name in models])
    ax.set_ylabel("Mean primary metric across tasks")
    ax.set_title("Per-model comparison")
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylim(0.0, 1.05)
    return save_figure(fig, output_path)


def main() -> None:
    args = parse_args()
    results_path = resolve_path(args.results).resolve()
    output_json_path = resolve_path(args.output_json).resolve()
    figures_dir = resolve_path(args.figures_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    analysis_df, metadata_df, layer_df, dedup_summary, training_curve_summary = load_inputs(results_path)
    per_group_tests, overall_tests = compute_paired_tests(analysis_df)
    bootstrap = compute_bootstrap(analysis_df, n_resamples=args.bootstrap_resamples, seed=args.bootstrap_seed)
    rank_df = compute_rank_table(analysis_df)
    cell_means_df = balanced_cell_means(analysis_df)

    figures = {
        "main_comparison_bar_chart": plot_main_comparison_bar(
            analysis_df, overall_tests, figures_dir / "01_main_comparison_bar.png"
        ),
        "per_dataset_accuracy_heatmap": plot_per_dataset_accuracy_heatmap(
            analysis_df, figures_dir / "02_per_dataset_accuracy_heatmap.png"
        ),
        "method_ranking_across_datasets": plot_method_ranking_across_datasets(
            analysis_df, figures_dir / "03_method_ranking_across_datasets.png"
        ),
        "gradaware_rank_allocation_visualization": plot_gradaware_rank_allocation(
            layer_df, figures_dir / "04_gradaware_rank_allocation_visualization.png"
        ),
        "training_curves_loss_over_epochs": plot_training_curves_placeholder(
            metadata_df, training_curve_summary, figures_dir / "05_training_curves_loss_over_epochs.png"
        ),
        "parameter_efficiency_scatter_plot": plot_parameter_efficiency_scatter(
            analysis_df, figures_dir / "06_parameter_efficiency_scatter_plot.png"
        ),
        "statistical_significance_matrix": plot_statistical_significance_matrix(
            per_group_tests, figures_dir / "07_statistical_significance_matrix.png"
        ),
        "per_model_comparison": plot_per_model_comparison(
            analysis_df, figures_dir / "08_per_model_comparison.png"
        ),
    }

    payload = {
        "inputs": {
            "results_json": str(results_path),
            "rows_after_dedup": int(len(analysis_df)),
            "available_tasks": sorted(analysis_df["task_name"].dropna().unique().tolist()),
            "available_models": sorted(analysis_df["model_name"].dropna().unique().tolist()),
            "available_methods": sorted(analysis_df["method"].dropna().unique().tolist()),
            "primary_metric_by_task": {
                task_name: sorted(
                    analysis_df.loc[analysis_df["task_name"] == task_name, "primary_metric_name"].dropna().unique().tolist()
                )
                for task_name in sorted(analysis_df["task_name"].dropna().unique().tolist())
            },
        },
        "deduplication": dedup_summary,
        "training_curve_availability": training_curve_summary,
        "analysis_notes": [
            "Paired tests align GradAware-LoRA with each baseline by dataset, model, and seed after deduplicating repeated reruns.",
            "Reported 95% CIs in paired tests are t-based confidence intervals on the mean paired difference.",
            "Bootstrap CIs use 1000 paired resamples for the main GradAware-LoRA vs LoRA comparison.",
            "Benjamini-Hochberg adjusted p-values are included for the family of per-dataset×model comparisons and for the pooled overall comparisons.",
            "The requested training-curves figure is an availability diagnostic because epoch-level loss histories were not saved in the current artifact bundle.",
        ],
        "paired_t_tests_by_dataset_model": per_group_tests,
        "paired_t_tests_overall": overall_tests,
        "bootstrap_main_comparison": bootstrap,
        "summary_tables": {
            "method_balanced_means": (
                cell_means_df.groupby("method")["primary_metric_mean"].mean().reindex(METHOD_ORDER).to_dict()
            ),
            "dataset_method_accuracy_means": (
                analysis_df.groupby(["task_name", "method"], as_index=False)["eval_accuracy"].mean().to_dict(orient="records")
            ),
            "method_ranking": (
                rank_df.groupby("method")["rank"].agg(["mean", "std", "count"]).reindex(METHOD_ORDER).reset_index().to_dict(orient="records")
            ),
            "parameter_efficiency_points": cell_means_df.to_dict(orient="records"),
        },
        "figures": figures,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(native_value(payload), indent=2, sort_keys=True))
    print(output_json_path)
    for figure_path in figures.values():
        print(figure_path)


if __name__ == "__main__":
    main()
