from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPORTS = ROOT / "outputs" / "reports"
FIGURES = ROOT / "outputs" / "figures"


def save_regression_comparison() -> None:
    metrics = pd.read_csv(REPORTS / "final_national_population_regression_metrics.csv")
    rows = metrics[metrics["target"].eq("next_year_sgg_school_age_pop_0_19")].copy()
    rows = rows[rows["model"].isin(["baseline_previous_year_population", "ridge", "random_forest"])]
    label_map = {
        "baseline_previous_year_population": "Naive\n전년 유지",
        "ridge": "Base\nRidge",
        "random_forest": "Tuned\nRandomForest",
    }
    rows["label"] = rows["model"].map(label_map)
    rows["order"] = rows["model"].map({k: i for i, k in enumerate(label_map)})
    rows = rows.sort_values("order")
    rows.to_csv(REPORTS / "regression_base_vs_tuned_comparison.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(rows["label"], rows["mae"], color=["#94a3b8", "#60a5fa", "#2563eb"])
    ax.set_title("회귀모델 비교: 베이스라인 vs 튜닝 모델", fontweight="bold")
    ax.set_ylabel("MAE 낮을수록 좋음")
    ax.bar_label(bars, labels=[f"{v:,.0f}" for v in rows["mae"]], padding=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "regression_base_vs_tuned_comparison.png", dpi=170)
    plt.close(fig)


def save_classifier_comparison() -> None:
    metrics = pd.read_csv(REPORTS / "final_supervised_closure_classifier_metrics.csv")
    label_map = {
        "baseline_logistic_regression": "Base\nLogistic",
        "final_context_hist_gradient_boosting": "Tuned\nHistGB",
    }
    rows = metrics[metrics["model"].isin(label_map)].copy()
    rows["label"] = rows["model"].map(label_map)
    rows["order"] = rows["model"].map({k: i for i, k in enumerate(label_map)})
    rows = rows.sort_values("order")
    rows.to_csv(REPORTS / "classification_base_vs_tuned_comparison.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = range(len(rows))
    width = 0.22
    for offset, metric, color in [
        (-width, "roc_auc", "#60a5fa"),
        (0, "pr_auc", "#f97316"),
        (width, "f1", "#22c55e"),
    ]:
        values = rows[metric].astype(float).tolist()
        bars = ax.bar([i + offset for i in x], values, width=width, label=metric.upper(), color=color)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=3, fontsize=8)
    ax.set_xticks(list(x), rows["label"])
    ax.set_ylim(0, 1.05)
    ax.set_title("분류모델 비교: Logistic 베이스라인 vs 튜닝 모델", fontweight="bold")
    ax.set_ylabel("점수 높을수록 좋음")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "classification_base_vs_tuned_comparison.png", dpi=170)
    plt.close(fig)


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    save_regression_comparison()
    save_classifier_comparison()
    print("saved:", FIGURES / "regression_base_vs_tuned_comparison.png")
    print("saved:", FIGURES / "classification_base_vs_tuned_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
