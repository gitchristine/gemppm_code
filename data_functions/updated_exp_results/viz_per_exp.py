"""
Visualization script for model comparison CSVs.
Renames 'Oyamada' (and 'Oyamada*') to 'FT-LLM' in all plots.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Setup ----
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent

sns.set_theme(style="whitegrid", context="talk")
PALETTE = {"DATL-TAIA": "#1f77b4", "FT-LLM": "#d62728"}


def rename_models(df: pd.DataFrame) -> pd.DataFrame:
    """Map Oyamada / Oyamada* -> FT-LLM."""
    df = df.copy()
    df["Model"] = df["Model"].replace({"Oyamada": "FT-LLM", "Oyamada*": "FT-LLM"})
    return df


# =====================================================
# 1. model_results.csv  (backbone scaling)
# =====================================================
df1 = rename_models(pd.read_csv(DATA_DIR / "diff_backbone.csv"))
df1 = df1[df1["Backbone"] != "GPT2"]
# Order backbones by parameter size
backbone_order = ["TinyLLM", "Qwen2.5", "Llama3.2"]
df1["Backbone"] = pd.Categorical(df1["Backbone"], categories=backbone_order, ordered=True)

# 1a. Accuracy by backbone, faceted by dataset, with CI error bars
g = sns.catplot(
    data=df1, x="Backbone", y="NA_Acc", hue="Model",
    col="Dataset", col_wrap=2, kind="bar",
    palette=PALETTE, height=4, aspect=1.3,
    errorbar=None,
)
for ax, dataset in zip(g.axes.flat, df1["Dataset"].unique()):
    sub = df1[df1["Dataset"] == dataset]
    for container, model in zip(ax.containers, sub["Model"].unique()):
        m_sub = sub[sub["Model"] == model].sort_values("Backbone")
        ax.errorbar(
            x=[bar.get_x() + bar.get_width() / 2 for bar in container],
            y=m_sub["NA_Acc"].values,
            yerr=m_sub["NA_CI"].values,
            fmt="none", ecolor="black", capsize=3, linewidth=1,
        )
g.set_titles("{col_name}")
g.set_axis_labels("Backbone", "Next-Activity Accuracy (%)")
g.fig.suptitle("Accuracy by Backbone (with 95% CI)", y=1.02)
plt.savefig(OUT_DIR / "01a_accuracy_by_backbone.png", dpi=150, bbox_inches="tight")
plt.close()

# 1a-bis. RT_MAE by backbone, faceted by dataset, with CI error bars
g = sns.catplot(
    data=df1, x="Backbone", y="RT_MAE", hue="Model",
    col="Dataset", col_wrap=2, kind="bar",
    palette=PALETTE, height=4, aspect=1.3,
    errorbar=None, sharey=False,
)
for ax, dataset in zip(g.axes.flat, df1["Dataset"].unique()):
    sub = df1[df1["Dataset"] == dataset]
    for container, model in zip(ax.containers, sub["Model"].unique()):
        m_sub = sub[sub["Model"] == model].sort_values("Backbone")
        ax.errorbar(
            x=[bar.get_x() + bar.get_width() / 2 for bar in container],
            y=m_sub["RT_MAE"].values,
            yerr=m_sub["MAE_CI"].values,
            fmt="none", ecolor="black", capsize=3, linewidth=1,
        )
g.set_titles("{col_name}")
g.set_axis_labels("Backbone", "Remaining-Time MAE (lower is better)")
g.fig.suptitle("RT MAE by Backbone (with 95% CI)", y=1.02)
plt.savefig(OUT_DIR / "01b_mae_by_backbone.png", dpi=150, bbox_inches="tight")
plt.close()

# 1b. Accuracy vs Runtime trade-off (one subplot per backbone)
backbones = [b for b in backbone_order if b in df1["Backbone"].unique()]
fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), 5), sharey=True)
if len(backbones) == 1:
    axes = [axes]

dataset_markers = {
    "BPI2012": "o",
    "BPI2017": "X",
    "BPI2015_2": "s",
    "BPI2020_DD": "D",
}

for ax, backbone in zip(axes, backbones):
    sub = df1[df1["Backbone"] == backbone]
    for model, color in PALETTE.items():
        for dataset, marker in dataset_markers.items():
            point = sub[(sub["Model"] == model) & (sub["Dataset"] == dataset)]
            if point.empty:
                continue
            ax.scatter(
                point["Runtime_Hours"], point["NA_Acc"],
                color=color, marker=marker, s=180,
                edgecolor="white", linewidth=1.2,
                label=f"{model} – {dataset}",
            )
    ax.set_title(backbone)
    ax.set_xlabel("Runtime (hours)")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Next-Activity Accuracy (%)")

# Two-section legend: colors for Model, shapes for Dataset
from matplotlib.lines import Line2D

model_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
           markersize=11, label=model)
    for model, color in PALETTE.items()
]
dataset_handles = [
    Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray",
           markersize=11, label=dataset)
    for dataset, marker in dataset_markers.items()
]

legend_handles = (
    [Line2D([0], [0], color="none", label="Model")]
    + model_handles
    + [Line2D([0], [0], color="none", label="")]
    + [Line2D([0], [0], color="none", label="Dataset")]
    + dataset_handles
)
fig.legend(
    handles=legend_handles,
    loc="center right", fontsize=10, framealpha=0.95,
    bbox_to_anchor=(1.15, 0.5), handletextpad=0.6,
)

fig.suptitle("Accuracy vs Runtime Trade-off", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "02_accuracy_vs_runtime.png", dpi=150, bbox_inches="tight")
plt.close()


# 1c. Remaining-Time MAE vs Runtime trade-off (one subplot per backbone)
fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), 5), sharey=False)
if len(backbones) == 1:
    axes = [axes]

for ax, backbone in zip(axes, backbones):
    sub = df1[df1["Backbone"] == backbone]
    for model, color in PALETTE.items():
        for dataset, marker in dataset_markers.items():
            point = sub[(sub["Model"] == model) & (sub["Dataset"] == dataset)]
            if point.empty:
                continue
            ax.scatter(
                point["Runtime_Hours"], point["RT_MAE"],
                color=color, marker=marker, s=180,
                edgecolor="white", linewidth=1.2,
            )
    ax.set_title(backbone)
    ax.set_xlabel("Runtime (hours)")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Remaining-Time MAE (lower is better)")

# Reuse the same two-section legend
fig.legend(
    handles=legend_handles,
    loc="center right", fontsize=10, framealpha=0.95,
    bbox_to_anchor=(1.15, 0.5), handletextpad=0.6,
)

fig.suptitle("Remaining-Time MAE vs Runtime Trade-off", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "02b_mae_vs_runtime.png", dpi=150, bbox_inches="tight")
plt.close()


# =====================================================
# 2. training_curve_results.csv  (data scaling)
# =====================================================
df2 = rename_models(pd.read_csv(DATA_DIR / "data_percentage.csv"))

# 2a. Learning curves: Accuracy vs cumulative training data (area plot)
# Each step (13%, 26%, ...) includes all prior data, so we use filled areas
# anchored at 0 to visually communicate the cumulative/nested nature.
datasets = df2["Dataset"].unique()
n = len(datasets)
fig, axes = plt.subplots(
    nrows=(n + 1) // 2, ncols=2, figsize=(13, 4 * ((n + 1) // 2)), sharey=True
)
axes = axes.flat

for ax, dataset in zip(axes, datasets):
    sub = df2[df2["Dataset"] == dataset]
    for model, color in PALETTE.items():
        m = sub[sub["Model"] == model].sort_values("Training_Percentage")
        # Anchor area at 0 to show "data accumulates from zero"
        x = m["Training_Percentage"].values
        y = m["NA_Acc"].values
        ax.fill_between(x, 0, y, color=color, alpha=0.25, label=f"{model}")
        ax.plot(x, y, color=color, marker="o", linewidth=2)
        # CI ribbon on the line itself
        ax.fill_between(
            x, y - m["NA_CI"].values, y + m["NA_CI"].values,
            color=color, alpha=0.15, linewidth=0,
        )
    ax.set_title(dataset)
    ax.set_xlabel("Cumulative training data (%)")
    ax.set_ylabel("Next-Activity Accuracy (%)")
    ax.set_xlim(0, 70)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=10)

# Hide unused subplots if odd number of datasets
for ax in list(axes)[n:]:
    ax.set_visible(False)

fig.suptitle("Learning Curves: Accuracy vs Cumulative Training Data", y=1.00)
fig.tight_layout()
plt.savefig(OUT_DIR / "03_learning_curves_acc.png", dpi=150, bbox_inches="tight")
plt.close()

# 2b. Same but for runtime MAE (lower is better)
g = sns.relplot(
    data=df2, x="Training_Percentage", y="RT_MAE",
    hue="Model", col="Dataset", col_wrap=2, kind="line",
    palette=PALETTE, marker="o", height=4, aspect=1.3,
    facet_kws={"sharey": False},
)
for ax, dataset in zip(g.axes.flat, df2["Dataset"].unique()):
    sub = df2[df2["Dataset"] == dataset]
    for model, color in PALETTE.items():
        m = sub[sub["Model"] == model].sort_values("Training_Percentage")
        ax.fill_between(
            m["Training_Percentage"], m["RT_MAE"] - m["MAE_CI"], m["RT_MAE"] + m["MAE_CI"],
            color=color, alpha=0.2,
        )
g.set_titles("{col_name}")
g.set_axis_labels("Training data (%)", "Remaining-Time MAE")
g.fig.suptitle("Runtime MAE vs Training Size (lower is better)", y=1.02)
plt.savefig(OUT_DIR / "04_learning_curves_mae.png", dpi=150, bbox_inches="tight")
plt.close()


# =====================================================
# 3. prefix_results.csv  (prefix-length sensitivity)
# =====================================================
df3 = rename_models(pd.read_csv(DATA_DIR / "prefix_bucket.csv"))

prefix_order = ["10-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
df3["Prefix_Percentage"] = pd.Categorical(
    df3["Prefix_Percentage"], categories=prefix_order, ordered=True
)

# 3a. Lines: Accuracy vs prefix bucket
g = sns.relplot(
    data=df3, x="Prefix_Percentage", y="NA_Acc",
    hue="Model", col="Dataset", col_wrap=2, kind="line",
    palette=PALETTE, marker="o", height=4, aspect=1.3, sort=False,
)
for ax, dataset in zip(g.axes.flat, df3["Dataset"].unique()):
    sub = df3[df3["Dataset"] == dataset]
    for model, color in PALETTE.items():
        m = sub[sub["Model"] == model].sort_values("Prefix_Percentage")
        ax.fill_between(
            range(len(m)), m["NA_Acc"] - m["NA_CI"], m["NA_Acc"] + m["NA_CI"],
            color=color, alpha=0.2,
        )
g.set_titles("{col_name}")
g.set_axis_labels("Prefix length (% of trace)", "Next-Activity Accuracy (%)")
g.fig.suptitle("Accuracy vs Prefix Length", y=1.02)
plt.savefig(OUT_DIR / "05a_prefix_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()

# 3b. Lines: RT_MAE vs prefix bucket
g = sns.relplot(
    data=df3, x="Prefix_Percentage", y="RT_MAE",
    hue="Model", col="Dataset", col_wrap=2, kind="line",
    palette=PALETTE, marker="o", height=4, aspect=1.3, sort=False,
    facet_kws={"sharey": False},
)
for ax, dataset in zip(g.axes.flat, df3["Dataset"].unique()):
    sub = df3[df3["Dataset"] == dataset]
    for model, color in PALETTE.items():
        m = sub[sub["Model"] == model].sort_values("Prefix_Percentage")
        ax.fill_between(
            range(len(m)), m["RT_MAE"] - m["MAE_CI"], m["RT_MAE"] + m["MAE_CI"],
            color=color, alpha=0.2,
        )
g.set_titles("{col_name}")
g.set_axis_labels("Prefix length (% of trace)", "Remaining-Time MAE (lower is better)")
g.fig.suptitle("RT MAE vs Prefix Length", y=1.02)
plt.savefig(OUT_DIR / "05b_prefix_mae.png", dpi=150, bbox_inches="tight")
plt.close()

# 3c. Heatmap: dataset x prefix, Accuracy (DATL-TAIA)
pivot_acc = (
    df3[df3["Model"] == "DATL-TAIA"]
    .pivot(index="Dataset", columns="Prefix_Percentage", values="NA_Acc")
)
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_acc, annot=True, fmt=".1f", cmap="viridis", cbar_kws={"label": "Accuracy (%)"})
plt.title("DATL-TAIA: Accuracy by Dataset × Prefix Length")
plt.tight_layout()
plt.savefig(OUT_DIR / "06a_prefix_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# 3d. Heatmap: dataset x prefix, RT_MAE (DATL-TAIA)
pivot_mae = (
    df3[df3["Model"] == "DATL-TAIA"]
    .pivot(index="Dataset", columns="Prefix_Percentage", values="RT_MAE")
)
plt.figure(figsize=(8, 4))
sns.heatmap(pivot_mae, annot=True, fmt=".1f", cmap="viridis_r", cbar_kws={"label": "RT MAE"})
plt.title("DATL-TAIA: RT MAE by Dataset × Prefix Length")
plt.tight_layout()
plt.savefig(OUT_DIR / "06b_prefix_mae_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Done. Figures saved to ./{OUT_DIR}/")