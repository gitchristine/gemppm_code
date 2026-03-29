"""
Visualization for backbone scale comparison:
D-TAIA with TinyLLM (10M) vs Qwen2.5-1.5B (1500M)
across one dataset per entropy type.
Produces a single figure with three grouped bar subplots:
  - NA Accuracy
  - F1-Macro
  - RT MAE
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BACKBONE_COLOURS = {
    "TinyLLM":       "#1f77b4",   # blue
    "Qwen2.5-1.5B":  "#2ca02c",   # green
}
ENTROPY_ORDER  = ["High", "Medium", "Low"]
ENTROPY_LABELS = {
    "High":   "BPI2015_2\n(High Entropy)",
    "Medium": "BPI2015_1\n(Medium Entropy)",
    "Low":    "BPI2012\n(Low Entropy)",
}

FONT = {"family": "sans-serif", "size": 12}
plt.rc("font", **FONT)


# ---------------------------------------------------------------------------
# Helper: one grouped-bar subplot
# ---------------------------------------------------------------------------
def backbone_bar(ax, df, metric_col, ci_col, title, ylabel,
                 best_is_min=False, show_legend=False):
    backbones  = list(BACKBONE_COLOURS.keys())
    n_backbones = len(backbones)
    bar_w      = 0.28
    group_gap  = 0.10
    group_w    = n_backbones * bar_w + group_gap
    x_centers  = np.arange(len(ENTROPY_ORDER)) * group_w

    for i, bb in enumerate(backbones):
        offsets = x_centers + (i - n_backbones / 2 + 0.5) * bar_w
        vals, cis = [], []
        for etype in ENTROPY_ORDER:
            row = df[(df["Entropy_Type"] == etype) & (df["Backbone"] == bb)]
            if row.empty:
                vals.append(0); cis.append(0)
            else:
                vals.append(float(row[metric_col].values[0]))
                cis.append(float(row[ci_col].values[0]))

        ax.bar(offsets, vals, width=bar_w,
               color=BACKBONE_COLOURS[bb], label=bb,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(offsets, vals, yerr=cis,
                    fmt="none", color="black", capsize=4, linewidth=1.2)

        # Annotate gain/loss arrow between bars within each group
        if i == 1:   # annotate on the Qwen bar
            for j, etype in enumerate(ENTROPY_ORDER):
                tiny_row = df[(df["Entropy_Type"] == etype) &
                              (df["Backbone"] == "TinyLLM")]
                qwen_row = df[(df["Entropy_Type"] == etype) &
                              (df["Backbone"] == bb)]
                if tiny_row.empty or qwen_row.empty:
                    continue
                tiny_val = float(tiny_row[metric_col].values[0])
                qwen_val = float(qwen_row[metric_col].values[0])
                delta    = qwen_val - tiny_val
                sign     = "+" if delta >= 0 else ""
                offset_x = x_centers[j] + (i - n_backbones / 2 + 0.5) * bar_w
                # place label just above the taller bar
                y_top = max(tiny_val, qwen_val) + float(
                    qwen_row[ci_col].values[0]) + (0.012 * qwen_val
                                                   if not best_is_min
                                                   else 0.5)
                fmt_str = f"{sign}{delta:.1f}" if best_is_min else f"{sign}{delta:.1f}"
                ax.text(offset_x, y_top, fmt_str,
                        ha="center", va="bottom", fontsize=9.5,
                        color="#333333", fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels([ENTROPY_LABELS[e] for e in ENTROPY_ORDER],
                       fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    if show_legend:
        handles = [mpatches.Patch(color=BACKBONE_COLOURS[b], label=b)
                   for b in backbones]
        ax.legend(handles=handles, fontsize=10, framealpha=0.85,
                  loc="lower right")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def plot_backbone_comparison(csv_path, out_path):
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "TinyLLM (10M) vs Qwen2.5-1.5B Backbone:\n"
        "D-TAIA framework for one dataset per entropy level",
        fontsize=13, fontweight="bold", y=1.02
    )

    backbone_bar(axes[0], df, "NA_Acc", "NA_CI",
                 "NA Accuracy", "NA Accuracy (%) ± CI",
                 show_legend=True)

    backbone_bar(axes[1], df, "NA_F1", "F1_CI",
                 "F1-Macro", "NA F1-Macro ± CI")

    backbone_bar(axes[2], df, "RT_MAE", "MAE_CI",
                 "RT MAE", "RT MAE (days) ± CI",
                 best_is_min=True)

    # Shared note about delta annotations
    # fig.text(0.5, -0.03,
    #          "Numbers above Qwen2.5-1.5B bars show absolute gain over TinyLLM baseline.",
    #          ha="center", fontsize=9.5, color="#555555", style="italic")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # base = "data_functions/experimental_results"

    plot_backbone_comparison(
        f"llm-vs-slm.csv",
        f"llm-vs-slm.png"
    )

