"""
Visualization script for D-TAIA experiment results.
Generates grouped bar charts with 95% CI error bars for Experiments 1–4.
Each experiment produces two figures:
  - *_perf.png  : NA Accuracy + F1-Macro side by side
  - *_rt.png    : RT MAE alone
Exp4 produces a single log-scale training time figure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
MODEL_COLOURS = {
    "D-TAIA":        "#2ca02c",   # green
    "FT+TinyLLM":  "#1f77b4",   # blue
    "LSTM":             "#ff7f0e",   # orange
    "XGBoost":          "#7f7f7f",   # grey
    "MT-RNN":           "#d62728",   # red
}
MODEL_ORDER = list(MODEL_COLOURS.keys())

FONT = {"family": "sans-serif", "size": 12}
plt.rc("font", **FONT)


# ---------------------------------------------------------------------------
# Helper: draw one grouped-bar subplot
# ---------------------------------------------------------------------------
def grouped_bar(ax, df, metric_col, ci_col, datasets, title, ylabel,
                best_is_min=False, show_legend=False, x_rotation=0, legend_loc ="lower left", y_min=None):
    n_models  = len(MODEL_ORDER)
    bar_w     = 0.16        # was 0.14
    group_gap = 0.06        # was 0.05
    group_w   = n_models * bar_w + group_gap
    x_centers = np.arange(len(datasets)) * group_w

    for i, model in enumerate(MODEL_ORDER):
        offsets = x_centers + (i - n_models / 2 + 0.5) * bar_w
        vals, cis = [], []
        for ds in datasets:
            row = df[(df["Dataset"] == ds) & (df["Model"] == model)]
            if row.empty:
                vals.append(0); cis.append(0)
            else:
                vals.append(float(row[metric_col].values[0]))
                cis.append(float(row[ci_col].values[0]))

        ax.bar(offsets, vals, width=bar_w,
               color=MODEL_COLOURS[model], label=model,
               edgecolor="white", linewidth=0.5)
        ax.errorbar(offsets, vals, yerr=cis,
                    fmt="none", color="black", capsize=4, linewidth=1.2)  # bigger caps

    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, fontsize=12,          # was 10
                       rotation=x_rotation,
                       ha="right" if x_rotation > 0 else "center")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)   # was 12
    ax.set_ylabel(ylabel, fontsize=13)                            # was 11
    ax.tick_params(axis="y", labelsize=11)                        # was 10
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # add axis break for clarity
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
        ax.spines['left'].set_visible(False)
        ax2 = ax.twinx()
        ax2.set_ylim(0, y_min)
        ax2.spines['right'].set_visible(False)
        ax2.set_yticks([])
        # Draw break marks
        kwargs = dict(transform=ax.transAxes, color='k',
                      clip_on=False, linewidth=1.5)
        ax.plot([-0.02, 0.02], [-0.02, 0.02], **kwargs)
        ax.plot([-0.02, 0.02], [-0.035, -0.015], **kwargs)

    if show_legend:
        handles = [mpatches.Patch(color=MODEL_COLOURS[m], label=m)
                   for m in MODEL_ORDER]
        ax.legend(handles=handles, fontsize=12, framealpha=0.85,
                  loc=legend_loc, ncol=1)



# ---------------------------------------------------------------------------
# EXP 1 - High Entropy
# ---------------------------------------------------------------------------
def plot_exp1(csv_path, out_path_perf, out_path_rt):
    df = pd.read_csv(csv_path)
    datasets = ["BPI2015_1", "BPI2015_2", "BPI2015_3", "BPI2015_4", "BPI2015_5"]

    # --- NA Accuracy + F1-Macro ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Exp1: High Entropy - NA Accuracy & F1-Macro\n(BPI2015 datasets)",
                 fontsize=15, fontweight="bold", y=1.02)

    grouped_bar(axes[0], df, "NA_Acc", "NA_CI", datasets,
                "NA Accuracy", "NA Accuracy (%) +/- CI", show_legend=True, y_min=20)
    grouped_bar(axes[1], df, "NA_F1", "F1_CI", datasets,
                "F1-Macro", "NA F1-Macro +/- CI", y_min=0.2)

    plt.tight_layout()
    fig.savefig(out_path_perf, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_perf}")

    # --- RT MAE (unchanged) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fig.suptitle("Exp1: High Entropy - RT MAE\n(BPI2015 datasets)",
                 fontsize=12, fontweight="bold", y=1.01)

    grouped_bar(ax, df, "RT_MAE", "MAE_CI", datasets,
                "RT MAE", "RT MAE (days) +/- CI",
                best_is_min=True, show_legend=True, legend_loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path_rt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_rt}")


# ---------------------------------------------------------------------------
# EXP 2 - Low Data
# ---------------------------------------------------------------------------
def plot_exp2(csv_path, out_path_perf, out_path_rt):
    df = pd.read_csv(csv_path)
    datasets = ["BPI2012", "BPI2020_DD", "BPI2015_2"]

    # --- NA Accuracy + F1-Macro stacked vertically ---
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    fig.suptitle("Exp2: Low-Data Robustness (832 traces) - NA Accuracy & F1-Macro",
                 fontsize=15, fontweight="bold", y=1.01)

    grouped_bar(axes[0], df, "NA_Acc", "NA_CI", datasets,
                "NA Accuracy", "NA Accuracy (%) +/- CI",
                show_legend=True, x_rotation=35)
    grouped_bar(axes[1], df, "NA_F1", "F1_CI", datasets,
                "F1-Macro", "NA F1-Macro +/- CI",
                show_legend=False, x_rotation=35)

    plt.tight_layout()
    fig.savefig(out_path_perf, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_perf}")

    # --- RT MAE (unchanged) ---
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Exp2: Low-Data Robustness (832 traces) - RT MAE",
                 fontsize=12, fontweight="bold", y=1.01)

    grouped_bar(ax, df, "RT_MAE", "MAE_CI", datasets,
                "RT MAE", "RT MAE (days) +/- CI",
                best_is_min=True, show_legend=True, x_rotation=35, legend_loc="upper left")

    plt.tight_layout()
    fig.savefig(out_path_rt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_rt}")

# ---------------------------------------------------------------------------
# EXP 3 - Low Entropy
# ---------------------------------------------------------------------------
def plot_exp3(csv_path, out_path_perf, out_path_rt):
    df = pd.read_csv(csv_path)
    datasets = ["BPI2012", "BPI2017", "BPI2020_DD",
                "BPI2020_ID", "BPI2020_PTC", "BPI2020_RFP"]

    # --- NA Accuracy + F1-Macro stacked vertically ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    fig.suptitle("Exp3: Low Entropy - NA Accuracy & F1-Macro\n(BPI2020 & BPI2012/17 datasets)",
                 fontsize=15, fontweight="bold", y=1.01)

    grouped_bar(axes[0], df, "NA_Acc", "NA_CI", datasets,
                "NA Accuracy", "NA Accuracy (%) +/- CI",
                show_legend=True, x_rotation=25, y_min=50)
    grouped_bar(axes[1], df, "NA_F1", "F1_CI", datasets,
                "F1-Macro", "NA F1-Macro +/- CI",
                show_legend=False, x_rotation=25, y_min=0.5)

    plt.tight_layout()
    fig.savefig(out_path_perf, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_perf}")

    # --- RT MAE (unchanged) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Exp3: Low Entropy - RT MAE\n(BPI2020 & BPI2012/17 datasets)",
                 fontsize=12, fontweight="bold", y=1.01)

    grouped_bar(ax, df, "RT_MAE", "MAE_CI", datasets,
                "RT MAE", "RT MAE (days) +/- CI",
                best_is_min=True, show_legend=True, x_rotation=25)

    plt.tight_layout()
    fig.savefig(out_path_rt, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path_rt}")

# ---------------------------------------------------------------------------
# EXP 4 - Computational Efficiency (log-scale, single figure)
# ---------------------------------------------------------------------------
def plot_exp4(csv_path, out_path):
    df = pd.read_csv(csv_path)
    datasets  = df["Dataset"].tolist()
    models    = ["D-TAIA", "FT+TinyLLM", "LSTM", "XGBoost", "MT-RNN"]
    col_map   = {
        "D-TAIA": ("D-TAIA",  "DTAIA_std"),
        "FT+TinyLLM":   ("FT+TinyLLM",    "FT_std"),
        "LSTM":      ("LSTM",       "LSTM_std"),
        "XGBoost":   ("XGBoost",    "XGBoost_std"),
        "MT-RNN":    ("MT-RNN",     "MTRNN_std"),
    }
    colour_map = {
        "D-TAIA": MODEL_COLOURS["D-TAIA"],
        "FT+TinyLLM":   MODEL_COLOURS["FT+TinyLLM"],
        "LSTM":      MODEL_COLOURS["LSTM"],
        "XGBoost":   MODEL_COLOURS["XGBoost"],
        "MT-RNN":    MODEL_COLOURS["MT-RNN"],
    }

    n_models  = len(models)
    bar_w     = 0.14
    group_gap = 0.05
    group_w   = n_models * bar_w + group_gap
    x_centers = np.arange(len(datasets)) * group_w

    fig, ax = plt.subplots(figsize=(16, 5))

    for i, model in enumerate(models):
        val_col, std_col = col_map[model]
        offsets = x_centers + (i - n_models / 2 + 0.5) * bar_w
        vals = df[val_col].values
        stds = df[std_col].values

        ax.bar(offsets, vals, width=bar_w,
               color=colour_map[model], label=model,
               edgecolor="white", linewidth=0.4)
        ax.errorbar(offsets, vals, yerr=stds,
                    fmt="none", color="black", capsize=2.5, linewidth=0.9)

    ax.set_yscale("log")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(datasets, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Training Time (hours, log scale)", fontsize=11)
    ax.set_title("Exp4: Computational Efficiency - Training Time per Dataset",
                 fontsize=12, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=10)

    handles = [mpatches.Patch(color=colour_map[m], label=m) for m in models]
    ax.legend(handles=handles, fontsize=9, framealpha=0.85,
              loc="lower left", ncol=1)

    # ax.annotate("MT-RNN fastest;\nLLM models ~22-25x slower",
    #             xy=(x_centers[-3] + 0.3, 0.018), fontsize=9,
    #             color="#555555", style="italic")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    base = "data_functions/experimental_results"
    #
    # plot_exp1(f"exp1_high_entropy.csv",
    #           f"exp1_perf.png",
    #           f"exp1_rt.png")

    # plot_exp2(f"exp2_low_data.csv",
    #           f"exp2_perf.png",
    #           f"exp2_rt.png")

    # plot_exp3(f"exp3_low_entropy.csv",
    #           f"exp3_perf.png",
    #           f"exp3_rt.png")
    # #
    plot_exp4(f"exp4_computational_time.csv",
              f"exp4_computational_time.png")

    print("\nAll plots saved to", base)