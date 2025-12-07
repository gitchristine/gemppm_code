"""
Simple Plotting Script for XGBoost Predictions
===============================================
Creates publication-ready plots for comparing XGBoost with other methods.

Usage:
    python plot_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_activity_accuracy_comparison(predictions_dict: dict, output_path: str = 'activity_accuracy_comparison.png'):
    """
    Bar chart comparing activity prediction accuracy across methods.

    Parameters:
    -----------
    predictions_dict : dict
        {'Method Name': 'path/to/predictions.csv', ...}
        Example: {'XGBoost': 'outputs/predictions_bpi2012.csv',
                  'LLM': 'llm_outputs/predictions_bpi2012.csv'}
    """

    accuracies = {}

    for method, csv_path in predictions_dict.items():
        df = pd.read_csv(csv_path)
        accuracy = df['activity_correct'].mean() * 100  # Convert to percentage
        accuracies[method] = accuracy
        print(f"{method}: {accuracy:.2f}% accuracy")

    # Create bar plot
    plt.figure(figsize=(8, 6))
    methods = list(accuracies.keys())
    values = list(accuracies.values())

    bars = plt.bar(methods, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    plt.xlabel('Method', fontsize=13, fontweight='bold')
    plt.title('Next Activity Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {output_path}")
    plt.close()


def plot_time_mae_comparison(predictions_dict: dict, output_path: str = 'time_mae_comparison.png'):
    """
    Bar chart comparing time prediction MAE across methods.

    Parameters:
    -----------
    predictions_dict : dict
        {'Method Name': 'path/to/predictions.csv', ...}
    """

    maes = {}

    for method, csv_path in predictions_dict.items():
        df = pd.read_csv(csv_path)
        mae = df['time_error'].mean()
        maes[method] = mae
        print(f"{method}: MAE = {mae:.2f} days")

    # Create bar plot
    plt.figure(figsize=(8, 6))
    methods = list(maes.keys())
    values = list(maes.values())

    bars = plt.bar(methods, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('Mean Absolute Error (days)', fontsize=13, fontweight='bold')
    plt.xlabel('Method', fontsize=13, fontweight='bold')
    plt.title('Remaining Time Prediction Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    plt.close()


def plot_time_error_distribution(csv_path: str, method_name: str = 'XGBoost',
                                 output_path: str = 'time_error_distribution.png'):
    """
    Histogram of time prediction errors.
    """

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    plt.hist(df['time_error'], bins=50, edgecolor='black', alpha=0.7, color='#2E86AB')
    plt.xlabel('Time Prediction Error (days)', fontsize=13, fontweight='bold')
    plt.ylabel('Frequency', fontsize=13, fontweight='bold')
    plt.title(f'{method_name} - Time Error Distribution', fontsize=14, fontweight='bold')

    # Add statistics
    mean_error = df['time_error'].mean()
    median_error = df['time_error'].median()
    plt.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
    plt.axvline(median_error, color='green', linestyle='--', linewidth=2, label=f'Median: {median_error:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    plt.close()


def plot_confusion_matrix(csv_path: str, top_n: int = 10,
                          output_path: str = 'confusion_matrix.png'):
    """
    Confusion matrix for top N most common activities.
    """

    from sklearn.metrics import confusion_matrix

    df = pd.read_csv(csv_path)

    # Get top N activities
    top_activities = df['true_activity'].value_counts().head(top_n).index
    subset = df[df['true_activity'].isin(top_activities)]

    # Create confusion matrix
    cm = confusion_matrix(subset['true_activity'], subset['pred_activity'],
                          labels=top_activities)

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=top_activities, yticklabels=top_activities,
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted Activity', fontsize=13, fontweight='bold')
    plt.ylabel('True Activity', fontsize=13, fontweight='bold')
    plt.title(f'Confusion Matrix - Top {top_n} Activities', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    plt.close()


def plot_accuracy_by_position(csv_path: str, method_name: str = 'XGBoost',
                              output_path: str = 'accuracy_by_position.png'):
    """
    Accuracy throughout the case lifecycle (early, mid, late).

    Note: Requires 'case_progress_pct' in predictions_[dataset]_full.csv
    """

    df = pd.read_csv(csv_path)

    if 'case_progress_pct' not in df.columns:
        print("⚠️  Warning: 'case_progress_pct' not found. Skipping this plot.")
        print("    Use predictions_[dataset]_full.csv instead of predictions_[dataset].csv")
        return

    # Bin by case progress
    df['position'] = pd.cut(df['case_progress_pct'],
                            bins=[0, 0.25, 0.5, 0.75, 1.0],
                            labels=['Early\n(0-25%)', 'Mid-Early\n(25-50%)',
                                    'Mid-Late\n(50-75%)', 'Late\n(75-100%)'])

    accuracy_by_position = df.groupby('position')['activity_correct'].mean() * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(accuracy_by_position)), accuracy_by_position.values,
                   color='#2E86AB', edgecolor='black')

    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.xticks(range(len(accuracy_by_position)), accuracy_by_position.index)
    plt.ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    plt.xlabel('Case Position', fontsize=13, fontweight='bold')
    plt.title(f'{method_name} - Accuracy Throughout Case Lifecycle', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_path}")
    plt.close()


def create_results_table(predictions_dict: dict, output_path: str = 'results_table.csv'):
    """
    Create a summary table with all metrics for each method.
    Perfect for thesis tables!

    Parameters:
    -----------
    predictions_dict : dict
        {'Method Name': 'path/to/predictions.csv', ...}
    """

    results = []

    for method, csv_path in predictions_dict.items():
        df = pd.read_csv(csv_path)

        # Calculate metrics
        accuracy = df['activity_correct'].mean() * 100
        mae = df['time_error'].mean()
        median_ae = df['time_error'].median()
        rmse = np.sqrt((df['time_error'] ** 2).mean())

        results.append({
            'Method': method,
            'Activity_Accuracy_%': f'{accuracy:.2f}',
            'Time_MAE_days': f'{mae:.2f}',
            'Time_Median_AE_days': f'{median_ae:.2f}',
            'Time_RMSE_days': f'{rmse:.2f}'
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY TABLE")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    print(f"\n✓ Saved to: {output_path}\n")

    return results_df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("XGBOOST RESULTS PLOTTING")
    print("=" * 80 + "\n")

    # ========================================================================
    # CONFIGURE YOUR PREDICTIONS HERE
    # ========================================================================

    # Example 1: Plot XGBoost results only
    xgboost_predictions = 'outputs/predictions_bpi2012.csv'

    # Example 2: Compare multiple methods
    all_methods = {
        'XGBoost': '201217_predictions.csv',
        # 'LSTM': 'lstm_outputs/predictions_bpi2012.csv',
        # 'GPT-2': 'gpt2_outputs/predictions_bpi2012.csv',
        # 'LLaMA': 'llama_outputs/predictions_bpi2012.csv',
    }

    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================

    # Uncomment the plots you want to generate:

    # # 1. Summary table (always useful!)
    print("Creating results summary table...")
    create_results_table(all_methods, 'outputs/results_comparison.csv')

    # # 2. Activity accuracy comparison
    print("\nPlotting activity accuracy comparison...")
    plot_activity_accuracy_comparison(all_methods, 'outputs/activity_accuracy.png')

    # # 3. Time MAE comparison
    print("\nPlotting time MAE comparison...")
    plot_time_mae_comparison(all_methods, 'outputs/time_mae.png')

    # # 4. Time error distribution (single method)
    print("\nPlotting time error distribution...")
    plot_time_error_distribution(xgboost_predictions, 'XGBoost', 'outputs/time_error_dist.png')

    # # 5. Confusion matrix
    # print("\nPlotting confusion matrix...")
    # plot_confusion_matrix(xgboost_predictions, top_n=10, 'outputs/confusion_matrix.png')

    # # 6. Accuracy by case position (requires _full.csv)
    # print("\nPlotting accuracy by position...")
    # plot_accuracy_by_position('outputs/predictions_bpi2012_full.csv', 'XGBoost',
    #                           'outputs/accuracy_by_position.png')

    print("\n" + "=" * 80)
    print("INSTRUCTIONS")
    print("=" * 80)
    print("\n1. Update the 'all_methods' dictionary with your prediction file paths")
    print("2. Uncomment the plots you want to generate")
    print("3. Run: python plot_results.py")
    print("\nOutputs will be saved to outputs/ as high-resolution PNG files (300 DPI)")
    print("Perfect for including in your thesis!\n")
    print("=" * 80 + "\n")