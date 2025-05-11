"""
Analysis script for comparing single-neuron and two-neuron binary classification models.
This script analyzes experiment results and produces comparative visualizations.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_experiment_results(results_dir="results"):
    """
    Load results from all experiments.
    
    Args:
        results_dir: Directory containing experiment results
        
    Returns:
        DataFrame with consolidated results
    """
    # Find all results CSV files
    result_files = glob.glob(os.path.join(results_dir, "**/results_summary.csv"), recursive=True)
    
    if not result_files:
        raise ValueError(f"No result files found in {results_dir}")
    
    # Load and concatenate all results
    dfs = []
    for file_path in result_files:
        df = pd.read_csv(file_path)
        # Add experiment name based on directory
        experiment_dir = os.path.basename(os.path.dirname(file_path))
        df['Experiment'] = experiment_dir
        dfs.append(df)
    
    # Combine all results
    all_results = pd.concat(dfs, ignore_index=True)
    
    return all_results

def compare_metrics(results_df):
    """
    Compare performance metrics between single-neuron and two-neuron models.
    
    Args:
        results_df: DataFrame with experiment results
        
    Returns:
        DataFrame with statistical comparison
    """
    # Group by model and classes
    grouped = results_df.groupby(['Model', 'Classes'])
    
    # Create empty list for comparison results
    comparisons = []
    
    # Metrics to compare
    metrics = ['Test_Accuracy', 'Test_Loss', 'Test_AUC', 'F1_Score', 'ROC_AUC']
    
    # For each model + dataset combination
    for (model, classes), group in grouped:
        # Ensure we have both single and dual neuron results
        if len(group) < 2:
            continue
            
        # Get results for each configuration
        single_neuron = group[group['Output_Neurons'] == 1]
        dual_neuron = group[group['Output_Neurons'] == 2]
        
        if len(single_neuron) == 0 or len(dual_neuron) == 0:
            continue
        
        # Compare metrics
        comparison = {
            'Model': model,
            'Classes': classes
        }
        
        for metric in metrics:
            # Calculate difference
            single_value = single_neuron[metric].values[0]
            dual_value = dual_neuron[metric].values[0]
            diff = dual_value - single_value
            
            # Add to comparison dictionary
            comparison[f'{metric}_Single'] = single_value
            comparison[f'{metric}_Dual'] = dual_value
            comparison[f'{metric}_Diff'] = diff
            
            # Determine which is better (higher is better except for Loss)
            if metric == 'Test_Loss':
                comparison[f'{metric}_Better'] = 'Dual' if diff < 0 else 'Single'
            else:
                comparison[f'{metric}_Better'] = 'Dual' if diff > 0 else 'Single'
        
        comparisons.append(comparison)
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparisons)
    
    return comparison_df

def plot_metric_comparison(comparison_df, output_dir="results"):
    """
    Create visualizations comparing metrics between single and dual neuron models.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to plot
    metrics = ['Test_Accuracy', 'Test_Loss', 'Test_AUC', 'F1_Score', 'ROC_AUC']
    
    # 1. Bar charts for each metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        models = comparison_df['Model'].unique()
        x = np.arange(len(models))
        width = 0.35
        
        # Calculate average for each model across datasets
        single_means = []
        dual_means = []
        
        for model in models:
            model_data = comparison_df[comparison_df['Model'] == model]
            single_means.append(model_data[f'{metric}_Single'].mean())
            dual_means.append(model_data[f'{metric}_Dual'].mean())
        
        # Plot bars
        plt.bar(x - width/2, single_means, width, label='Single Neuron')
        plt.bar(x + width/2, dual_means, width, label='Dual Neuron')
        
        # Add labels and title
        plt.xlabel('Model')
        plt.ylabel(metric.replace('_', ' '))
        plt.title(f'Comparison of {metric.replace("_", " ")} Between Single and Dual Neuron Models')
        plt.xticks(x, models)
        plt.legend()
        
        # Add value labels on bars
        for i, v in enumerate(single_means):
            plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
        
        for i, v in enumerate(dual_means):
            plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # 2. Create a heatmap of performance differences
    # Reshape data for heatmap
    heatmap_data = []
    
    for _, row in comparison_df.iterrows():
        model = row['Model']
        classes = row['Classes']
        
        for metric in metrics:
            if metric != 'Test_Loss':  # Skip loss for this visualization
                diff = row[f'{metric}_Diff']
                heatmap_data.append({
                    'Model': f"{model} ({classes})",
                    'Metric': metric.replace('_', ' '),
                    'Difference': diff
                })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # Create pivot table
    pivot_data = heatmap_df.pivot(index='Model', columns='Metric', values='Difference')
    
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, cmap='RdBu_r', center=0, annot=True, fmt='.3f', cbar_kws={'label': 'Dual - Single'})
    plt.title('Performance Difference: Dual Neuron - Single Neuron')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_difference_heatmap.png'))
    plt.close()
    
    # 3. Create convergence rate comparison
    # This requires history CSV files, so let's load those
    history_files = glob.glob(os.path.join(output_dir, "**/training_history.csv"), recursive=True)
    
    if history_files:
        plt.figure(figsize=(15, 10))
        
        for file_path in history_files:
            # Extract model info from directory name
            dir_name = os.path.basename(os.path.dirname(file_path))
            parts = dir_name.split('_')
            
            if len(parts) >= 2:
                model_name = parts[0]
                neurons = parts[1].replace('neuron', '')
                
                # Load history
                history = pd.read_csv(file_path)
                
                # Plot validation accuracy
                if 'val_accuracy' in history.columns:
                    plt.plot(history['val_accuracy'], label=f"{model_name} ({neurons})")
        
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Convergence Rate Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'convergence_comparison.png'))
        plt.close()
    
    # 4. Win rate summary
    win_counts = {'Single': 0, 'Dual': 0, 'Tie': 0}
    
    for _, row in comparison_df.iterrows():
        for metric in metrics:
            if metric != 'Test_Loss':  # Higher is better
                if row[f'{metric}_Diff'] > 0.001:  # Threshold for meaningful difference
                    win_counts['Dual'] += 1
                elif row[f'{metric}_Diff'] < -0.001:
                    win_counts['Single'] += 1
                else:
                    win_counts['Tie'] += 1
            else:  # Lower is better for loss
                if row[f'{metric}_Diff'] < -0.001:
                    win_counts['Dual'] += 1
                elif row[f'{metric}_Diff'] > 0.001:
                    win_counts['Single'] += 1
                else:
                    win_counts['Tie'] += 1
    
    # Plot win rate
    plt.figure(figsize=(8, 6))
    plt.bar(win_counts.keys(), win_counts.values())
    plt.title('Win Count Across All Metrics and Experiments')
    plt.ylabel('Number of Wins')
    
    # Add value labels
    for i, (k, v) in enumerate(win_counts.items()):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'win_count_summary.png'))
    plt.close()
    
    return True

def generate_summary_report(comparison_df, output_dir="results"):
    """
    Generate a text summary report of the findings.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to include in report
    metrics = ['Test_Accuracy', 'Test_Loss', 'Test_AUC', 'F1_Score', 'ROC_AUC']
    
    # Start building the report
    report = ["# Binary Classification: Single vs. Dual Neuron Output Layer\n\n"]
    report.append("## Summary of Findings\n\n")
    
    # Calculate overall averages
    avg_diff = {}
    for metric in metrics:
        avg_diff[metric] = comparison_df[f'{metric}_Diff'].mean()
    
    # Write overall summary
    report.append("### Overall Performance Difference\n\n")
    report.append("| Metric | Average Difference (Dual - Single) | Better Approach |\n")
    report.append("| ------ | ---------------------------------- | -------------- |\n")
    
    for metric in metrics:
        diff = avg_diff[metric]
        if metric == 'Test_Loss':
            better = "Dual" if diff < 0 else "Single"
        else:
            better = "Dual" if diff > 0 else "Single"
        
        report.append(f"| {metric.replace('_', ' ')} | {diff:.4f} | {better} |\n")
    
    report.append("\n")
    
    # Model-specific analysis
    report.append("### Model-Specific Analysis\n\n")
    
    for model in comparison_df['Model'].unique():
        report.append(f"#### {model}\n\n")
        
        model_data = comparison_df[comparison_df['Model'] == model]
        
        report.append("| Dataset | Metric | Single Neuron | Dual Neuron | Difference | Better |\n")
        report.append("| ------- | ------ | ------------- | ----------- | ---------- | ------ |\n")
        
        for _, row in model_data.iterrows():
            classes = row['Classes']
            
            for metric in metrics:
                single_val = row[f'{metric}_Single']
                dual_val = row[f'{metric}_Dual']
                diff = row[f'{metric}_Diff']
                better = row[f'{metric}_Better']
                
                report.append(f"| {classes} | {metric.replace('_', ' ')} | {single_val:.4f} | {dual_val:.4f} | {diff:.4f} | {better} |\n")
        
        report.append("\n")
    
    # Key observations
    report.append("## Key Observations\n\n")
    
    # Count wins for each approach
    win_counts = {'Single': 0, 'Dual': 0, 'Tie': 0}
    
    for _, row in comparison_df.iterrows():
        for metric in metrics:
            if metric != 'Test_Loss':  # Higher is better
                if row[f'{metric}_Diff'] > 0.001:  # Threshold for meaningful difference
                    win_counts['Dual'] += 1
                elif row[f'{metric}_Diff'] < -0.001:
                    win_counts['Single'] += 1
                else:
                    win_counts['Tie'] += 1
            else:  # Lower is better for loss
                if row[f'{metric}_Diff'] < -0.001:
                    win_counts['Dual'] += 1
                elif row[f'{metric}_Diff'] > 0.001:
                    win_counts['Single'] += 1
                else:
                    win_counts['Tie'] += 1
    
    total_comparisons = sum(win_counts.values())
    
    report.append(f"1. Across all {total_comparisons} metric comparisons:\n")
    report.append(f"   - Single-neuron models performed better in {win_counts['Single']} cases ({win_counts['Single']/total_comparisons*100:.1f}%)\n")
    report.append(f"   - Dual-neuron models performed better in {win_counts['Dual']} cases ({win_counts['Dual']/total_comparisons*100:.1f}%)\n")
    report.append(f"   - Performance was tied in {win_counts['Tie']} cases ({win_counts['Tie']/total_comparisons*100:.1f}%)\n\n")
    
    # Add more detailed observations
    report.append("2. Notable observations:\n")
    
    # Check if dual neurons consistently better for any specific metric
    for metric in metrics:
        better_count = 0
        total_count = 0
        
        for _, row in comparison_df.iterrows():
            if metric == 'Test_Loss':
                if row[f'{metric}_Diff'] < 0:
                    better_count += 1
            else:
                if row[f'{metric}_Diff'] > 0:
                    better_count += 1
            
            total_count += 1
        
        if better_count > total_count * 0.7:  # If dual is better more than 70% of the time
            report.append(f"   - Dual-neuron models consistently performed better for {metric.replace('_', ' ')} ({better_count}/{total_count} cases)\n")
        elif better_count < total_count * 0.3:  # If single is better more than 70% of the time
            report.append(f"   - Single-neuron models consistently performed better for {metric.replace('_', ' ')} ({total_count-better_count}/{total_count} cases)\n")
    
    # Save report to file
    with open(os.path.join(output_dir, 'summary_report.md'), 'w') as f:
        f.write(''.join(report))
    
    return ''.join(report)

if __name__ == "__main__":
    # Load results
    try:
        results_df = load_experiment_results()
        
        # Compare metrics
        comparison_df = compare_metrics(results_df)
        
        # Plot comparisons
        plot_metric_comparison(comparison_df)
        
        # Generate summary report
        report = generate_summary_report(comparison_df)
        
        print("Analysis completed successfully.")
        print("Summary report preview:")
        print(report[:500] + "...")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
