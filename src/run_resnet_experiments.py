"""
ResNet50 experiments for binary classification research.
This script runs comprehensive experiments using ResNet50 as the backbone architecture
to compare single-neuron vs. dual-neuron output layers for binary classification.
"""

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

from train import run_experiment

# Define CIFAR-10 class names for reference
CIFAR10_CLASSES = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def run_all_experiments(epochs=30, batch_size=32, test_pairs=None):
    """
    Run all ResNet50 experiments for the research paper.
    
    Args:
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        test_pairs: List of class pairs to test. If None, uses default pairs.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Default test pairs if none provided
    if test_pairs is None:
        test_pairs = [
            (0, 1),  # airplane vs. automobile
            (3, 5),  # cat vs. dog
            (2, 9),  # bird vs. truck
            (4, 7),  # deer vs. horse
            (6, 8)   # frog vs. ship
        ]
    
    # ResNet50 requires images of at least 224x224
    img_size = (224, 224, 3)
    
    # For each pair, run the experiment and collect results
    all_results = []
    for class_a, class_b in test_pairs:
        print(f"\n{'='*80}\nRunning ResNet50 experiment: {CIFAR10_CLASSES[class_a]} vs {CIFAR10_CLASSES[class_b]}\n{'='*80}\n")
        
        result = run_experiment(
            model_name='resnet50',
            dataset=(class_a, class_b),
            batch_size=batch_size,
            epochs=epochs,
            img_size=img_size
        )
        
        # Extract key metrics for the summary
        single_metrics = result['single_neuron']['test_metrics']
        dual_metrics = result['dual_neuron']['test_metrics']
        
        summary = {
            'Class_A': CIFAR10_CLASSES[class_a],
            'Class_B': CIFAR10_CLASSES[class_b],
            'Single_Accuracy': single_metrics['accuracy'],
            'Dual_Accuracy': dual_metrics['accuracy'],
            'Accuracy_Diff': dual_metrics['accuracy'] - single_metrics['accuracy'],
            'Single_F1': single_metrics['f1'],
            'Dual_F1': dual_metrics['f1'],
            'F1_Diff': dual_metrics['f1'] - single_metrics['f1'],
            'Single_AUC': result['single_neuron']['roc']['auc'],
            'Dual_AUC': result['dual_neuron']['roc']['auc'],
            'AUC_Diff': result['dual_neuron']['roc']['auc'] - result['single_neuron']['roc']['auc'],
            'Single_Convergence_Epochs': len(result['single_neuron']['history']['train_loss']),
            'Dual_Convergence_Epochs': len(result['dual_neuron']['history']['train_loss']),
            'Report_Path': result['comparison_path']
        }
        
        all_results.append(summary)
    
    # Create consolidated results report
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/resnet50_summary_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    results_df.to_csv(os.path.join(results_dir, 'resnet50_experiments_summary.csv'), index=False)
    
    # Create a consolidated markdown report
    create_consolidated_report(results_df, results_dir)
    
    return results_df

def create_consolidated_report(results_df, output_dir):
    """
    Create a consolidated markdown report of all experiments.
    
    Args:
        results_df: DataFrame with all experiment results
        output_dir: Directory to save the report
    """
    report_path = os.path.join(output_dir, 'consolidated_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Binary Classification Research: Single vs. Dual Neuron Output\n\n")
        f.write("## ResNet50 Experiments Summary\n\n")
        
        # Overall statistics
        single_wins = sum(results_df['Accuracy_Diff'] < 0)
        dual_wins = sum(results_df['Accuracy_Diff'] > 0)
        ties = sum(results_df['Accuracy_Diff'] == 0)
        
        f.write("### Overall Performance\n\n")
        f.write(f"* Total experiments: {len(results_df)}\n")
        f.write(f"* Single neuron wins: {single_wins} ({single_wins/len(results_df)*100:.1f}%)\n")
        f.write(f"* Dual neuron wins: {dual_wins} ({dual_wins/len(results_df)*100:.1f}%)\n")
        f.write(f"* Ties: {ties} ({ties/len(results_df)*100:.1f}%)\n\n")
        
        # Average differences
        avg_acc_diff = results_df['Accuracy_Diff'].mean()
        avg_f1_diff = results_df['F1_Diff'].mean()
        avg_auc_diff = results_df['AUC_Diff'].mean()
        
        f.write("### Average Metric Differences (Dual - Single)\n\n")
        f.write(f"* Accuracy: {avg_acc_diff:.4f}\n")
        f.write(f"* F1 Score: {avg_f1_diff:.4f}\n")
        f.write(f"* AUC: {avg_auc_diff:.4f}\n\n")
        
        # Main performance table
        f.write("### Detailed Results\n\n")
        f.write("| Class Pair | Single Accuracy | Dual Accuracy | Diff | Single F1 | Dual F1 | Diff | Single AUC | Dual AUC | Diff | Winner |\n")
        f.write("|------------|----------------|---------------|------|-----------|---------|------|------------|----------|------|--------|\n")
        
        for _, row in results_df.iterrows():
            class_pair = f"{row['Class_A']} vs {row['Class_B']}"
            
            # Determine overall winner based on majority of metrics
            metrics_diff = [row['Accuracy_Diff'], row['F1_Diff'], row['AUC_Diff']]
            positive_diffs = sum(diff > 0 for diff in metrics_diff)
            negative_diffs = sum(diff < 0 for diff in metrics_diff)
            
            if positive_diffs > negative_diffs:
                winner = "Dual"
            elif negative_diffs > positive_diffs:
                winner = "Single"
            else:
                winner = "Tie"
            
            f.write(f"| {class_pair} | {row['Single_Accuracy']:.4f} | {row['Dual_Accuracy']:.4f} | {row['Accuracy_Diff']:.4f} | ")
            f.write(f"{row['Single_F1']:.4f} | {row['Dual_F1']:.4f} | {row['F1_Diff']:.4f} | ")
            f.write(f"{row['Single_AUC']:.4f} | {row['Dual_AUC']:.4f} | {row['AUC_Diff']:.4f} | {winner} |\n")
        
        # Convergence analysis
        f.write("\n### Convergence Analysis\n\n")
        f.write("| Class Pair | Single Epochs | Dual Epochs | Faster Convergence |\n")
        f.write("|------------|---------------|-------------|--------------------|\n")
        
        for _, row in results_df.iterrows():
            class_pair = f"{row['Class_A']} vs {row['Class_B']}"
            single_epochs = row['Single_Convergence_Epochs']
            dual_epochs = row['Dual_Convergence_Epochs']
            
            if single_epochs < dual_epochs:
                faster = "Single"
            elif dual_epochs < single_epochs:
                faster = "Dual"
            else:
                faster = "Tie"
                
            f.write(f"| {class_pair} | {single_epochs} | {dual_epochs} | {faster} |\n")
        
        # Conclusions
        f.write("\n## Key Findings\n\n")
        
        # Overall better approach based on average differences
        if avg_acc_diff > 0 and avg_f1_diff > 0 and avg_auc_diff > 0:
            better_approach = "dual-neuron"
        elif avg_acc_diff < 0 and avg_f1_diff < 0 and avg_auc_diff < 0:
            better_approach = "single-neuron"
        else:
            better_approach = "mixed, with no clear winner"
            
        f.write(f"1. Overall, the **{better_approach}** approach performed better across the test cases.\n")
        
        # Most significant difference
        metrics = ['Accuracy', 'F1', 'AUC']
        avg_diffs = [abs(avg_acc_diff), abs(avg_f1_diff), abs(avg_auc_diff)]
        most_significant = metrics[avg_diffs.index(max(avg_diffs))]
        
        f.write(f"2. The most significant difference was observed in the **{most_significant}** metric.\n")
        
        # Convergence pattern
        single_faster = sum(results_df['Single_Convergence_Epochs'] < results_df['Dual_Convergence_Epochs'])
        dual_faster = sum(results_df['Single_Convergence_Epochs'] > results_df['Dual_Convergence_Epochs'])
        
        if single_faster > dual_faster:
            convergence_winner = "single-neuron"
        elif dual_faster > single_faster:
            convergence_winner = "dual-neuron"
        else:
            convergence_winner = "neither"
            
        f.write(f"3. The **{convergence_winner}** approach generally converged faster during training.\n")
        
        f.write("\n## Implications for Neural Network Design\n\n")
        
        if better_approach == "single-neuron":
            f.write("The results support the conventional approach of using a single output neuron with sigmoid activation for binary classification tasks. This suggests that the simpler model structure may provide better generalization capabilities for binary problems.\n")
        elif better_approach == "dual-neuron":
            f.write("The results challenge the conventional wisdom, showing that using two output neurons with softmax activation can outperform the traditional single-neuron approach. This suggests that explicitly modeling both classes can provide better discrimination capabilities in some binary classification scenarios.\n")
        else:
            f.write("The mixed results suggest that the choice between single-neuron and dual-neuron approaches may be problem-dependent. Factors such as class separation, feature distribution, and model capacity may influence which approach performs better for a specific binary classification task.\n")
    
    print(f"Consolidated report created: {report_path}")
    return report_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ResNet50 experiments for binary classification research')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--all_pairs', action='store_true', help='Run experiments on all possible class pairs')
    args = parser.parse_args()
    
    if args.all_pairs:
        # Generate all possible pairs of CIFAR-10 classes
        all_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]
        print(f"Running experiments on all {len(all_pairs)} class pairs. This will take a while...")
        run_all_experiments(epochs=args.epochs, batch_size=args.batch_size, test_pairs=all_pairs)
    else:
        # Run on a subset of pairs
        run_all_experiments(epochs=args.epochs, batch_size=args.batch_size)
