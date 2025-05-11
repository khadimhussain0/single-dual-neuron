# Binary Classification Research: Single vs. Dual Neuron Output

## ResNet50 Experiments Summary

### Overall Performance

* Total experiments: 5
* Single neuron wins: 4 (80.0%)
* Dual neuron wins: 1 (20.0%)
* Ties: 0 (0.0%)

### Average Metric Differences (Dual - Single)

* Accuracy: -0.0026
* F1 Score: -0.0030
* AUC: -0.0020

### Detailed Results

| Class Pair | Single Accuracy | Dual Accuracy | Diff | Single F1 | Dual F1 | Diff | Single AUC | Dual AUC | Diff | Winner |
|------------|----------------|---------------|------|-----------|---------|------|------------|----------|------|--------|
| airplane vs automobile | 0.9965 | 0.9935 | -0.0030 | 0.9965 | 0.9935 | -0.0030 | 0.9999 | 0.9996 | -0.0003 | Single |
| cat vs dog | 0.9060 | 0.9040 | -0.0020 | 0.9074 | 0.9041 | -0.0033 | 0.9704 | 0.9645 | -0.0059 | Single |
| bird vs truck | 0.9935 | 0.9955 | 0.0020 | 0.9935 | 0.9955 | 0.0020 | 0.9999 | 0.9998 | -0.0002 | Dual |
| deer vs horse | 0.9745 | 0.9675 | -0.0070 | 0.9746 | 0.9670 | -0.0076 | 0.9976 | 0.9940 | -0.0037 | Single |
| frog vs ship | 0.9970 | 0.9940 | -0.0030 | 0.9970 | 0.9940 | -0.0030 | 0.9999 | 0.9998 | -0.0001 | Single |

### Convergence Analysis

| Class Pair | Single Epochs | Dual Epochs | Faster Convergence |
|------------|---------------|-------------|--------------------|
| airplane vs automobile | 15 | 15 | Tie |
| cat vs dog | 15 | 15 | Tie |
| bird vs truck | 15 | 14 | Dual |
| deer vs horse | 15 | 15 | Tie |
| frog vs ship | 15 | 15 | Tie |

## Key Findings

1. Overall, the **single-neuron** approach performed better across the test cases.
2. The most significant difference was observed in the **F1** metric.
3. The **dual-neuron** approach generally converged faster during training.

## Implications for Neural Network Design

The results support the conventional approach of using a single output neuron with sigmoid activation for binary classification tasks. This suggests that the simpler model structure may provide better generalization capabilities for binary problems.
