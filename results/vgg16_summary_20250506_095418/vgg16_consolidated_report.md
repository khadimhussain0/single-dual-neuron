# VGG16 Binary Classification Research: Single vs. Dual Neuron Output

## VGG16 Experiments Summary

### Overall Performance

* Total experiments: 2
* Single neuron wins: 2 (100.0%)
* Dual neuron wins: 0 (0.0%)
* Ties: 0 (0.0%)

### Average Metric Differences (Dual - Single)

* Accuracy: -0.0245
* F1 Score: -0.0280
* AUC: -0.0197

### Detailed Results

| Class Pair | Single Accuracy | Dual Accuracy | Diff | Single F1 | Dual F1 | Diff | Single AUC | Dual AUC | Diff | Winner |
|------------|----------------|---------------|------|-----------|---------|------|------------|----------|------|--------|
| airplane vs automobile | 0.9885 | 0.9755 | -0.0130 | 0.9886 | 0.9754 | -0.0132 | 0.9997 | 0.9954 | -0.0043 | Single |
| cat vs dog | 0.8880 | 0.8520 | -0.0360 | 0.8929 | 0.8501 | -0.0429 | 0.9658 | 0.9306 | -0.0352 | Single |

### Convergence Analysis

| Class Pair | Single Epochs | Dual Epochs | Faster Convergence |
|------------|---------------|-------------|--------------------|
| airplane vs automobile | 15 | 15 | Tie |
| cat vs dog | 15 | 15 | Tie |

## VGG16-Specific Findings

1. For VGG16, the **single-neuron** approach performed better overall.
2. In terms of convergence speed, the **neither** approach was generally faster when using VGG16.
3. Compared to smaller CNN architectures, VGG16 shows **reduced performance gaps** between the two approaches, reflecting the pattern seen with ResNet50 where larger models diminish the impact of output layer choice.
