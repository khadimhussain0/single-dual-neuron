# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** resnet50
- **Classification Task:** Class 6 vs Class 8
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9970 | 0.9940 | -0.0030 |
| Precision | 0.9960 | 0.9970 | 0.0010 |
| Recall | 0.9980 | 0.9910 | -0.0070 |
| F1 | 0.9970 | 0.9940 | -0.0030 |
| ROC AUC | 0.9999 | 0.9998 | -0.0001 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
