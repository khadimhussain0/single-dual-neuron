# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** vit
- **Classification Task:** Class 3 vs Class 5
- **Batch Size:** 16
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9425 | 0.9465 | 0.0040 |
| Precision | 0.9403 | 0.9598 | 0.0195 |
| Recall | 0.9450 | 0.9320 | -0.0130 |
| F1 | 0.9426 | 0.9457 | 0.0031 |
| ROC AUC | 0.9871 | 0.9868 | -0.0004 |

## Key Findings

- The **dual neuron** approach performed better overall for this experiment.
- Largest difference observed in **precision** metric.
- **Single** neuron converged faster (13 vs 15 epochs).
