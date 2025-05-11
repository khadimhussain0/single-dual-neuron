# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** resnet50
- **Classification Task:** Class 0 vs Class 1
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9965 | 0.9935 | -0.0030 |
| Precision | 0.9990 | 0.9930 | -0.0060 |
| Recall | 0.9940 | 0.9940 | 0.0000 |
| F1 | 0.9965 | 0.9935 | -0.0030 |
| ROC AUC | 0.9999 | 0.9996 | -0.0003 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **precision** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
