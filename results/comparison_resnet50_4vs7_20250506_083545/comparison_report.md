# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** resnet50
- **Classification Task:** Class 4 vs Class 7
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9745 | 0.9675 | -0.0070 |
| Precision | 0.9712 | 0.9825 | 0.0113 |
| Recall | 0.9780 | 0.9520 | -0.0260 |
| F1 | 0.9746 | 0.9670 | -0.0076 |
| ROC AUC | 0.9976 | 0.9940 | -0.0037 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
