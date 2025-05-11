# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** small_cnn
- **Classification Task:** Class 0 vs Class 1
- **Batch Size:** 64
- **Max Epochs:** 20
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9835 | 0.9735 | -0.0100 |
| Precision | 0.9811 | 0.9740 | -0.0071 |
| Recall | 0.9860 | 0.9730 | -0.0130 |
| F1 | 0.9835 | 0.9735 | -0.0101 |
| ROC AUC | 0.9981 | 0.9973 | -0.0008 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (20 vs 20 epochs).
