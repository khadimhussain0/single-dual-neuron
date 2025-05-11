# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** vit
- **Classification Task:** Class 0 vs Class 1
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9965 | 0.9960 | -0.0005 |
| Precision | 0.9980 | 0.9980 | -0.0000 |
| Recall | 0.9950 | 0.9940 | -0.0010 |
| F1 | 0.9965 | 0.9960 | -0.0005 |
| ROC AUC | 0.9999 | 0.9999 | -0.0000 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
