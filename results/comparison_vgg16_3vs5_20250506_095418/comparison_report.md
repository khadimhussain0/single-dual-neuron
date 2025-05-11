# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** vgg16
- **Classification Task:** Class 3 vs Class 5
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.8880 | 0.8520 | -0.0360 |
| Precision | 0.8553 | 0.8614 | 0.0061 |
| Recall | 0.9340 | 0.8390 | -0.0950 |
| F1 | 0.8929 | 0.8501 | -0.0429 |
| ROC AUC | 0.9658 | 0.9306 | -0.0352 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
