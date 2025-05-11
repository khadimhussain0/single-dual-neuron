# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** vgg16
- **Classification Task:** Class 0 vs Class 1
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9885 | 0.9755 | -0.0130 |
| Precision | 0.9813 | 0.9789 | -0.0024 |
| Recall | 0.9960 | 0.9720 | -0.0240 |
| F1 | 0.9886 | 0.9754 | -0.0132 |
| ROC AUC | 0.9997 | 0.9954 | -0.0043 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
