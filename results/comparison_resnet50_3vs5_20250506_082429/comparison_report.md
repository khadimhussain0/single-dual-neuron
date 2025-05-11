# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** resnet50
- **Classification Task:** Class 3 vs Class 5
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9060 | 0.9040 | -0.0020 |
| Precision | 0.8942 | 0.9032 | 0.0090 |
| Recall | 0.9210 | 0.9050 | -0.0160 |
| F1 | 0.9074 | 0.9041 | -0.0033 |
| ROC AUC | 0.9704 | 0.9645 | -0.0059 |

## Key Findings

- The **single neuron** approach performed better overall for this experiment.
- Largest difference observed in **recall** metric.
- **Both approaches** neurons converged faster (15 vs 15 epochs).
