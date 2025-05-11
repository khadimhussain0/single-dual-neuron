# Single vs. Dual Neuron Binary Classification Comparison

## Experiment Details

- **Model Architecture:** resnet50
- **Classification Task:** Class 2 vs Class 9
- **Batch Size:** 32
- **Max Epochs:** 15
- **Learning Rate:** 0.001

## Performance Comparison

| Metric | Single Neuron | Dual Neuron | Difference (Dual - Single) |
| ------ | ------------- | ---------- | -------------------------- |
| Accuracy | 0.9935 | 0.9955 | 0.0020 |
| Precision | 0.9891 | 0.9950 | 0.0059 |
| Recall | 0.9980 | 0.9960 | -0.0020 |
| F1 | 0.9935 | 0.9955 | 0.0020 |
| ROC AUC | 0.9999 | 0.9998 | -0.0002 |

## Key Findings

- The **dual neuron** approach performed better overall for this experiment.
- Largest difference observed in **precision** metric.
- **Dual** neuron converged faster (14 vs 15 epochs).
