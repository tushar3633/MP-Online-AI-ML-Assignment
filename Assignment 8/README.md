# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

**Author:** Tushar Verma

**Registration Number:** 23BCE10097

**Application Number:** IN26011831

**Batch Number:** 1A

**Email ID:** tushar366.verma@gmail.com 

## Objective
The objective of this project is to develop an Artificial Neural Network (ANN) using TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset to automate postal code recognition.

## Dataset Link
- [Kaggle: MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

## Libraries Used
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow` / `keras`
- `scikit-learn`
- `kaggle`

## Methodology
1. **Data Understanding**: Analyzed $28 \times 28$ pixel feature representations (784 features) and visualized raw digit samples.
2. **Data Preprocessing**:
   - Verified no missing values.
   - Normalized pixel values to the range $[0, 1]$.
   - Split dataset into 80% training (48,000 samples) and 20% testing (12,000 samples).
   - Applied One-Hot Encoding to the target labels.
3. **Model Architecture**:
   - **Input Layer**: 784 nodes
   - **Hidden Layer 1**: 128 neurons (ReLU activation)
   - **Hidden Layer 2**: 64 neurons (ReLU activation)
   - **Output Layer**: 10 neurons (Softmax activation)
4. **Model Compilation & Training**:
   - **Optimizer**: Adam
   - **Loss Function**: Categorical Crossentropy
   - Trained for 10 epochs.
5. **Evaluation**: Computed test accuracy, loss, confusion matrix, classification report, and plotted accuracy/loss curves per epoch.

## Model Architecture Summary

| Layer | Type | Output Shape | Param # |
| :--- | :--- | :--- | :--- |
| `dense` | Dense (ReLU) | (None, 128) | 100,480 |
| `dense_1` | Dense (ReLU) | (None, 64) | 8,256 |
| `dense_2` | Dense (Softmax) | (None, 10) | 650 |

## Results
- **Test Accuracy:** 97.41%
- **Test Loss:** 0.0983
- **Macro F1-Score:** 0.9739

## Conclusion
The multi-layer ANN accurately classifies handwritten digits, proving the effectiveness of deep hidden layers for automatic feature extraction. While fully connected networks perform well on normalized datasets like MNIST, CNN architectures remain preferable for complex computer vision tasks due to spatial translation invariance.