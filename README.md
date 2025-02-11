# RNNs for Text Classification

## Overview
This repository implements **Recurrent Neural Networks (RNNs) for sequence classification**, focusing on **text classification tasks**. It includes **Vanilla RNNs, GRUs, and LSTMs** for classifying questions into six categories. The models are trained and evaluated on a **question classification dataset**.

## Dataset
The dataset consists of 2,000 labeled questions categorized into six classes:

- Abbreviation (ABBR)
- Entity (ENTY)
- Description (DESC)
- Human (HUM)
- Location (LOC)
- Numeric (NUM)

### **Data Split**
- **80%** training set
- **10%** validation set
- **10%** test set

## Model Architecture
This repository includes different RNN-based architectures:
- **Simple RNN**: A basic recurrent network with hidden layers.
- **GRU (Gated Recurrent Units)**: An improved RNN model that addresses vanishing gradient issues.
- **LSTM (Long Short-Term Memory)**: A powerful recurrent model for long-range dependencies.

### **Data Processing**
- Tokenization using **BERT Tokenizer**.
- Categorical labels converted to numerical values using **Label Encoding**.
- **PyTorch DataLoader** is used for batching.

## Training Methodology
### **1. Training Setup**
- Used **PyTorch** to build and train RNN models.
- **Adam Optimizer** with a learning rate of `1e-3`.
- **Cross-Entropy Loss** for classification.

### **2. Models Trained**
- **Simple RNN** (with `mean`, `max`, and `last_state` pooling).
- **GRU** (Gated Recurrent Units).
- **LSTM** (Long Short-Term Memory networks).

### **3. Results**
| Model          | Output Type  | Train Accuracy | Validation Accuracy |
|---------------|-------------|---------------|---------------------|
| Simple RNN    | Mean        | ~91.88%       | ~91.92%             |
| Simple RNN    | Max         | ~100.00%      | ~95.95%             |
| Simple RNN    | Last State  | ~36.22%       | ~38.88%             |
| LSTM          | Mean        | ~100.00%      | ~96.97%             |
| LSTM          | Max         | ~100.00%      | ~98.48%             |

## Installation
To run this project, install the required dependencies:
```bash
pip install torch transformers datasets numpy pandas scikit-learn matplotlib
```

## How to Run
### 1. Clone the Repository
```bash
git clone https://github.com/mohibkhann/RNNs-Sequence-Classification.git
cd RNNs-Sequence-Classification
```

### 2. Run the Python Script
```bash
python train_rnn.py
```

## Conclusion
This project showcases **Recurrent Neural Networks (RNNs), GRUs, and LSTMs** for text classification tasks. It demonstrates how different pooling methods (`mean`, `max`, `last_state`) impact performance. Future improvements may include **Bidirectional RNNs** and **attention mechanisms**.

---
**Author:** Mohib Ali Khan

