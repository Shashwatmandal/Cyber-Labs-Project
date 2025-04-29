# 📚 Machine Learning Library from Scratch

## 📌 Problem Statement

Develop a machine learning library by implementing the following algorithms from scratch: **Linear Regression (and Polynomial Regression), Logistic Regression, K-Nearest Neighbors (KNN), K-Means Clustering, and N-Layer Neural Network**. Train the implemented algorithms on a provided training dataset and evaluate their performance on a hidden test dataset to validate your implementation. Prepare a detailed report documenting all methodologies, experiments, and decisions, including rejected approaches with justifications.

---

## 🚀 Approach Summary

### 🔹 Dataset Overview
- **Linear, Polynomial, Logistic Regression:** Dataset with 20 features and 1 target label.
- **KNN, K-Means, Neural Network:** Dataset with 12 features and categorical/multi-class labels.

---

## 📈 Implemented Algorithms and Approaches

### 1. **Linear Regression**
- Implemented using **vectorized operations** to boost performance and reduce code complexity.
- **Model:** `f(x) = w·x + b`
- **Normalization:** Z-score standardization (mean = 0, std = 1)
- **Optimization:** Batch Gradient Descent
- **Cost Function:** Mean Squared Error (MSE)
- **Performance:**
  - MSE (Train): ~0.127
  - R² Score: 1.0 (Train), 0.9999 (Cross Validation)
- **Hyperparameters:**
  - Iterations: 1000
  - Learning Rate: 0.01
  - Weight & Bias Init: Zeros

---

### 2. **Polynomial Regression**
- Extended linear regression with **degree-5 polynomial features** (total 55).
- **Preprocessing:** Z-score normalization
- **Optimization:** Gradient Descent
- **Cost Function:** MSE
- **Performance:**
  - R² Score: 0.946 (Train), 0.938 (CV)
  - Cost reduced from ~5.4e13 to ~3.35e12
- **Hyperparameters:**
  - Iterations: 20,000
  - Learning Rate: 0.001
- Optimal degree should have used - **6**
---

### 3. **Logistic Regression**
- Multi-class classification using **one-hot encoding**.
- **Model:** `sigmoid(X·θ)`
- **Cost Function:** Binary Cross Entropy
- **Gradient Descent:** `θ = θ - α * ∇θ`
- **Preprocessing:** Z-score normalization, bias incorporated into data matrix
- **Performance:**
  - Accuracy: 90.48% (Train), 90.37% (Test)
  - Smooth cost reduction observed over 1000 iterations
- **Hyperparameters:**
  - Iterations: 1000
  - Learning Rate: 0.01
  - Weight Init: Random (scaled)

---

### 4. **K-Nearest Neighbors (KNN)**
- **Instance-based learning algorithm** implemented from scratch.
- **Steps:**
  - Calculated Euclidean distance to all training points
  - Sorted distances and selected K nearest
  - Chose most frequent label among K
- **Performance:**
  - Accuracy: 95.59%
  - Time: ~8 minutes
- **Hyperparameters:**
  - K = 60
- **Preprocessing:** Standardized input features

---

### 5. **K-Means Clustering**
- **Process:**
  - Random centroid initialization
  - Assignment based on Euclidean distance
  - Iterative centroid updates until convergence
- **Visualization:** Clusters plotted using `matplotlib`
- **Result:** 13 well-separated clusters and centroid plot
- **Hyperparameters:**
  - K = 13
  - Iterations: Max 200
- **Convergence:** Based on minimal centroid movement threshold (`< 0.0001`)

---

### 6. **Feedforward Neural Network**
- Built a **multi-layer neural network** with:
  - 3 Hidden Layers: 50 → 25 → 15 neurons
  - 1 Output Layer: 10 neurons
- **Activations:** ReLU (hidden layers), Softmax (output layer)
- **Training:**
  - Forward Propagation and Backpropagation coded manually
  - Loss: Categorical Cross-Entropy
- **Preprocessing:** Z-score normalization + one-hot encoding
- **Performance:**
  - Accuracy: 94.53% (Test)
  - Cost reduced from 2.29 to 0.19 over 2250 iterations
- **Hyperparameters:**
  - Iterations: 2250
  - Learning Rate: 0.009

---

## 🏁 Final Note

This was my **first major Machine Learning project in college**, where I implemented core ML algorithms from scratch without using any high-level libraries. It gave me hands-on exposure to how machine learning models work internally and strengthened my fundamentals.

---

## 👤 Author

**Shashwat Mandal**  
