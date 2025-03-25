```md
# 🏭 Predictive Maintenance: A CRISP-ML(Q) Approach

<!-- 
  This README is part of a scientific project focusing on predictive maintenance. 
  It identifies machine failures using sensor data and applies CRISP-ML(Q) methodology. 
  Please feel free to explore and adapt the provided materials. 
-->

## 🎯 Project Overview

Welcome to our Predictive Maintenance project! This repository demonstrates how machine learning and data-driven analysis can anticipate equipment failures before they cause costly downtime. By collecting and processing sensor data, we aim to **detect** and **classify** machine failures early so maintenance operations can be scheduled proactively.

<details>
<summary>🔎 Quick Facts</summary>

- **Focus**: Classification of machine failures using ML  
- **Data**: Synthetic data with multiple failure labels  
- **Methodology**: CRISP-ML(Q) (adapted without model maintenance)  
- **Techniques**: Oversampling (SMOTETomek), Classification algorithms (Random Forest, KNN, Decision Tree)  
- **Key Metric**: F1-score (balancing Precision & Recall)  
- **Goal**: Identify failure roots precisely, thereby reducing unplanned downtime  
</details>

---

## 📝 Table of Contents
1. [Introduction](#introduction)  
2. [Methodology](#methodology)  
   - [1. Business Understanding](#1-business-understanding)  
   - [2. Data Understanding](#2-data-understanding)  
   - [3. Data Preparation](#3-data-preparation)  
   - [4. Model Building](#4-model-building)  
   - [5. Evaluation](#5-evaluation)  
3. [Installation & Usage](#installation--usage)  
4. [Tips & Tricks](#tips--tricks)  
5. [Conclusions & Outlook](#conclusions--outlook)  
6. [License](#license)  

---

## Introduction

Predictive Maintenance leverages **sensor data** and **machine learning** models to detect anomalies and forecast potential failures. Our approach uses **CRISP-ML(Q)** as a guideline to ensure transparent and iterative ML development. The goal is to proactively replace parts or schedule machine servicing before major breakdowns occur.

> **Comment**: <!-- This section gives a concise overview of how we combine data science with operational needs. -->

---

## Methodology

### 1. Business Understanding

- **Objective**: Early identification of machine failures to reduce downtime and maintenance costs.  
- **Business Need**: Minimize production stoppages, avoid catastrophic failures, and optimize resource usage.

### 2. Data Understanding

- **Data Source**: Synthetic dataset representing sensor readings from a milling machine.  
- **Size**: ~10,000 rows × 14 columns.  
- **Class-Imbalance**: Only ~3.39% of all rows contain a failure → Imbalanced dataset.  
- **Failure Types**:  
  - Tool wear failure  
  - Heat dissipation failure  
  - Power failure  
  - Overstrain failure  
  - Random failures  

### 3. Data Preparation

1. **Categorical Encoding**:  
   - One-Hot-Encoding for machine type (e.g., Type_H, Type_M, Type_L).
2. **Label Creation**:  
   - Combine multiple failure flags (tool wear, power, heat dissipation, etc.) into one label column. 
3. **Train-Test Split**:  
   - 80% training, 20% test.
4. **Oversampling with SMOTETomek**:  
   - Handle imbalanced classes by adding synthetic minority samples.
5. **Scaling**:  
   - StandardScaler to normalize sensor values without losing overall distribution.

### 4. Model Building

- **Main Algorithms**:  
  1. **Decision Tree** (with pruning via `ccp_alpha`)  
  2. **Random Forest** (optimized via grid search over `n_estimators`, `max_depth`, `min_samples_split`, etc.)  
  3. **K-Nearest Neighbors** (optimized for best `k`)  

- **Hyperparameter Tuning**:  
  - Performed using grid search + 5-fold cross-validation to find best performance on the training data.

<details>
<summary>📌 Example Model Training Code (Python)</summary>

```python
# Example code snippet for Random Forest training

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=param_grid,
                           scoring='f1_macro',  # multi-class average F1
                           cv=5)

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best F1-Score:", grid_search.best_score_)
```
</details>

### 5. Evaluation

- **Metrics**:  
  - **Accuracy**  
  - **Precision**  
  - **Recall**  
  - **F1-Score** (primary metric to balance FN and FP)  
- **Observations**:  
  - Random Forest achieved the highest F1-score (~0.968 on test data).  
  - No strong sign of overfitting despite 1.0 F1 on training set.  
  - Oversampling maintained the distribution without major distortions.  

---

## Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/predictive-maintenance-ml.git
   cd predictive-maintenance-ml
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate     # On Linux/Mac
   .\venv\Scripts\activate      # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run notebooks or scripts**:
   - Explore `notebooks/data_preparation.ipynb` for data cleaning and preprocessing.
   - Run `python src/train_model.py` to train the model and save results.

5. **Check results**:
   - Evaluation metrics are displayed in your console or Jupyter cells.
   - Trained model artifacts are stored in `models/`.

---

## Tips & Tricks

> **:bulb: Tip:** 
> - Keep an eye on the **class imbalance**. Oversampling or undersampling strategies can significantly affect performance.  
> - Perform **feature scaling** consistently to avoid data leakage (fit on train set, then apply to test set).

> **⚠️ Caution:** 
> - Synthetic data may not perfectly represent real-world conditions. Additional tuning or domain expertise is crucial when applying these techniques to a live environment.

---

## Conclusions & Outlook

- **Best Model**: Random Forest with hyperparameter tuning.  
- **Key Benefits**:  
  - High recall to minimize undetected failures.  
  - High precision to reduce false alarms.  
- **Future Work**:  
  - Apply to **real-world production data** with continuous monitoring.  
  - Test additional ensemble approaches or deep learning models.  
  - Integrate with **real-time streaming** for on-the-fly predictions.

---

## License

This repository is available for academic and educational purposes. Please review the [LICENSE](LICENSE) file for more information.  
Enjoy exploring predictive maintenance! 🚀
```
