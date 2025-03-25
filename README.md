# Predictive Maintenance – Data Understanding, Preparation, Modeling & Evaluation

<!-- 
  Welcome to the Predictive Maintenance project README! 
  This document focuses on Data Understanding, Data Preparation, Modeling, and Evaluation, 
  based on the content of our scientific work. 
-->

> **Note**  
> This repository explores how to detect machine failures using sensor data through various classification models. It focuses on the essential steps of data processing, model building, and evaluation.

---
## Table of Contents
1. [Data Understanding](#data-understanding-🔎)
2. [Data Preparation](#data-preparation-🛠)
3. [Modeling](#modeling-🤖)
4. [Evaluation](#evaluation-📊)
5. [How to Run](#how-to-run-▶️)
6. [References & Further Reading](#references--further-reading-📚)

---

## Data Understanding 🔎

In our project, we analyze **synthetically generated sensor data** for milling machines, aiming to classify **five failure categories** plus a *no_failure* case. The dataset characteristics can be summarized as follows:

- **Sensor Features**:  
  - *air temperature [K]* (mean ~300K, σ=2)  
  - *process temperature [K]* (air temperature + 10K, σ=1)  
  - *rotational speed [rpm]* (mean ~1500, σ=300)  
  - *torque [Nm]* (mean ~40, σ=10)  
  - *tool wear [min]* (range: 10–200)  

- **Categorical Variables**:  
  - *machine type* (L, M, H)  
  - *machine failure* (binary indicator)  

- **Failure Classes**:  
  - `TWF` = Tool Wear Failure  
  - `HDF` = Heat Dissipation Failure  
  - `PWF` = Power Failure  
  - `OSF` = Overstrain Failure  
  - `RNF` = Random Failures  
  - `no_failure` = No machine failure  

> :bulb: **Tip:**  
> Our analyses revealed strong **class imbalance** (only ~3.39% failures overall), which influences the choice of data preparation techniques.

### Statistical Insights
- Certain failure types appear at specific sensor ranges:
  - **HDF**: Occurs more frequently at higher temperatures.
  - **PWF**: Often at higher rotational speeds (> 1900 rpm).
  - **TWF** & **OSF**: Linked to higher tool wear (> 172 min).
- Strong negative correlation between **torque** and **rotational speed** (r ≈ -0.88).
- No missing or invalid data thanks to a well-crafted synthetic generation process.

---

## Data Preparation 🛠

Data preparation ensures the dataset is **cleaned**, **transformed**, and **standardized** for modeling:

1. **Data Import and Type Assignment**  
   We import the data with manual dtype specifications:
   ```python
   import pandas as pd

   # Example: Setting correct types during CSV import
   df = pd.read_csv(
       "predictive_maintenance.csv",
       dtype={
           "type": "category",
           "machine failure": "bool",
           "air temperature [K]": "float32",
           "process temperature [K]": "float32",
           "rotational speed [rpm]": "float32",
           "torque [Nm]": "float32",
           "tool wear [min]": "float32"
       }
   )
   ```

2. **One-Hot Encoding**  
   - We convert the categorical `machine type` (L, M, H) into dummy variables: `Type_L`, `Type_M`, `Type_H`.

3. **New Target Variable**  
   - We create a single target column `label` with six categories: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`, and `no_failure`.

4. **Train-Test Split**  
   - We split the data into **80% training** and **20% test** (e.g., `train_test_split` from scikit-learn).

5. **Oversampling**  
   - The classes are **highly imbalanced**. To address this, we apply SMOTETomek *only on training data*, preserving the original distribution in the test set:
     ```python
     from imblearn.combine import SMOTETomek

     sm = SMOTETomek(random_state=42)
     X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
     ```

6. **Feature Scaling**  
   - All **numeric features** are standardized via `StandardScaler` (mean=0, std=1) to ensure uniform scaling:
     ```python
     from sklearn.preprocessing import StandardScaler

     scaler = StandardScaler()
     X_train_scaled = scaler.fit_transform(X_resampled)
     X_test_scaled = scaler.transform(X_test)
     ```

> <!-- GitHub markdown comment -->  
> **Comment**: We store the processed data in `dataset_train_resampled.csv` and `dataset_test.csv` for quick reloading.

---

## Modeling 🤖

### Problem Definition
We frame this as a **multiclass classification** task with six target labels. Our goal is to predict one of the five failure types or `no_failure`.

### Algorithms & Hyperparameter Tuning
We trained **seven models**:

1. **Logistic Regression**  
2. **Random Forest**  
3. **Support Vector Machine (SVM)**  
4. **K-Nearest Neighbors (KNN)**  
5. **Decision Tree**  
6. **KNN with optimal K**  
7. **Pruned Decision Tree**

**Grid Search** with cross-validation (5-fold) was performed to optimize key hyperparameters. For example, we tuned:
- `n_estimators`, `max_depth` in **Random Forest**.
- `n_neighbors` in **KNN**.
- `ccp_alpha` for **Decision Tree Pruning**.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_scaled, y_resampled)

best_rf_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
```

> :warning: **Hint:**  
> We used **F1-score** as our primary metric because of the class imbalance and the importance of harmonic mean between precision and recall.

---

## Evaluation 📊

We evaluated each model on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

| Model                             | F1 (Train) | F1 (Test) |
|:----------------------------------|:----------:|:---------:|
| Logistic Regression               | 0.9161     | 0.8664    |
| Random Forest                     | 1.0000     | 0.9676    |
| Decision Tree                     | 1.0000     | 0.9621    |
| Support Vector Machine            | 0.9733     | 0.9052    |
| K-Nearest Neighbors               | 0.9891     | 0.9346    |
| **Random Forest (Tuned)**         | **1.0000** | **0.9681**|
| KNN (optimal K)                   | 1.0000     | 0.9490    |
| Decision Tree (Pruned)            | 0.9985     | 0.9618    |

### Key Findings
- **Random Forest with Hyperparameter Tuning** yielded the highest F1-score (≈0.968 on test data).
- **Oversampling** improved minority class detection without causing overfitting.  
- **Scatterplots** comparing train vs. test F1-scores showed no major overfitting.

> :heavy_check_mark: **Conclusion**: Random Forest (tuned) is our best model, with strong generalization demonstrated on original test data.

---

## How to Run ▶️

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/PredictiveMaintenance.git
   cd PredictiveMaintenance
   ```

2. **Install Requirements**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preprocessing & Modeling**  
   ```bash
   python data_preparation.py
   python model_building.py
   ```

4. **Evaluation**  
   ```bash
   python evaluate_model.py
   ```

---

## References & Further Reading 📚

- **Extracts from the Scientific Work**:  
  - Our references include the *Predictive Maintenance* project documentation, covering data exploration, modeling, and evaluation steps.
- **External Libraries**:  
  - [scikit-learn.org](https://scikit-learn.org/)
  - [imbalanced-learn.org](https://imbalanced-learn.org/)

---

> **Disclaimer**  
> This README is based solely on the **concrete content** of the scientific work. All analysis steps, findings, and discussion points adhere to the original methodology and results. 

Enjoy exploring predictive maintenance with sensor data! 🚀
