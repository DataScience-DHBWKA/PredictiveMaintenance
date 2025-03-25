# 🛠️ Predictive Maintenance with Machine Learning

This repository presents a comprehensive machine learning project aimed at identifying machine failures through sensor data analysis. The project follows the structured CRISP-MLQ methodology, emphasizing Data Understanding, Data Preparation, and Modeling with rigorous Evaluation.

---

## 📊 Data Understanding

> **🔎 Objective:** Exploratory Data Analysis (EDA) to understand feature distributions, detect patterns, and identify correlations.

### 📌 Dataset Overview

The dataset used contains sensor readings from CNC milling machines, featuring numeric and categorical attributes:

| Feature | Description | Type |
|---------|-------------|------|
| air_temperature | Air temperature (in Kelvin) | Numeric |
| process_temperature | Process temperature (in Kelvin) | Numeric |
| rotational_speed | Rotational speed (in rpm) | Numeric |
| torque | Torque (in Nm) | Numeric |
| tool_wear | Tool wear duration (in minutes) | Numeric |
| machine_failure | Machine failure status (binary) | Categorical |

### 📌 Key Insights from EDA
- 🔥 **Heat Dissipation Failures** correlate strongly with high temperatures.
- ⚙️ **Power Failures** primarily occur at higher rotational speeds.
- 🔧 **Tool Wear and Overstrain Failures** appear significantly with higher tool wear durations.

### 📌 Correlation Highlights
- Strong negative correlation (ρ = -0.88) between torque and rotational speed.
- Positive correlation between air and process temperature.

> **💡 Tip:** Always visualize feature distributions and class imbalance early to inform data preparation strategies.

---

## 🧹 Data Preparation

> **🔄 Objective:** Transform raw data into a structured, balanced format suitable for modeling.

### 📌 Steps Taken

1. **Data Import & Type Adjustment**
```python
import pandas as pd

# Explicitly set correct data types for optimization
 df = pd.read_csv("data.csv", dtype={
   'air_temperature': 'float32',
   'machine_failure': 'bool',
   'type': 'category',
})
```

2. **Feature Engineering**
- Categorical variable encoding via One-Hot-Encoding.
- Created consolidated target variable `label` combining all failure types.

3. **Addressing Class Imbalance**
- Applied SMOTETomek Oversampling on training data only to balance class distributions:
```python
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
```

4. **Standardization**
- Numeric features standardized using `StandardScaler`:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
```

> **⚠️ Important:** Do not oversample test data to maintain evaluation validity.

---

## 🤖 Modeling & 📐 Evaluation

> **🎯 Objective:** Develop and compare machine learning models to accurately classify machine failures.

### 📌 Problem Formulation
- Multi-class classification problem with 6 distinct classes: TWF, HDF, PWF, OSF, RNF, and No Failure.

### 📌 Models Implemented
- Logistic Regression
- Random Forest (Best Performer 🎉)
- Decision Tree (with and without pruning)
- Support Vector Machine (SVM)
- K-Nearest Neighbors (optimized k)

### 📌 Hyperparameter Tuning
- Utilized Grid Search with 5-fold cross-validation:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1_macro', cv=5)
grid_search.fit(X_train_res_scaled, y_train_res)
```

### 📌 Evaluation Metrics
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest (Tuned)** | **96.35%** | **97.34%** | **96.35%** | **96.81%** |
| Decision Tree (Pruned) | 94.95% | 97.52% | 94.95% | 96.18% |
| SVM | 85.35% | 97.17% | 85.35% | 90.52% |

- Random Forest showed the best balance between Bias and Variance.

### 📌 Bias-Variance Analysis
- Models with tuned hyperparameters effectively reduced both Bias and Variance, enhancing generalization.

### 📌 Insights from Oversampling
- Boxplot analysis confirmed oversampling did not distort data distributions, ensuring valid evaluation.

> **🚀 Recommendation:** Random Forest (Hyperparameter-Tuned) is recommended due to high accuracy, excellent generalization, and robustness.

---

## 📂 Repository Structure
```bash
├── data/
│   ├── dataset_train_resampled.csv
│   └── dataset_test.csv
├── notebooks/
│   ├── DataUnderstanding.ipynb
│   ├── DataPreparation.ipynb
│   └── ModelBuilding.ipynb
└── README.md
```

---

✨ **Happy Modeling!** ✨
