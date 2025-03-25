```md
<!--
  README.md for the Data Understanding, Data Preparation, and Modeling + Evaluation phases
  of our Predictive Maintenance project. Written in English, with emojis and GitHub Markdown features.
-->

# Predictive Maintenance Project: Data Understanding, Preparation & Modeling

Welcome to our **Predictive Maintenance** repository! This project demonstrates how to predict different machine failure types using a synthetic dataset, following a structured approach inspired by the CRISP-MLQ methodology. This README provides an overview of the **Data Understanding**, **Data Preparation**, and **Modeling + Evaluation** stages.

> **Note**  
> This project was developed as part of a scientific work focusing on machine failure classification. Some code snippets and details are simplified for illustrative purposes.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Data Understanding](#data-understanding-🔎)
3. [Data Preparation](#data-preparation-⚙️)
4. [Modeling & Evaluation](#modeling--evaluation-🤖)
5. [Usage & How to Run](#usage--how-to-run-🚀)
6. [Tips & Tricks](#tips--tricks-💡)
7. [License](#license)

---

## Project Structure

```txt
.
├── DataUnderstanding.ipynb        <-- Notebook for initial EDA & visualization
├── DataPreperation.ipynb          <-- Notebook for cleaning, encoding, splitting
├── ModelBuilding.ipynb            <-- Notebook for training & evaluating models
├── dataset.csv                    <-- Original (synthetic) dataset
├── README.md                      <-- This file
└── ...
```

---

## Data Understanding 🔎

In this phase, we explore the dataset and derive insights about the features and potential relationships. Our primary goals:

1. **Explore the Dataset**  
   - The dataset contains numeric and categorical features describing CNC milling machines.  
   - Each row corresponds to one machine’s sensor readings (e.g., temperature, torque, tool wear, etc.).

2. **Identify Variable Types and Distributions**  
   - Categorical features (Machine Type: L, M, H)  
   - Numerical features (Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear)  
   - Binary flags for failures (TWF, HDF, PWF, OSF, RNF) and a combined failure indicator.

3. **Visualize & Summarize**  
   - We performed univariate and bivariate analyses (histograms, scatter plots, correlation heatmap).  
   - **Finding:** There is a heavy class imbalance—only ~3.39% of machines fail, and among failures, certain failure types are rare (e.g., `RNF`).

### Example EDA Code

```python
# DataUnderstanding.ipynb

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("dataset.csv")

# Quick statistical summary
print(df.describe())

# Correlation matrix
corr_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```

<details>
<summary><strong>📊 Example Findings</strong></summary>

- **Temperature**: Air Temperature and Process Temperature are (unsurprisingly) strongly correlated.  
- **Torque vs. Rotational Speed**: Strong negative correlation (~ -0.88).  
- **Failure Distribution**: Only a small portion of rows indicate failures, highlighting significant class imbalance.
</details>

---

## Data Preparation ⚙️

During Data Preparation, we transform the dataset into a form more suitable for modeling. Key steps:

1. **Data Cleaning**  
   - The synthetic dataset contained no missing or invalid entries.  
   - We confirmed no duplicates or out-of-range values existed.

2. **Feature Engineering**  
   - **One-Hot Encoding**: Converted `type` (L, M, H) into three binary features (`Type_L`, `Type_M`, `Type_H`).  
   - **Combined Target Label**: Merged multiple failure indicators into a single multi-class target (`no_failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).

3. **Handling Class Imbalance**  
   - We used **SMOTETomek** to oversample the minority failure types and remove Tomek links from majority classes.  
   - Importantly, we **only** oversampled **training** data. The test set remained untouched to keep an unbiased evaluation.

4. **Standardization**  
   - We applied a `StandardScaler` to numerical features (mean = 0, std = 1), crucial for distance-based models such as **kNN** or **SVM**.

### Example Preparation Code

```python
# DataPreperation.ipynb

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek

# 1. Load Data
df = pd.read_csv("dataset.csv")

# 2. One-Hot Encoding
df = pd.get_dummies(df, columns=["type"])

# 3. Create Multi-Class Target
def create_label(row):
    if row["machine failure"] == 0:
        return "no_failure"
    # If there's any machine failure, pick one from TWF, HDF, PWF, OSF, RNF
    # (assuming only one can be 1 at a time in this dataset)
    for ftype in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        if row[ftype] == 1:
            return ftype

df["label"] = df.apply(create_label, axis=1)

# 4. Train/Test Split
X = df.drop(["label", "machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    stratify=y,
    test_size=0.2,
    random_state=42
)

# 5. SMOTETomek Oversampling (only on training data)
sm = SMOTETomek(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 7. Export for modeling
pd.DataFrame(X_train_res_scaled).to_csv("dataset_train_resampled.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("dataset_test.csv", index=False)
y_train_res.to_csv("y_train_res.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
```

---

## Modeling & Evaluation 🤖

Here, we train multiple classification algorithms to predict the machine’s failure class. Our primary metric is the **F1-Score** (macro-averaged across classes).

1. **Models Considered**  
   - **Logistic Regression**  
   - **Random Forest**  
   - **Support Vector Machine**  
   - **k-Nearest Neighbors**  
   - **Decision Tree** (+ Pruning)  
   - **Random Forest** with Hyperparameter Tuning  
   - **kNN** with optimal k

2. **Hyperparameter Tuning**  
   - Used **GridSearchCV** with 5-fold cross-validation for *Random Forest* (`n_estimators`, `max_depth`, `min_samples_split`, etc.)  
   - Found the optimal **k** for kNN.

3. **Evaluation**  
   - Split data: **80% Training** + **20% Test**.  
   - **SMOTETomek** was applied **only** to the training subset.  
   - Calculated metrics: `Accuracy`, `Precision`, `Recall`, `F1-Score`.  
   - Created **scatterplots** comparing training vs. test F1-scores to check for overfitting.

4. **Best Model**  
   - **Random Forest with Hyperparameter Tuning** achieved the best F1-score (~0.96 on test data).  
   - Excellent generalization, no strong sign of overfitting.

### Example Modeling Code

```python
# ModelBuilding.ipynb

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load prepared data
X_train = pd.read_csv("dataset_train_resampled.csv")
X_test = pd.read_csv("dataset_test.csv")
y_train = pd.read_csv("y_train_res.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# Train a Random Forest (example hyperparameters)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"F1-Score (RF): {f1:.4f}")

# => Expected ~0.96 - 0.97
```

**Visualization Example (F1-Score Scatterplot)**

```python
import matplotlib.pyplot as plt

models = ["LogReg", "RF", "SVM", "KNN", "DT", "RF_tuned", "KNN_opt", "DT_pruned"]
f1_train_scores = [0.91, 1.00, 0.97, 0.99, 1.00, 1.00, 1.00, 0.99]   # example
f1_test_scores  = [0.79, 0.96, 0.85, 0.91, 0.95, 0.96, 0.94, 0.95]   # example

plt.figure(figsize=(8, 6))
plt.scatter(f1_train_scores, f1_test_scores, color="blue")
plt.plot([0.7, 1.0], [0.7, 1.0], "--", color="gray")
plt.xlabel("F1-Score (Train)")
plt.ylabel("F1-Score (Test)")
plt.title("Training vs. Test F1-Scores")
for i, txt in enumerate(models):
    plt.annotate(txt, (f1_train_scores[i], f1_test_scores[i]))
plt.show()
```

---

## Usage & How to Run 🚀

1. **Clone** this repository:
   ```bash
   git clone https://github.com/username/predictive-maintenance.git
   ```
2. **Install dependencies** (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebooks** in the following order:
   1. `DataUnderstanding.ipynb`  
   2. `DataPreperation.ipynb`  
   3. `ModelBuilding.ipynb`  
4. **Check results** in the final notebook cells or the generated metrics in the console.

---

## Tips & Tricks 💡

- **Tip**: Always split your data into train/test before applying SMOTE or other resampling techniques, to avoid data leakage.  
- **Pro Tip**: Use **Cross-Validation** (e.g., `GridSearchCV`) to get more robust estimates of model performance.  
- **Comment**: If you see extreme differences between training and testing scores, consider pruning or regularization to reduce overfitting.

---

## License

This project is published under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes. If you utilize or build upon our work, we kindly ask for appropriate citation or reference to this repository.

---

<!--
  End of README
-->
```
