📊 Predictive Maintenance with Machine Learning

🎯 Project Overview

This project explores the use of machine learning (ML) techniques to predict machine failures based on sensor data, contributing significantly to reducing unplanned downtime.

📚 Data Understanding

🔍 Dataset

Source: Synthetic sensor data from milling machines (UCI ML Repository).

Features:

Numerical: Air Temperature, Process Temperature, Rotational Speed, Torque, Tool Wear

Categorical: Machine Type (L/M/H)

Target Variables: Types of failures (TWF, HDF, PWF, OSF, RNF)

📈 Exploratory Data Analysis (EDA)

Conducted statistical analyses and created visualizations:

📌 Class imbalance identified (3.39% failure rate)

📌 Feature insights: High correlation between rotational speed & torque (-0.88)

📌 Significant patterns:

HDF occurs at higher temperatures

PWF linked to higher rotational speeds

TWF and OSF occur with high tool wear

🛠️ Data Preparation

⚙️ Steps Undertaken

Data Import & Type Adjustment:

Corrected data types manually during import (e.g., numerical, categorical, binary).

Applied one-hot encoding to categorical features.

Data Splitting & Oversampling:

Split into Training/Test sets (80:20).

Applied SMOTETomek to address class imbalance in training set.

Data Scaling:

StandardScaler used for numerical variables.

💾 Resulting Datasets

Training data: dataset_train_resampled.csv

Testing data: dataset_test.csv

🤖 Modeling

🎯 Problem Definition

Multi-class classification (6 categories: TWF, HDF, PWF, OSF, RNF, no_failure)

🧩 Models Implemented

Logistic Regression

Random Forest (with hyperparameter tuning)

Decision Trees (standard & pruned)

Support Vector Machines (SVM)

K-Nearest Neighbors (standard & optimized)

⚙️ Hyperparameter Optimization

Grid Search with 5-fold cross-validation for Random Forest and KNN

Cost-Complexity-Pruning for Decision Trees

🧪 Evaluation

📐 Metrics Used

Accuracy, Precision, Recall, and F1-Score

🥇 Best Performing Model

🌲 Random Forest with Hyperparameter Tuning:

Training F1-Score: 1.000

Testing F1-Score: 0.9681

📊 Scatterplot Analysis

Evaluated consistency between training and test metrics to identify overfitting.

📉 Bias-Variance Tradeoff

Analyzed flexibility parameters influencing model performance.

🗣️ Discussion

Confirmed oversampling improved model generalization without data distortion.

Final evaluation conducted exclusively on original (non-synthetic) test data.

🚀 Conclusions

ML effectively predicts machine failures.

Random Forest with hyperparameter tuning is highly reliable and robust.

Future work: Validation with real industry data and real-time system integration.

📦 Tools & Resources

Python, pandas, imbalanced-learn, scikit-learn

Visualization: Matplotlib, seaborn

🏫 Authors

Daniel Weissenberger

Eduardo Stein Mössner

Jonas Sigmund

👩‍🏫 Supervisor

Prof. Dr. Jennifer Schoch

📅 Date

25.03.2025

🌟 Thanks for exploring our Predictive Maintenance ML Project! 🌟

