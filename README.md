# Loan-Default-Risk
# Project Overview
This project focuses on building a robust machine learning pipeline to predict the risk of loan default. The goal is to enable better credit decisions, minimize financial losses, and improve customer risk profiling.

# Problem Statement
Financial institutions face significant challenges with loan defaults, leading to revenue losses and increased risk exposure. By leveraging machine learning, this project aims to predict whether a customer is likely to default, helping institutions make informed lending decisions.

# Dataset
The dataset includes customer demographic and financial information, along with historical loan performance.
Key features include:

- Customer demographics (age, gender, etc.)

- Financial behavior (loan amount, repayment history)

- Credit-related features

Target variable:

Default → 1 (default), 0 (no default)

# Approach

Data Preprocessing

Cleaning missing values

Feature engineering

Balancing the dataset (SMOTE)

# Modeling

Logistic Regression (baseline)

Random Forest

Gradient Boosting

Hyperparameter tuning

Evaluation Metrics

# Accuracy

Precision, Recall, F1-score

ROC-AUC

Confusion Matrix

# Results

The best-performing model achieved strong recall on the default class, ensuring fewer missed defaults.

Balanced accuracy between default and non-default classes after handling class imbalance.
