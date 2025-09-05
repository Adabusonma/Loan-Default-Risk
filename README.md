Loan Default Risk Prediction 💳

An end-to-end machine learning project that predicts whether a borrower will default on a loan. This project demonstrates how data science can be applied in financial risk management — from exploratory data analysis (EDA) and feature engineering to model building, evaluation, and deployment.

🚀 Project Overview

Financial institutions face major challenges in assessing loan default risk. Traditional credit scoring methods often miss subtle behavioral patterns that signal potential default.
This project builds a robust machine learning pipeline to predict loan default, enabling better credit decisions, minimizing financial losses, and improving portfolio quality.

Key Goals:

Identify high-risk borrowers before loan approval.

Improve prediction accuracy using behavioral + financial data.

Build a deployment-ready pipeline for real-time loan risk assessment.

# Problem Statement

Develop a classification model that predicts loan default risk using customer demographic, performance, and historical loan data.

The target variable is:

good_bad_flag = 1 → Loan repaid (Good)

good_bad_flag = 0 → Loan defaulted (Bad)

# Dataset

We worked with three datasets provided as .csv files:

Performance Data (trainperf.csv
)

Loan performance records

Shape: 4,368 rows × 10 columns

Demographics Data (traindemographics.csv
)

Borrower demographic details

Shape: 4,346 rows × 9 columns

Previous Loans Data (trainprevloans.csv
)

Historical borrowing behavior

Shape: 18,183 rows × 12 columns

### Key Features:

Loan Amount, Term Days, Interest Rate, Repayment Ratio

Borrower Age, Employment Status, Bank Account Type

Previous loan history & repayment patterns

# Approach

The workflow follows a full data science pipeline:

Exploratory Data Analysis (EDA)

Visualized loan amounts, terms, repayment ratios, and demographics.

Identified patterns in repayment burden, default behavior, and age distributions.

# Data Preprocessing

Handled missing values (imputation & dropping when >80%).

Converted date fields to datetime.

Removed duplicates.

Managed outliers (kept some as they represent real-world cases).

# Feature Engineering
Created meaningful new features, including:

Customer Age (loan creation date – birthdate)

Late Payment Rate (% of past loans paid late)

Repayment Efficiency Ratio (amount repaid ÷ amount due)

# Class Imbalance Handling

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance defaulters vs non-defaulters.

# Modeling
Trained and evaluated multiple models:

Logistic Regression (baseline, interpretable, strong performance)

Decision Tree

Random Forest

Gradient Boosting

XGBoost

LightGBM

# Evaluation Metrics

Accuracy, Precision, Recall, F1-score

ROC AUC

Confusion Matrix

# Deployment

Built a Streamlit web app: Loan Default Risk Predictor

Users can enter borrower + loan details to get real-time risk predictions.

# Key Insights

Loan amounts are clustered around ₦10,000–₦15,000 → mostly small, short-term loans.

Most borrowers are young (25–40 years) and relatively new (1–3 previous loans).

Repayment ratios cluster between 78%–88%, but full repayment is rare.

Late Payment Rate is the strongest predictor of default.

Logistic Regression outperformed more complex models, achieving higher recall (catching more true defaulters).

# Results

Best Model: Logistic Regression

Reason: High recall for default cases (critical for minimizing loan losses), interpretability, and alignment with business goals.

After Hyperparameter Tuning:

Recall for defaulters improved from 51% → 55%

False Negatives reduced, meaning fewer missed defaulters → direct financial savings.

# Business Implications

By catching more potential defaulters, the bank can save millions in avoided losses.

False positives (good borrowers wrongly flagged) represent lost opportunities, but far less costly than missed defaults.

Transparent models like Logistic Regression improve trust with regulators and credit officers.

With income data (missing here), predictions could become even stronger.

# Tech Stack

Python (Pandas, NumPy, Scikit-learn, Imbalanced-learn, XGBoost, LightGBM)

Visualization: Matplotlib, Seaborn, Plotly

Deployment: Streamlit
