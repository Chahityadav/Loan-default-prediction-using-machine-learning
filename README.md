# Loan-default-prediction-using-machine-learning

🏦 Loan Default Prediction using Machine Learning

Predicting loan default risk using advanced machine learning algorithms and explainable AI techniques.
This project demonstrates a complete ML workflow — from data preprocessing and model building to explainability and interactive dashboards.

📘 Overview

The goal of this project is to build a predictive model that determines the likelihood of a borrower defaulting on a loan.
Using classification algorithms like Logistic Regression, Random Forest, and XGBoost, the model analyzes various financial and demographic factors to predict default risk.

The project also integrates SHAP (Explainable AI) for model interpretability and a Dash web dashboard for visualizing key insights.

🚀 Key Features

✅ End-to-end ML pipeline (data → model → dashboard)

🧹 Robust preprocessing (handling missing data, encoding, scaling)

📊 Exploratory Data Analysis (EDA) with visual insights

🧠 Multiple ML models — Logistic Regression, Random Forest, XGBoost

🔍 Model evaluation (Accuracy, F1 Score, ROC-AUC, Confusion Matrix)

📈 Feature importance and explainability using SHAP

💾 Model persistence with joblib

🌐 Interactive analytics dashboard built with Dash & Plotly

📤 Exportable results for Power BI or further analysis

🧩 Technologies & Libraries
Category	Tools / Libraries
Programming Language	Python
Data Handling	pandas, numpy
Visualization	matplotlib, seaborn, plotly, dash
Machine Learning	scikit-learn, xgboost
Explainability (XAI)	shap
Model Saving	joblib
Automation	Pipeline, ColumnTransformer
Dashboarding	Dash, Plotly, Bootstrap
🧠 Machine Learning Models Used

Logistic Regression — baseline linear classifier

Random Forest Classifier — ensemble model for robust predictions

XGBoost Classifier — gradient boosting model for optimized performance

Each model’s performance is compared using Accuracy, F1 Score, and ROC-AUC metrics.

📊 Dashboard Preview

The interactive dashboard includes:

Confusion Matrix visualization

ROC Curve analysis

Prediction probability distribution

Model performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)

Feature importance chart

Built using Dash, Plotly, and Bootstrap components.

⚙️ Project Workflow
1️⃣ Data Loading / Generation
2️⃣ Exploratory Data Analysis (EDA)
3️⃣ Feature Engineering
4️⃣ Data Preprocessing (Imputation, Encoding, Scaling)
5️⃣ Model Training & Evaluation (Logistic, RF, XGBoost)
6️⃣ Model Explainability using SHAP
7️⃣ Dashboard & Visualization
8️⃣ Model Export (loan_default_predictor.pkl)

📁 Folder Structure
loan_default_prediction/
│
├── loan_default_prediction_using_machine_learning.py
├── model_predictions.csv
├── model_metrics.csv
├── loan_default_predictor.pkl
├── README.md
└── requirements.txt
