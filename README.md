# 🏦 ML Loan Approval Prediction System

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial data. Two classification models are trained, evaluated, and compared — with Random Forest selected as the best performer.

---

## 📌 Overview

This project builds an end-to-end loan approval prediction pipeline — from raw data ingestion and preprocessing to model training, evaluation, and visualization. It targets binary classification: `1 = Approved`, `0 = Rejected`.

---

## 🧠 Models Used

| Model | Notes |
|---|---|
| Logistic Regression | Scaled features, `liblinear` solver, balanced class weights |
| Random Forest ⭐ | 500 estimators, best overall performance |

---

## 📊 Evaluation Metrics

Each model is assessed using:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (heatmap)
- ROC Curve & AUC Score
- 5-Fold Cross-Validation
- Specificity (True Negative Rate)

---

## 🗂 Project Structure

```
├── Loan_Approval_Prediction_System.ipynb   # Main notebook
├── loan_data.csv                           # Dataset
└── README.md
```

---

## 🔄 Workflow

```
Load CSV Dataset
       ↓
Exploratory Data Analysis (EDA)
  - Loan status distribution
  - CIBIL score vs loan status
  - Income & loan amount distributions
       ↓
Data Preprocessing
  - Drop ID columns
  - Fill missing values (median for numeric, mode for categorical)
  - One-Hot Encoding
       ↓
Train-Test Split (80/20, stratified)
       ↓
Feature Scaling (Logistic Regression only)
       ↓
Model Training (Logistic Regression + Random Forest)
       ↓
Evaluation & Comparison
       ↓
Best Model Selected → Random Forest
```

---

## 📈 Key Visualizations

- **Loan Status Distribution** — class balance check
- **CIBIL Score vs Loan Status** — credit score impact
- **Annual Income Distribution** — applicant income spread
- **Loan Amount Distribution** — with KDE overlay
- **Confusion Matrices** — for both models
- **ROC Curves** — with AUC scores
- **Top 10 Feature Importances** — from Random Forest
- **ML Pipeline Flow Diagram** — end-to-end process overview

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook Loan_Approval_Prediction_System.ipynb
```

> Make sure `loan_data.csv` is in the same directory as the notebook.

---

## 🔑 Key Features in Dataset

| Feature | Description |
|---|---|
| `cibil_score` | Applicant's credit score |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_status` | Target variable (Approved / Rejected) |

---

## 🛠 Tech Stack

- Python 3
- scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Jupyter Notebook

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
