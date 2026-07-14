# Credit Card Approval Prediction Using IBM Watson Machine Learning

## 🌟 APSCHE Semester Term Project & Placement Portfolio

An intelligent, real-time Machine Learning system designed to automate the screening and evaluation of bank credit card applications. This project leverages advanced classifiers (Logistic Regression, Decision Trees, Random Forests, and XGBoost) to predict credit default risk, categorize applicant risk tier levels, and render instant approvals.

The complete system is designed to meet strict **APSCHE (Andhra Pradesh State Council of Higher Education)** evaluation guidelines and integrates with **IBM Watson Machine Learning** for online cloud deployment.

---

## 🏛️ Technical Architecture Workflow

The system is designed with a modular multi-tier architecture:

```
[User Web Browser] 
      │ (Fills demographic & financial evaluation form)
      ▼
[HTML5 / CSS3 / JS Presentation Layer]
      │ (Bootstrap UI validation & AJAX loading spinner)
      ▼
[Flask Application Controller] (app.py)
      │ (Handles routing, request parsing, in-memory logs, fallback heuristics)
      ▼
[Data Preprocessing & Feature Engineering Layer] (src/)
      │ (Ages/tenure conversion, categorical encoding, scaling transformers)
      ▼
[Machine Learning Inference Engine] (best_model.pkl)
      │ (Computes probability score via Random Forest/XGBoost)
      ▼
[IBM Watson Machine Learning Registry] (IBM Cloud)
      │ (Online model scoring, cloud repository, COS workspace)
      ▼
[Response Payload] 
      │ (Sends back predicted status, risk level, confidence gauge)
      ▼
[User Web Browser] (Displays visual approval/rejection panel)
```

---

## 📂 Project Structure

```
Credit_Card_Approval_Prediction/
│
├── dataset/
│   ├── application_record.csv      # Raw demographic profiles dataset
│   └── credit_record.csv           # Raw monthly payment history dataset
│
├── notebooks/
│   └── CreditCardApproval.ipynb    # Jupyter Notebook for EDA & model training
│
├── src/
│   ├── generate_data.py            # Script to generate realistic datasets
│   ├── data_preprocessing.py       # Data cleaning, target mapping, merging
│   ├── feature_engineering.py      # Scaling, one-hot encoding, feature creation
│   ├── train_models.py             # Model training, comparison and selection
│   ├── evaluate_models.py          # Metric calculation and plotting utilities
│   └── predict.py                  # Single record prediction inference
│
├── models/
│   ├── best_model.pkl              # Serialized best-performing model (XGBoost)
│   ├── encoder.pkl                 # Fitted categorical OneHotEncoder
│   └── scaler.pkl                  # Fitted numerical StandardScaler
│
├── templates/
│   ├── layout.html                 # Master layout template (Jinja2 inheritance)
│   ├── home.html                   # Bank portal dashboard landing page
│   ├── index.html                  # Applicant evaluation form
│   ├── result.html                 # Decision result page with gauge animation
│   ├── about.html                  # Project technical specs page
│   ├── contact.html                # Message sending card
│   └── error.html                  # Custom HTTP error pages (404/500/400)
│
├── static/
│   ├── css/
│   │   └── style.css               # Glassmorphic, dark/light theme stylesheet
│   └── js/
│       └── main.js                 # Theme toggler, form validation, gauge control
│
├── documentation/
│   ├── project_report.md           # 25+ Page Academic Project Report
│   ├── ppt_presentation.md         # 20+ Slides Presentation Text
│   ├── ibm_watson_guide.md         # Model deployment on IBM Cloud WML
│   ├── installation_guide.md       # Step-by-step setup walkthrough
│   ├── execution_guide.md          # Guide to run scripts and Flask backend
│   ├── user_manual.md              # User interaction guidelines
│   ├── developer_guide.md          # Architecture and development guide
│   └── testing_guide.md            # Test cases and quality assurance procedures
│
├── app.py                          # Flask backend controller
├── requirements.txt                # Python environment requirements
├── .gitignore                      # Git exclusion rules
└── README.md                       # Main landing documentation
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- Python 3.6+ installed.
- Access to a command terminal (PowerShell, Command Prompt, or Bash).

### 2. Setup & Installation
Clone or navigate to the project directory:
```bash
# Navigate to scratch folder
cd Credit_Card_Approval_Prediction

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Generate Datasets & Train Models
```bash
# Step A: Generate raw dataset CSV files
python src/generate_data.py

# Step B: Train models, generate comparison charts, and save best model
python src/train_models.py
```
This generates comparison tables and saves charts (`roc_curve.png`, `feature_importance.png`, `model_comparison.png`) under `static/images/`.

### 4. Run Flask Web Application
```bash
# Launch the web application
python app.py
```
Open your browser and navigate to `http://localhost:5000` to interact with the banking dashboard and run card evaluations.

---

## 🚀 IBM Watson WML Deployment Summary
1. Create an **IBM Cloud** account and provision a **Watson Machine Learning** (WML) service.
2. Store models (`best_model.pkl`, `scaler.pkl`, `encoder.pkl`) inside your **IBM Cloud Object Storage** workspace.
3. Use the Python `ibm-watson-machine-learning` client SDK to register the model assets into your Deployment Space.
4. Deploy the model online to generate a REST scoring URL for Flask endpoint calls.
5. Review the full guide under [ibm_watson_guide.md](file:///C:/Users/chman/.gemini/antigravity/scratch/Credit_Card_Approval_Prediction/documentation/ibm_watson_guide.md).
