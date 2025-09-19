# Bank Loan Default Prediction with MLflow

This project builds a machine learning pipeline to predict **loan credit policy approval** using the Bank Loan dataset.  
It uses scikit-learn pipelines, XGBoost for modeling, and integrates with **MLflow** for experiment tracking.

---

##  Features
- Preprocessing with scikit-learn `Pipeline` and `ColumnTransformer`
- Separate pipelines for numerical and categorical features
- Regression modeling with **XGBRegressor**
- Hyperparameter tuning via **RandomizedSearchCV**
- Command-line arguments for test size and random state
- Experiment tracking with **MLflow**
- Requirements management with `pipreqs`

---

##  Project Structure

bank_default_mlflow_project/
│
├── data/
│ └── loan_data.csv # Dataset
│
├── scripts/
│ ├── train.py # Training script with pipeline + MLflow
│ ├── prediction_loop.py # Manual prediction script
│ ├── predict_auto_loop.py # Background prediction automation
│ └── feature_engineering.py # Custom feature transformers
│
├── models/ # Saved pipelines and artifacts
├── logs/ # Logs with timestamps/rotation
├── requirements.txt # Dependencies
└── README.md # Project documentation

##  Installation

Clone and install:

```bash
git clone <your-repo-link>
cd bank_default_mlflow_project
pip install -r requirements.txt

Training with MLflow

python train.py --test_size 0.2 --random_state 101

Start the MLflow UI

mlflow ui

Then open http://127.0.0.1:5000

Tech Stack

Python 3.12

pandas, numpy, scikit-learn

XGBoost (XGBRegressor)

MLflow

logging

pipreqs for dependency management


Author

Built by Emmanuel Nwankwo
