import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import argparse
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=101)
parser.add_argument("--data_path", type=str, default="loan_data.csv")
args = parser.parse_args()



# Load data
sal = pd.read_csv(args.data_path)

features = [
    'purpose', 'int.rate', 'installment', 'log.annual.inc',
    'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'
]
target = "credit.policy"

numerical_columns = [
    'int.rate', 'installment', 'log.annual.inc',
    'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
    'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid'
]
categorical_columns = ['purpose']

# Pipelines
numerical_pipeline = Pipeline([
    ("numerical_imputer", SimpleImputer(strategy="mean"))
])

categorical_pipeline = Pipeline([
    ("categorical_imputer", SimpleImputer(strategy="most_frequent")),
    ("categorical_encoder", OrdinalEncoder(unknown_value=None))
])

column_preprocessed = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
], verbose=True, remainder="passthrough")

# Model
model = XGBRegressor()
final_pipeline = Pipeline([
    ("processing", column_preprocessed),
    ("model", model)
])

# Hyperparameter search space
param = {
    "model__n_estimators": [100, 300, 500, 800, 1000],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "model__max_depth": [3, 4, 5, 6, 8, 10],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.1, 0.2, 0.5, 1],
    "model__reg_alpha": [0, 0.01, 0.1, 1, 10],
    "model__reg_lambda": [0.1, 1, 5, 10, 20]
}

grid = RandomizedSearchCV(
    estimator=final_pipeline,
    param_distributions=param,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error"
)

# Split
x_train, x_test, y_train, y_test = train_test_split(
    sal[features], sal[target],
    test_size=args.test_size,
    random_state=args.random_state
)

# Fit
grid.fit(x_train, y_train)
real_model = grid.best_estimator_

# Log best params and model
mlflow.log_params(grid.best_params_)
mlflow.sklearn.log_model(real_model, "model")
