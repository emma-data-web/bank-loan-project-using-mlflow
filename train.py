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

parser = argparse.ArgumentParser()

parser.add_argument("--test_size", type=float, default=0.2)

args = parser.parse_args()

#model = joblib.load("loan_model.pkl")

sal = pd.read_csv("loan_data.csv")

features = ['purpose', 'int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']

traget = "credit.policy"

numerical_columns =  ['int.rate', 'installment', 'log.annual.inc',
       'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
       'inq.last.6mths', 'delinq.2yrs', 'pub.rec', 'not.fully.paid']


categorical_columns = ['purpose']

numerical_pipeline = Pipeline(steps=[
    ("numerical_imputer",SimpleImputer(strategy="mean"))
])

categorical_pipeline = Pipeline(steps=[
    ("categorical_imputer",SimpleImputer(strategy="most_frequent")),
    ("categorical_encoder", OrdinalEncoder(unknown_value=None)) 
])

column_preprocesed = ColumnTransformer(transformers=[
    ("num",numerical_pipeline,numerical_columns),
    ("cat",categorical_pipeline,categorical_columns)
], verbose=True, remainder="passthrough")

model = XGBRegressor()

final_pipeline =  Pipeline(steps=[
    ("processing",column_preprocesed),
    ("model", model)
])

param= {
    "model__n_estimators": [100, 300, 500, 800, 1000],       # number of trees
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],    # step size
    "model__max_depth": [3, 4, 5, 6, 8, 10],                # tree depth
    "model__min_child_weight": [1, 3, 5, 7],                # min sum of weights
    "model__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],          # row sampling
    "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],   # feature sampling
    "model__gamma": [0, 0.1, 0.2, 0.5, 1],                  # min loss reduction
    "model__reg_alpha": [0, 0.01, 0.1, 1, 10],              # L1 regularization
    "model__reg_lambda": [0.1, 1, 5, 10, 20]                # L2 regularization
}

grid = RandomizedSearchCV(
    estimator=final_pipeline,
    param_distributions= param,
    n_iter=20,
    cv=3,
    n_jobs=-1,
    scoring="neg_root_mean_squared_error"
)

x_train, x_test, y_train, y_test = train_test_split(sal[features], sal[traget], test_size=0.3, random_state=101)

grid.fit(x_train, y_train)

real_model = grid.best_estimator_