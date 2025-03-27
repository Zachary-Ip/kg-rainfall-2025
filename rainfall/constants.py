from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

import numpy as np

RESULT_DIR = "results/"

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(random_state=42, n_jobs=-1),
    "RandomForest": RandomForestClassifier(random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    # "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    # "LGBM": LGBMClassifier(random_state=42),
    "SVC": SVC(probability=True, random_state=42),
}


# Define parameter grids for each model:
PARAM_GRIDS = {
    "LogisticRegression": {
        "clf__C": np.logspace(-3, 2, 10),
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l1", "l2"],  # Only valid for "liblinear"
        "clf__max_iter": [100, 300, 500],
    },
    "DecisionTree": {
        "clf__max_depth": [3, 5, 7, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__criterion": ["gini", "entropy"],
    },
    "ExtraTrees": {
        "clf__n_estimators": [50, 100, 150],
        "clf__max_depth": [5, 7, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2"],
    },
    "RandomForest": {
        "clf__n_estimators": [50, 100, 150],
        "clf__max_depth": [5, 7, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2"],
    },
    "XGBoost": {
        "clf__n_estimators": [50, 100, 150],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.3],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
    },
    "LGBM": {
        "clf__num_leaves": [31, 50],
        "clf__n_estimators": [50, 100, 150],
        "clf__learning_rate": [0.01, 0.1, 0.3],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_alpha": [0, 0.1, 1],
        "clf__reg_lambda": [0, 0.1, 1],
    },
    "SVC": {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto", 0.01, 0.1, 1],  # Controls influence range
    },
}

SUBMISSION_FILES = {
    "LogisticRegression": f"{RESULT_DIR}LogisticRegression_submission.csv",
    "DecisionTree": f"{RESULT_DIR}DecisionTree_submission.csv",
    "ExtraTrees": f"{RESULT_DIR}ExtraTrees_submission.csv",
    "RandomForest": f"{RESULT_DIR}RandomForest_submission.csv",
    "XGBoost": f"{RESULT_DIR}XGBoost_submission.csv",
    "CatBoost": f"{RESULT_DIR}CatBoost_submission.csv",
    "LGBM": f"{RESULT_DIR}LGBM_submission.csv",
    "KNN": f"{RESULT_DIR}KNN_submission.csv",
    "SVC": f"{RESULT_DIR}SVC_submission.csv",
}
