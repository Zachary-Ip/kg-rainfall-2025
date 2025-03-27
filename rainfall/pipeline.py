import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


import warnings
from rainfall.constants import SEED

warnings.filterwarnings("ignore")

np.random.seed(SEED)


def train_and_submit(
    test_ids, X_train, y_train, X_pred, model, param_dist, model_name, dir
):
    pipeline = Pipeline(
        [
            (
                "selector",
                SelectFromModel(
                    ExtraTreesClassifier(n_estimators=100, random_state=SEED),
                    threshold="median",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", model),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=100,
        scoring="roc_auc",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_score = search.best_score_
    print(f"Best CV ROC AUC for {model_name}: {best_score:.4f}")
    print("Best Params:", search.best_params_)

    # Extract selected features from the selector step
    # selector = RFE(estimator=ExtraTreesClassifier(n_estimators=100), n_features_to_select=10)
    selector = search.best_estimator_.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_features = X_train.columns[selected_mask]
    print(f"Selected features for {model_name}:")
    print(selected_features.tolist())

    # Reindex test set to match training set columns
    training_cols = X_train.columns
    X_pred_aligned = X_pred.reindex(columns=training_cols, fill_value=0)

    try:
        # If applying recursive prediction for lag based features (rain yesterday, days since rain)
        # code would go here
        preds = search.predict_proba(X_pred_aligned.values)[:, 1]
    except Exception as e:
        print(f"Error in predict_proba for {model_name}: {e}")
        preds = np.full(len(X_pred_aligned), 0.5)

    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Save the model
    model_path = output_dir / f"{model_name}_submission.csv"
    submission = pd.DataFrame({"id": test_ids, "rainfall": preds})
    submission.to_csv(model_path, index=False)
    return best_score


def model_selector(models, param_grids, X, y, X_test, test_ids, dir):
    """
    Train and evaluate models on the given dataset."
    """

    model_results = {}

    for model_name, model in models.items():
        print(
            f"\nTraining {model_name} on Extended IQR-treated & Standard Scaled features:"
        )
        auc = train_and_submit(
            test_ids, X, y, X_test, model, param_grids[model_name], model_name, dir
        )
        model_results[model_name] = auc

    results_df = pd.DataFrame(
        model_results.items(), columns=["Model", "AUC"]
    ).sort_values(by="AUC", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="AUC", y="Model", data=results_df, palette="viridis")
    plt.title("Model Comparison (Extended Features, IQR-treated & Standard Scaled)")
    plt.xlabel("ROC AUC")
    plt.xlim(0.75, 1.0)
    plt.show()

    return model_results


# Test train split
# from imblearn.over_sampling import SMOTE
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# from sklearn.model_selection import train_test_split
