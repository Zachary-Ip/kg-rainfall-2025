import pandas as pd
from rainfall.constants import SUBMISSION_FILES, RESULT_DIR
import shutil

from sklearn.impute import SimpleImputer


def reduce_memory_usage(df):
    """Downcasts numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")  # Downcast integers first
        df[col] = pd.to_numeric(df[col], downcast="float")  # Then downcast floats
    return df


def load_training_data(path):
    """
    Load training data from a CSV file.
    """
    df = pd.read_csv(path)
    df = reduce_memory_usage(df)
    df.columns = df.columns.str.strip()

    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # If rainfall is in Dtype object strings, convert to 1, 0 floats
    if df["rainfall"].dtype == "object":
        df["rainfall"] = df["rainfall"].replace({"no": 0, "yes": 1})
        df["rainfall"] = df["rainfall"].astype(float)

    # Impute missing values with median
    imputer = SimpleImputer(strategy="median")
    df.iloc[:, :] = imputer.fit_transform(df)

    X = df.drop(columns=["rainfall"])
    y = df["rainfall"]
    return X, y


def load_test_data(path):
    """
    Load test data from a CSV file.
    """
    df = pd.read_csv(path)
    df = reduce_memory_usage(df)

    if "id" in df.columns:
        test_ids = df["id"]
        df.drop(columns=["id"], inplace=True)
    else:
        test_ids = None

    return df, test_ids


def save_best_model_submission(model_results):
    # Identify best model based on AUC scores
    best_model = max(model_results, key=model_results.get)
    best_submission_file = SUBMISSION_FILES.get(best_model)

    # Save best model's submission as 'submission.csv'
    if best_submission_file is None:
        raise ValueError("Best model submission file not found.")
    else:
        shutil.copy(best_submission_file, f"{RESULT_DIR}submission.csv")
        print(f"Best scoring model: {best_model}: {model_results[best_model]:.4f}")
        print(f"Copied {best_submission_file} as 'submission.csv'")

    # Confirm all individual model submission files exist
    print("\n All individual model submissions saved:")
    for _, filename in SUBMISSION_FILES.items():
        print(f"- {filename}")
