import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from kneed import KneeLocator
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from rainfall.constants import SEED


def feat_sin(f, max_value):
    # convert day to radians
    return math.sin(f * 2 * math.pi / max_value)


def feat_cos(f, max_value):
    # convert day to radians
    return math.cos(f * 2 * math.pi / max_value)


def date_feature(df):
    # convert day to month by taking the floor of the day divided by 30
    df["month"] = (df["day"] // 30).astype(int)
    return df


def temp_features(df):
    """
    Add temperature features to the dataframe.
    """

    if "mintemp" in df.columns and "dewpoint" in df.columns:
        df["deg_below_dew"] = df["dewpoint"] - df["mintemp"]

    if "maxtemp" in df.columns and "mintemp" in df.columns:
        df["temp_range"] = df["maxtemp"] - df["mintemp"]
    return df


def poly_features(df, cols_to_expand):
    """
    Generate polynomial features (degree 2) for selected columns.
    Missing values are imputed (median) before transformation.
    """
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[cols_to_expand])

    # Get correct feature names directly from the transformation
    poly_feature_names = poly.get_feature_names_out(cols_to_expand)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

    return poly_df


def diff_features(df):
    df_diff = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        if "day" not in col and "month" not in col:
            df_diff[col + "_diff"] = df[col].diff().fillna(0)

    return df_diff


def rolling_features(df, window_sizes=[7, 14]):
    """
    Generate rolling window statistics (mean and std) for all numeric columns.
    Returns a DataFrame with new rolling features as 1D arrays.
    """
    df_roll = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        if "day" not in col and "month" not in col:
            series_col = df[col]
            for window in window_sizes:
                roll_mean = (
                    series_col.rolling(window=window, min_periods=1).mean().values
                )
                roll_std = (
                    series_col.rolling(window=window, min_periods=1)
                    .std()
                    .fillna(0)
                    .values
                )
                if roll_mean.ndim > 1:
                    roll_mean = roll_mean[:, 0]
                if roll_std.ndim > 1:
                    roll_std = roll_std[:, 0]
                df_roll[f"{col}_roll_mean_{window}"] = roll_mean
                df_roll[f"{col}_roll_std_{window}"] = roll_std
    return df_roll


def interaction_features(df):
    """
    Create interaction features for the dataframe.
    """
    if (
        "temp_change" in df.columns
        and "humidity" in df.columns
        and "pressure" in df.columns
    ):
        df["weather"] = -df["temp_change"] * df["humidity"] / df["pressure"]

    if "sunshine" in df.columns and "cloud" in df.columns:
        df["sunshine_cloud_ratio"] = df["sunshine"] / (df["cloud"] + 1e-3)
        df["cloud_sunshine_ratio"] = df["cloud"] / (df["sunshine"] + 1e-3)

    if "pressure" in df.columns and "humidity" in df.columns:
        df["humidity_pressure_ratio"] = df["humidity"] / (df["pressure"] + 1e-3)

    if "pressure" in df.columns and "mintemp" in df.columns:
        df["min_temp_pressure_ratio"] = df["mintemp"] / (df["pressure"] + 1e-3)

    if "pressure" in df.columns and "deg_below_dew" in df.columns:
        df["dewpoint_pressure_ratio"] = df["deg_below_dew"] / (df["pressure"] + 1e-3)

    if "cloud" in df.columns and "humidity" in df.columns:
        df["cloud_humidity_mult"] = df["cloud"] * df["humidity"]

    if "cloud" in df.columns and "deg_below_dew" in df.columns:
        df["cloud_dew_mult"] = df["cloud"] * df["deg_below_dew"]

    if "cloud" in df.columns and "winddirection" in df.columns:
        df["cloud_wind_mult"] = df["cloud"] * df["winddirection"]

    return df


def treat_outliers_iqr(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        series = df[col].squeeze()  # Ensure it's a 1D array/Series
        Q1 = np.nanquantile(series, 0.25)
        Q3 = np.nanquantile(series, 0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # Convert to NumPy array, clip, and assign back
        df[col] = np.clip(series.values, lower, upper)
    return df


def add_features(df):
    """
    Add features to the dataframe.
    """
    new_df = df.copy()
    # Convert day to month
    new_df = date_feature(new_df)

    # Add temperature features
    new_df = temp_features(new_df)

    # Add interaction features
    new_df = interaction_features(new_df)

    # Decompose cyclic features
    new_df["day_sin"] = new_df["day"].apply(feat_sin, max_value=365)
    new_df["day_cos"] = new_df["day"].apply(feat_cos, max_value=365)

    # Create month feature
    new_df["month_sin"] = new_df["month"].apply(feat_sin, max_value=12)
    new_df["month_cos"] = new_df["month"].apply(feat_cos, max_value=12)
    # Decompose wind direction
    new_df["winddirection_sin"] = new_df["winddirection"].apply(feat_sin, max_value=360)
    new_df["winddirection_cos"] = new_df["winddirection"].apply(feat_cos, max_value=360)

    # Drop the original cyclic columns
    new_df.drop(columns=["day", "month", "winddirection"], inplace=True)

    # # Add polynomial features
    poly_cols = [
        "cloud",
        "humidity",
        "sunshine",
        "pressure",
    ]
    poly_df = poly_features(new_df, poly_cols)

    # Add rolling features
    # roll_df = rolling_features(new_df)

    # Add difference features
    diff_df = diff_features(new_df)
    # Concatenate all features
    final_df = pd.concat([new_df, poly_df, diff_df], axis=1)
    # Treat outliers
    final_df = treat_outliers_iqr(final_df)
    # Drop duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    return final_df


def calculate_feature_importance(X, y, X_test, verbose=True):
    """
    Calculate feature importance using ExtraTreesClassifier,
    plot the results, and return a pruned table of features.
    """
    selector = ExtraTreesClassifier(n_estimators=100, random_state=SEED)
    selector.fit(X, y)

    # Sort features by importance
    importances = selector.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]  # Descending order
    sorted_importances = importances[sorted_idx]
    sorted_features = X.columns[sorted_idx]

    # Compute cumulative importance
    cumulative_importance = np.cumsum(sorted_importances)

    # Find the elbow (optimal number of features)
    knee_locator = KneeLocator(
        range(len(cumulative_importance)),
        cumulative_importance,
        curve="concave",
        direction="increasing",
    )
    elbow_idx = (
        knee_locator.knee if knee_locator.knee else len(cumulative_importance) - 1
    )

    # Plot feature importance and cumulative importance
    if verbose:
        fig, ax1 = plt.subplots(figsize=(24, 10))

        sns.barplot(
            x=np.arange(len(sorted_importances)),
            y=sorted_importances,
            ax=ax1,
            color="royalblue",
        )
        ax1.set_ylabel("Feature Importance", fontsize=14)
        ax1.set_xlabel("Features (sorted by importance)", fontsize=14)
        ax1.tick_params(axis="y", labelcolor="royalblue")
        ax1.set_xticks(np.arange(len(sorted_features)))
        ax1.set_xticklabels(sorted_features, rotation=90, fontsize=10)
        ax1.spines["top"].set_visible(False)

        # Second y-axis for cumulative importance
        ax2 = ax1.twinx()
        ax2.plot(
            np.arange(len(cumulative_importance)),
            cumulative_importance,
            color="crimson",
            marker="o",
            label="Cumulative Importance",
        )
        # make the font of the second y-axis crimson
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.set_ylabel("Cumulative Importance", fontsize=14, color="crimson")
        ax2.axvline(
            x=elbow_idx,
            color="black",
            linestyle="--",
            label=f"Elbow at {elbow_idx} Features",
        )
        ax2.legend(loc="lower right")
        ax2.spines["top"].set_visible(False)

        plt.title("Feature Importance & Elbow Selection", fontsize=16)
        plt.show()

    # Select only the top features up to the elbow index
    selected_features = X.columns[sorted_idx][:elbow_idx]
    return X[selected_features], X_test[selected_features]
