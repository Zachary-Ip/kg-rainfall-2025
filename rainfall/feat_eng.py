import math
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from kneed import KneeLocator
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
from rainfall.constants import SEED

from sklearn.cluster import KMeans


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

    # Add lag features
    # lag_df = lag_features(new_df)

    # Add difference features
    diff_df = diff_features(new_df)
    # Concatenate all features
    final_df = pd.concat([new_df, poly_df, diff_df], axis=1)
    final_df.drop(columns=poly_cols, inplace=True)

    # replace NaN values with 0
    final_df.fillna(0, inplace=True)
    # print any columns that contain missing values
    missing_cols = final_df.columns[final_df.isnull().any()].tolist()
    if missing_cols:
        print("Missing values in columns:")
        print(missing_cols)
    # Add cluster based features
    final_df = cluster_based_features(final_df)
    # Treat outliers
    final_df = treat_outliers_iqr(final_df)
    # Drop duplicate columns
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    return final_df


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


def lag_features(df, lag_days=[1, 2, 3]):
    """
    Generate lag features for all numeric columns.
    """
    df_lag = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        if "day" not in col and "month" not in col:
            series_col = df[col]
            for lag in lag_days:
                lagged_series = series_col.shift(lag).fillna(0)
                if lagged_series.ndim > 1:
                    lagged_series = lagged_series[:, 0]
                df_lag[f"{col}_lag_{lag}"] = lagged_series
    return df_lag


def interaction_features(df):
    """
    Create interaction features for the dataframe.
    """

    df["sunshine_cloud_ratio"] = df["sunshine"] / (df["cloud"] + 1e-3)
    df["humidity_pressure_ratio"] = df["humidity"] / (df["pressure"] + 1e-3)
    df["min_temp_pressure_ratio"] = df["mintemp"] / (df["pressure"] + 1e-3)
    df["cloud_humidity_mult"] = df["cloud"] * df["humidity"]
    df["cloud_wind_mult"] = df["cloud"] * df["winddirection"]

    if "pressure" in df.columns and "deg_below_dew" in df.columns:
        df["dewpoint_pressure_ratio"] = df["deg_below_dew"] / (df["pressure"] + 1e-3)

    if "cloud" in df.columns and "deg_below_dew" in df.columns:
        df["cloud_dew_mult"] = df["cloud"] * df["deg_below_dew"]

    if (
        "temp_change" in df.columns
        and "humidity" in df.columns
        and "pressure" in df.columns
    ):
        df["cold_front"] = -df["temp_change"] * df["humidity"] / df["pressure"]

    df["positives"] = df["humidity"] * df["cloud"] * df["dewpoint"]
    df["negatives"] = (df["cloud"] - df["sunshine"]) - df["temparature"]
    df["wet_bulb_temp"] = calc_wet_bulb(df["temparature"], df["humidity"])

    df["e_s_temp"] = calc_saturation_vapor_pressure(df["temparature"])
    df["e_s_dewpoint"] = calc_saturation_vapor_pressure(df["dewpoint"])

    # vapor pressure deficit
    df["vapor_pressure_deficit"] = df["e_s_temp"] - df["e_s_dewpoint"]
    # wet-bulb temperature
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


def cluster_based_features(df):
    """
    Creat cluster based features using KMeans clustering.
    """

    # Drop non-numeric columns for clustering
    numeric_df = df.select_dtypes(include=[np.number])

    # Find optimal number of clusters using the elbow method
    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=SEED)
        kmeans.fit(numeric_df)
        inertia.append(kmeans.inertia_)
    kneelocator = KneeLocator(
        range(1, 11), inertia, curve="convex", direction="decreasing"
    )

    optimal_k = kneelocator.elbow
    if optimal_k is None:
        optimal_k = 3
        print("No optimum found, using 3")
    else:
        print("Optimal number of clusters:", optimal_k)
    # Fit KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED)
    cluster_labels = kmeans.fit_predict(numeric_df)

    # use target encoding to create cluster based features
    if "sunshine_cloud_ratio" in df.columns:
        label = "sunshine_cloud_ratio"
    elif "cloud" in df.columns:
        label = "cloud"
    else:
        label = "dewpoint"
    # Calculate the mean of the target variable for each cluster
    cluster_means = df.groupby(cluster_labels)[label].mean()
    cluster_means = cluster_means.reindex(cluster_labels).values
    df["cluster"] = cluster_means

    return df


def calc_wet_bulb(T, RH):
    return (
        T * np.arctan(0.151977 * np.sqrt(RH + 8.313659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676331)
        + 0.00391838 * RH ** (3 / 2) * np.arctan(0.023101 * RH)
        - 4.686035
    )

    # saturated vapor pressure


def calc_saturation_vapor_pressure(temp):
    return 6.11 * np.exp((17.27 * temp) / (temp + 237.3))
