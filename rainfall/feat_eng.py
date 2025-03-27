import math
import numpy as np
import pandas as pd


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
        df["dewpoint_diff"] = df["mintemp"] - df["dewpoint"]

    if "maxtemp" in df.columns and "mintemp" in df.columns:
        df["temp_range"] = df["maxtemp"] - df["mintemp"]
    if "temparature" in df.columns:
        df["temp_change"] = df["temparature"].diff()
        df["temp_change"] = df["temparature"].diff().fillna(0)
    return df


def rolling_features(df, window_sizes=[7, 14]):
    """
    Generate rolling window statistics (mean and std) for all numeric columns.
    Returns a DataFrame with new rolling features as 1D arrays.
    """
    df_roll = pd.DataFrame(index=df.index)
    for col in df.select_dtypes(include=[np.number]).columns:
        series_col = df[col]
        for window in window_sizes:
            roll_mean = series_col.rolling(window=window, min_periods=1).mean().values
            roll_std = (
                series_col.rolling(window=window, min_periods=1).std().fillna(0).values
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

    if "pressure" in df.columns and "humidity" in df.columns:
        df["humidity_pressure_ratio"] = df["humidity"] / (df["pressure"] + 1e-3)

    if "pressure" in df.columns and "mintemp" in df.columns:
        df["min_temp_pressure_ratio"] = df["mintemp"] / (df["pressure"] + 1e-3)

    if "pressure" in df.columns and "dewpoint_diff" in df.columns:
        df["dewpoint_pressure_ratio"] = df["dewpoint_diff"] / (df["pressure"] + 1e-3)

    if "cloud" in df.columns and "humidity" in df.columns:
        df["cloud_humidity_ratio"] = df["cloud"] * df["humidity"]

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

    # Add rolling features
    new_df = rolling_features(new_df)
    # Treat outliers
    new_df = treat_outliers_iqr(new_df)

    return new_df
