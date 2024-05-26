import pandas as pd
import numpy as np
import math
import scipy
from src.features.target_transformations import AutoStationaryTransformer

INTERIM_DATA_PATH = "data/interim/"
PROCESSED_DATA_PATH = "data/processed/"


def load_preprocess_data(building_id: str, key_features: list) -> pd.DataFrame:
    """
    Load and filter preprocessed data based on key features.

    :param building_id: ID of the building.
    :param key_features: List of features to filter in the dataset.
    :return: Filtered DataFrame.
    """
    data = pd.read_pickle(f"{INTERIM_DATA_PATH}{building_id}_Preprocessed.pkl")
    feature_columns = [
        col for col in data.columns if any(word in col for word in key_features)
    ]
    return data[feature_columns]


def remove_sparse_columns(data: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Remove columns with a high proportion of missing values based on a threshold.

    :param data: DataFrame from which to remove columns.
    :param threshold: Threshold for the proportion of missing values.
    :return: DataFrame with sparse columns removed.
    """
    print(f"Before removal, dataset shape: {data.shape}")
    valid_columns = data.columns[data.isna().mean() < threshold]
    removed_columns = data.columns[data.isna().mean() >= threshold]

    if len(removed_columns) > 0:
        print(
            f"Removing columns with >={threshold*100}% missing values: {list(removed_columns)}"
        )
    else:
        print("No columns meet the threshold for removal.")

    return data[valid_columns]


def data_slicing(data: pd.DataFrame, key_columns: list) -> pd.DataFrame:
    """
    Slice data dynamically based on the availability of key features.

    :param data: DataFrame to be sliced.
    :param key_columns: List of key columns for determining the slicing window.
    :return: Sliced DataFrame.
    """
    valid_start = max(data[col].dropna().index.min() for col in key_columns)
    valid_end = min(data[col].dropna().index.max() for col in key_columns)

    print(
        f"Slicing data from index {valid_start} to {valid_end} based on key columns: {key_columns}"
    )

    sliced_data = data[(data.index >= valid_start) & (data.index <= valid_end)]

    print(f"Before slicing, dataset shape: {data.shape}")
    print(f"After slicing, dataset shape: {sliced_data.shape}")

    return sliced_data


def mark_outliers_chauvenet(dataset, col, C=2):
    """
    Mark outliers in a dataset using Chauvenet's criterion.

    :param dataset: DataFrame containing the data.
    :param col: Column to check for outliers.
    :param C: Chauvenet's criterion constant.
    :return: DataFrame with an added outlier column.
    """
    dataset = dataset.copy()
    mean, std = dataset[col].mean(), dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    for i in range(0, len(dataset.index)):
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


def remove_outliers(sub_data, col):
    """
    Remove detected outliers from the dataset for a specified column.

    :param dataset: DataFrame containing the data.
    :param col: Column from which to remove outliers.
    :return: DataFrame with outliers removed.
    """
    dataset = mark_outliers_chauvenet(sub_data, col)
    dataset.loc[dataset[col + "_outlier"], col] = np.nan
    
    sub_data.loc[:, col] = dataset[col]
    n_outliers = dataset[col + "_outlier"].sum()
    print(f"Removed {n_outliers} outliers from {col}")

    return sub_data


def check_nan_values(dataset, remove=False):
    """
    Check for NaN values in the dataset, optionally remove them.

    :param dataset: DataFrame to check for NaN values.
    :param remove: Flag to determine whether to remove NaN values.
    :return: DataFrame after handling NaN values.
    """
    print(f"Size of dataset with missing values {dataset.shape}")
    print(f"Number of missing values per column: {dataset.isna().sum()}")
    if remove:
        dataset.dropna(inplace=True)
        print(f"Size of dataset after removing nan values {dataset.shape}")

    return dataset


def create_lags_features(dataset):
    """
    Create lagged features for time series columns.

    :param dataset: DataFrame containing the data.
    :return: DataFrame with lagged features added.
    """
    for col in dataset.columns:
        if (col.endswith("Air_temperature")) or (col.endswith("Relative_humidity")):
            col_dict = {
                f"{col}_{l}": dataset[col].shift(l).astype("float32") for l in [1, 2]
            }
            dataset = dataset.assign(**col_dict)
            added_features = list(col_dict.keys())
            print(f"Added Lag Features {added_features}")
            
    return dataset


def add_rolling_features(dataset):
    """
    Adds rolling window features to the DataFrame without checking for multiple time series IDs.

    :param dataset: DataFrame containing the data.
    return: Updated DataFrame with rolling window features added.
    """
    rolls = [12, 48]
    agg_funcs = ["mean", "std"]
    
    for col in dataset.columns:
        if (col.endswith("Air_temperature")) or (col.endswith("Relative_humidity")):
            rolling_df = pd.concat(
                [
                    dataset[col]
                    .shift(1)
                    .rolling(window=l)
                    .agg({f"{col}_rolling_{l}_{agg}": agg for agg in agg_funcs})
                    for l in rolls
                ],
                axis=1,
            )
            
            dataset = dataset.assign(**rolling_df.to_dict("list"))
            added_features = rolling_df.columns.tolist()
            print(f"Added Rolling Features {added_features}")

    return dataset


def make_series_stationary(df, columns, seasonal_period=48*7, freq="30min"):
    """
    Applies AutoStationaryTransformer to specified columns of a DataFrame to make the series stationary.

    Parameters:
    - df : pd.DataFrame - The DataFrame containing the time series data.
    - columns : list - List of column names to be made stationary.
    - seasonal_period : int - Seasonal period to be considered by the AutoStationaryTransformer.
    - freq : str - Frequency of the time series data, passed to the transformer.

    Returns:
    - pd.DataFrame - DataFrame with the specified columns transformed to be stationary.
    - dict - Dictionary of transformers for each column.
    """
    transformers = {}
    for col in columns:
        auto_stationary = AutoStationaryTransformer(seasonal_period=seasonal_period)
        y = df[col]
        
        y_stat = auto_stationary.fit_transform(y, freq=freq)
        df.loc[:, col] = y_stat.values
        
        # Store the transformer
        transformers[col] = auto_stationary

    return df, transformers


def fill_missing_values(dataset, method="bfill"):
    """
    Fill missing values in the dataset.

    :param dataset: DataFrame to fill missing values in.
    :param method: Method to use for filling missing values ('bfill' or 'ffill').
    :return: DataFrame with missing values filled.
    """
    dataset = dataset.fillna(method=method)
    return dataset


def add_time_features(data):
    """
    Adds time-related features to the DataFrame.

    :param data: DataFrame with DateTimeIndex.
    :return: DataFrame with added features.
    """
    data['day'] = data.index.day
    data['month'] = data.index.month
    # data['year'] = data.index.year
    data['hour'] = data.index.hour
    data['weekday'] = data.index.weekday

    return data


def save_processed_data(data, building_id):
    """
    Save the processed DataFrame as a pickle file.

    :param data: DataFrame to be saved.
    :param building_id: ID of the building corresponding to the data.
    """
    filepath = f"{PROCESSED_DATA_PATH}{building_id}_Processed.pkl"
    data.to_pickle(filepath)