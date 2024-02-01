import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.models import predict_model

PROCESSED_DATA_PATH = "data/processed/"

# Dictionary mapping model names to their respective classes
MODELS = {
    "LinearRegression": LinearRegression(),
    "RidgeCV": RidgeCV(),
    "MultiTaskLassoCV": MultiTaskLassoCV(),
    "RandomForest": RandomForestRegressor(),
    "ExtraTrees": ExtraTreesRegressor(),
    "XGBoost": XGBRegressor(),
    "MLP": MLPRegressor(),
    "KNN": KNeighborsRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
}


def load_processed_data(building_id: str) -> pd.DataFrame:
    """
    Load processed data for a given building.

    :param building_id: Identifier for the building.
    :return: Loaded DataFrame for the building.
    """
    data = pd.read_pickle(f"{PROCESSED_DATA_PATH}{building_id}_Processed.pkl")
    return data


def split_data(
    data: pd.DataFrame, test_months: int = 2
) -> (pd.DataFrame, pd.DataFrame):
    """
    Split data into training and testing sets based on the specified number of test months.

    :param data: DataFrame containing the data to split.
    :param test_months: Number of months to include in the test set.
    :return: Tuple containing the training and testing DataFrames.
    """
    unique_months = data.index.to_period("M").unique()
    test_mask = data.index.to_period("M").isin(unique_months[-test_months:])
    train = data[~test_mask]
    test = data[test_mask]
    return train, test


def get_input_output_features(data, target_columns):
    """
    Extract input and output features from the dataset.

    :param data: DataFrame containing the data.
    :param target_columns: Columns to be used as output features.
    :return: Tuple of input features (X) and output features (y).
    """
    X, y = data.drop(columns=target_columns), data[target_columns]
    return X, y


def train_model(model_name, X_train, y_train):
    """
    Train a specified machine learning model.

    :param model_name: Name of the model to train.
    :param X_train: Training input features.
    :param y_train: Training output features.
    :return: Trained model.
    """
    ml_model = MODELS[model_name]
    ml_model.fit(X_train, y_train)
    return ml_model


def scale_features(X_train, X_test):
    """
    Scale features using standard scaler.

    :param X_train: Training input features.
    :param X_test: Testing input features.
    :return: Scaled training and testing features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def time_series_cross_validation(model_name, X, y, n_splits=5):
    """
    Perform time series cross-validation on a given model.

    :param model_name: Name of the model for cross-validation.
    :param X: Input features for cross-validation.
    :param y: Output features for cross-validation.
    :param n_splits: Number of splits for cross-validation.
    :return: Dictionary containing performance metrics for each split.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {}

    for i, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trained_model = train_model(model_name, X_train, y_train)
        y_pred = predict_model.make_prediction(trained_model, X_test, y_test)
        metrics = predict_model.calculate_metrics(model_name, y_test, y_pred)

        results[f"Iteration {i}"] = metrics

    return results
