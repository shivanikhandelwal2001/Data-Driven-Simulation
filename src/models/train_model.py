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
from sklearn.metrics import mean_squared_error
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope

PROCESSED_DATA_PATH = "data/processed/"

# Dictionary mapping model names to their respective classes
MODELS = {
    "LinearRegression": LinearRegression,
    "RidgeCV": RidgeCV,
    "MultiTaskLassoCV": MultiTaskLassoCV,
    "RandomForest": RandomForestRegressor,
    "ExtraTrees": ExtraTreesRegressor,
    "XGBoost": XGBRegressor,
    "MLP": MLPRegressor,
    "KNN": KNeighborsRegressor,
    "DecisionTree": DecisionTreeRegressor,
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
    ml_model = MODELS[model_name]()
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


def time_series_cross_validation(model, X, y, n_splits=5):
    """
    Perform time series cross-validation on a given model.

    :param model_name: Name of the model for cross-validation.
    :param X: Input features for cross-validation.
    :param y: Output features for cross-validation.
    :param n_splits: Number of splits for cross-validation.
    :return: Dictionary containing performance metrics for each split.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse_scores = []

    for train_index, test_index in tscv.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        trained_model = model.fit(X_train, y_train)
        y_pred = trained_model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, y_pred))

    return np.mean(mse_scores)


def objective_function(params, model_name, X_train, y_train):
    """
    Objective function for Hyperopt to minimize. Returns the cross-validated score of the given model and parameters.
    """
    model = MODELS[model_name](**params)
    score = time_series_cross_validation(model, X_train, y_train)
    return {"loss": score, "status": STATUS_OK}


def define_hyperparameters_space(model_name):
    """
    Define the hyperparameter space based on the model
    """
    space = {}
    if model_name == "RandomForest":
        space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 100, 1)),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 50, 1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
        } 

    elif model_name == "RidgeCV":
        space = {
            "alphas": hp.lognormal("alpha", 0.2, 2),
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
        }

    elif model_name == "MultiTaskLassoCV":
        space = {
            "alphas": [hp.lognormal("alpha", 0.2, 2)],
            "fit_intercept": hp.choice("fit_intercept", [True, False]),
        }

    elif model_name == "ExtraTrees":
        space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 100, 1)),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 50, 1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
        }

    elif model_name == "XGBoost":
        space = {
            "n_estimators": scope.int(hp.quniform("n_estimators", 10, 100, 1)),
            "max_depth": scope.int(hp.quniform("max_depth", 1, 100, 1)),
            "learning_rate": hp.loguniform("learning_rate", -7, 0),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
            "min_child_weight": hp.loguniform("min_child_weight", -2, 3),
            "gamma": hp.loguniform("gamma", -10, 10),
            "reg_alpha": hp.loguniform("alpha", -10, 10),
            "reg_lambda": hp.loguniform("lambda", -10, 10),
        }

    elif model_name == "MLP":
        space = {
            "hidden_layer_sizes": hp.choice(
                "hidden_layer_sizes",
                [
                    (64,),
                    (128,),
                    (256,),
                    (512,),
                    (1024,),  # Single layer options
                    (64, 64),
                    (128, 128),
                    (256, 256),  # Two layers options
                    (64, 64, 64),
                    (128, 128, 128),  # Three layers options
                    (64, 64, 64, 64),
                    (128, 128, 128, 128),  # Four layers options
                ],
            ),
            "activation": hp.choice(
                "activation", ["relu", "sigmoid", "tanh", "linear"]
            ),
            "batch_size": hp.choice("batch_size", [28, 64, 128]),
            "optimizer": hp.choice("optimizer", ["Adadelta", "Adam", "Adamax"]),
            "epochs": scope.int(hp.quniform("epochs", 50, 1500, 10)),
            "dropout": hp.choice("dropout", [0.05, 0.75]),
        }

    elif model_name == "KNN":
        space = {
            "n_neighbors": scope.int(hp.quniform("n_neighbors", 1, 50, 1)),
            "weights": hp.choice("weights", ["uniform", "distance"]),
            "p": scope.int(hp.quniform("p", 1, 2, 1)),
        }

    elif model_name == "DecisionTree":
        space = {
            "max_depth": scope.int(hp.quniform("max_depth", 1, 50, 1)),
            "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 20, 1)),
            "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 20, 1)),
        }

    return space
