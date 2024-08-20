import numpy as np
import pandas as pd
import tensorflow as tf
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope
from joblib import dump
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor

from src.models import predict_model
from src.visualization import visualize

# tf.config.set_visible_devices([], 'GPU')

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


def split_data(data: pd.DataFrame, test_months: int = 2):
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


def standard_scale(train, test=None):
    """
    Scale features using standard scaler.

    :param X_train: Training input features.
    :param X_test: Testing input features.
    :return: Scaled training and testing features.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    if test is not None:
        test_scaled = scaler.transform(test)
        return train_scaled, test_scaled, scaler
    
    return train_scaled, scaler


def min_max_scale_features(train, test=None):
    """
    Scale features using MinMax scaler.

    :param X_train: Training input features.
    :param X_test: Testing input features.
    :return: Scaled training and testing features.
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    if test is not None:
        test_scaled = scaler.transform(test)
        return train_scaled, test_scaled, scaler
    
    return train_scaled, scaler


def train_linear_regression(X, y, params=None, tuning=False):
    model_name = "LinearRegression"
    
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train, X_test, scaler_X = min_max_scale_features(X_train, X_test)
        y_train, y_test, scaler_y = min_max_scale_features(y_train, y_test)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_ridge_cv(X, y, params=None, tuning=False):
    model_name = "RidgeCV"
    
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train, X_test, scaler_X = min_max_scale_features(X_train, X_test)
        y_train, y_test, scaler_y = min_max_scale_features(y_train, y_test)

        model = RidgeCV(alphas=params.get("alphas", [1.0])) if params else RidgeCV()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_lasso_cv(X, y, params=None, tuning=False):
    model_name = "MultiTaskLassoCV"
    
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train, X_test, scaler_X = min_max_scale_features(X_train, X_test)
        y_train, y_test, scaler_y = min_max_scale_features(y_train, y_test)

        model = MultiTaskLassoCV(alphas=params.get("alphas", [1.0])) if params else MultiTaskLassoCV()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_random_forest(X, y, params=None, tuning=False):
    model_name = "RandomForest"
    
    tscv = TimeSeriesSplit(n_splits=5)
    params = params if params else {}
    params["n_jobs"] = params.get("n_jobs", -1)

    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_extra_trees(X, y, params=None, tuning=False):
    model_name = "ExtraTrees"
    
    tscv = TimeSeriesSplit(n_splits=5)
    params = params if params else {}
    params["n_jobs"] = params.get("n_jobs", -1)

    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = ExtraTreesRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_xgboost(X, y, params=None, tuning=False):
    model_name = "XGBoost"
    
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBRegressor(**params) if params else XGBRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_mlp(X, y, params=None, tuning=False):
    model_name = "MLP"
    
    tscv = TimeSeriesSplit(n_splits=5)
    if not params:
        params = {
            "hidden_layer_sizes": (100,),
            "activation": "relu",
            "batch_size": 32,
            "dropout_rate": 0.5,   
            "optimizer": "adam",           
            "epochs": 200
        }
        
    num_outputs = y.shape[1]
    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        X_train, X_test, scaler_X = min_max_scale_features(X_train, X_test)
        y_train, y_test, scaler_y = min_max_scale_features(y_train, y_test)
        
        model = Sequential()
        for layer_size in params["hidden_layer_sizes"]:
            model.add(Dense(layer_size, activation=params["activation"]))
            model.add(Dropout(params["dropout_rate"]))
        model.add(Dense(num_outputs))
        model.compile(optimizer=params["optimizer"], loss="mean_squared_error")

        model.fit(X_train, y_train, batch_size=params.get("batch_size"), epochs=params["epochs"], verbose=0)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])
    
    return metrics_score


def train_knn(X, y, params=None, tuning=False):
    model_name = "KNN"
    
    tscv = TimeSeriesSplit(n_splits=5)
    params = params if params else {}
    params["n_jobs"] = params.get("n_jobs", -1)

    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = KNeighborsRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def train_decision_tree(X, y, params=None, tuning=False):
    model_name = "DecisionTree"
    
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_score = {model_name:
        {
            "MAE": [],
            "MSE": [],
            "RMSE": [],
            "R2 Score": [],
        }}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = DecisionTreeRegressor(**params) if params else DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    return metrics_score


def objective_function(params, model_name, X, y):
    """
    Objective function for Hyperopt to minimize. Returns the cross-validated score of the given model and parameters.
    """
    score = train_models[model_name](X, y, params, tuning=True)
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
            "dropout_rate": hp.choice("dropout_rate", [0.05, 0.75]),
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
        
    elif model_name == "LSTM":
        space = {
            "lstm_units": scope.int(hp.quniform("lstm_units", 50, 300, 10)),
            "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
            "optimizer": hp.choice("optimizer", ["adam", "rmsprop"]),
            "batch_size": scope.int(hp.quniform("batch_size", 16, 128, 16)),
            "epochs": scope.int(hp.quniform("epochs", 50, 200, 20)),
        }
        
    elif model_name == "EnhancedLSTM":
        space = {
            "lstm_units": scope.int(hp.quniform("lstm_units", 50, 300, 10)),
            "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.5),
            "batch_size": scope.int(hp.quniform("batch_size", 16, 128, 16)),
            "epochs": scope.int(hp.quniform("epochs", 50, 200, 20)),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.0001), np.log(0.01)),
        }

    return space


import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (LSTM, Bidirectional, Conv1D, Dense,
                                     Dropout, Flatten, Lambda, Multiply,
                                     Permute, TimeDistributed)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def train_lstm_model(X, y, params=None, tuning=False):
    """
    Trains an LSTM model on the provided features and target.
    """
    y_index, y_columns = y.index, y.columns
    model_name = "LSTM"
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    if not params:
        params = {
            'lstm_units': 64,
            'dropout_rate': 0.2,
            'optimizer': 'adam',
            'epochs': 50,
            'batch_size': 32
        }

    metrics_score = {"LSTM": {"MAE": [], "MSE": [], "RMSE": [], "R2 Score": []}}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test, scaler_X = min_max_scale_features(X_train, X_test)
        y_train, y_test, scaler_y = min_max_scale_features(y_train, y_test)

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential([
            LSTM(params['lstm_units'], input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(params['dropout_rate']),
            Dense(y_train.shape[1])
        ])
        model.compile(optimizer=params['optimizer'], loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))

    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])

    X, scaler_X = min_max_scale_features(X)
    y, scaler_y = min_max_scale_features(y)

    X = X.reshape((X.shape[0], 1, X.shape[1]))

    model = Sequential([
        LSTM(params['lstm_units'], input_shape=(X.shape[1], X.shape[2])),
        Dropout(params['dropout_rate']),
        Dense(y.shape[1])
    ])
    model.compile(optimizer=params['optimizer'], loss='mean_squared_error')
    model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)

    predictions = model.predict(X)
    predictions = pd.DataFrame(
        scaler_y.inverse_transform(predictions),
        index=y_index,
        columns=[f"{col}_pred" for col in y_columns]
    )

    return model, metrics_score, predictions


def attention_mechanism(inputs, input_dim):
    a = Dense(input_dim, activation='softmax')(inputs)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    
    return Flatten()(output_attention_mul)


def train_enhanced_lstm_model(X, y, params=None, tuning=False):
    """
    Trains an enhanced LSTM model with Bi-directional, CNN, and Attention layers on the provided features and target using time series cross-validation.
    Then, trains the model on the entire dataset.
    """
    y_index, y_columns = y.index, y.columns
    model_name = "EnhancedLSTM"

    if not params:
        params = {
            'lstm_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 64
        }
        
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_score = {model_name: {"MAE": [], "MSE": [], "RMSE": [], "R2 Score": []}}

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test, scaler_X = min_max_scale_features(X_train, X_test)
        y_train, y_test, scaler_y = min_max_scale_features(y_train, y_test)

        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(params['dropout_rate']),
            Bidirectional(LSTM(params['lstm_units'], return_sequences=True)),
            Bidirectional(LSTM(params['lstm_units'], return_sequences=True)),
            Permute((2, 1)), 
            Dense(X_train.shape[1], activation='softmax'),
            Permute((2, 1)),
            Flatten(),
            Dense(y_train.shape[1])
        ])
        model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
        y_pred = model.predict(X_test)
        
        metrics_score[model_name]["MAE"].append(mean_absolute_error(y_test, y_pred))
        metrics_score[model_name]["MSE"].append(mean_squared_error(y_test, y_pred))
        metrics_score[model_name]["RMSE"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics_score[model_name]["R2 Score"].append(r2_score(y_test, y_pred))
    
    if tuning:
        return np.mean(metrics_score[model_name]["MSE"])
    
    X, scaler_X = min_max_scale_features(X)
    y, scaler_y = min_max_scale_features(y)
    
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    
    model = Sequential([
            Conv1D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=(X.shape[1], X.shape[2])),
            Dropout(params['dropout_rate']),
            Bidirectional(LSTM(params['lstm_units'], return_sequences=True)),
            Bidirectional(LSTM(params['lstm_units'], return_sequences=True)),
            Permute((2, 1)), 
            Dense(X.shape[1], activation='softmax'),
            Permute((2, 1)),
            Flatten(),
            Dense(y.shape[1])
        ])
    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')
    model.fit(X, y, epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    
    predictions = model.predict(X)
    predictions = pd.DataFrame(
        scaler_y.inverse_transform(predictions),
        index=y_index,
        columns=[f"{col}_pred" for col in y_columns]
    )

    return model, metrics_score, predictions


train_models = {
    "LinearRegression": train_linear_regression,
    "RidgeCV": train_ridge_cv,
    "MultiTaskLassoCV": train_lasso_cv,
    "RandomForest": train_random_forest,
    "ExtraTrees": train_extra_trees,
    "XGBoost": train_xgboost,
    "MLP": train_mlp,
    "KNN": train_knn,
    "DecisionTree": train_decision_tree,
    "LSTM": train_lstm_model,
    "EnhancedLSTM": train_enhanced_lstm_model
}
