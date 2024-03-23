from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import json
import os
from joblib import dump

SAVE_RESULTS = "reports/predictions/"
SAVE_MODELS = "models/"


def make_prediction(model, X_test, y_test, y_test_index, y_test_columns):
    """
    Generate predictions using the provided model and test features.

    :param model: Trained machine learning model.
    :param X_test: Test data features.
    :param y_test: Test target values.
    :return: DataFrame containing the predictions.
    """
    y_pred = model.predict(X_test)
    predictions_df = pd.DataFrame(
        y_pred, index=y_test_index, columns=[f"{col}_pred" for col in y_test_columns]
    )
    return predictions_df


def calculate_metrics(model_name, y_test, y_pred):
    """
    Calculate and return various performance metrics for the model.

    :param model_name: Name of the model used for predictions.
    :param y_test: Actual target values.
    :param y_pred: Predicted target values.
    :return: Dictionary containing calculated metrics.
    """
    return {
        model_name: {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2 Score": r2_score(y_test, y_pred),
        }
    }


def get_feature_importance(model, feature_columns, target_columns):
    """
    Extract feature importance from the trained model, if available.

    :param model: Trained machine learning model.
    :param X_test: Test data features used for model training.
    :param y_test: Test data targets used for model training.
    :return: DataFrame with feature importances.
    """
    feat_df = pd.DataFrame()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_df = pd.DataFrame(
            importances, columns=["Feature Importance"], index=feature_columns
        )
    elif hasattr(model, "coef_"):
        coef_transposed = model.coef_.T
        feat_df = pd.DataFrame(
            coef_transposed, columns=target_columns, index=feature_columns
        )
    return feat_df


def convert_numpy(obj):
    """
    Recursively convert numpy data types in the object to Python native types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(value) for value in obj]
    else:
        return obj


def save_predictions(building_id, model_name, predictions):
    """
    Save the model predictions to a CSV file.
    """
    file_name = f"{SAVE_RESULTS}{building_id}_{model_name}_predictions.csv"
    predictions.to_csv(file_name)


def save_models_results(building_id, metrics):
    """
    Save model's results in a CSV file for a specific building.
    """
    results = f"{SAVE_RESULTS}{building_id}_ML_Models_Results.csv"

    if os.path.exists(results):
        df = pd.read_csv(results, index_col=0)
    else:
        df = pd.DataFrame()

    for model_name, value in metrics.items():
        if model_name in df.index:
            df.loc[model_name] = value
        else:
            df1 = pd.DataFrame(value, index=[model_name])
            df = pd.concat([df, df1])

    df.to_csv(results)


def save_best_params(building_id, model_name, params):
    """
    Save the model's best params to a json file.
    """
    converted_params = convert_numpy(params)
    file_name = f"{SAVE_MODELS}models_best_params/{building_id}_best_params.json"
    if os.path.exists(file_name):
        with open(file_name, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    existing_data[model_name] = converted_params

    with open(file_name, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)


def save_models(building_id, model_name, model):
    """
    Save the trained model to a file
    """
    file_name = f"{SAVE_MODELS}final_models/{building_id}_{model_name}.joblib"
    dump(model, file_name)
