from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import json
import os

SAVE_RESULTS = "reports/predictions/"


def make_prediction(model, X_test, y_test):
    """
    Generate predictions using the provided model and test features.

    :param model: Trained machine learning model.
    :param X_test: Test data features.
    :param y_test: Test target values.
    :return: DataFrame containing the predictions.
    """
    y_pred = model.predict(X_test)
    predictions_df = pd.DataFrame(
        y_pred, index=y_test.index, columns=[f"{col}_pred" for col in y_test.columns]
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


def save_predictions(building_id, model_name, predictions):
    """
    Save the model predictions to a CSV file.

    :param predictions: DataFrame containing model predictions.
    :param building_id: Identifier for the building.
    :param file_path: Path to save the predictions file.
    """
    file_name = f"{SAVE_RESULTS}{building_id}_{model_name}_predictions.csv"
    predictions.to_csv(file_name)


def save_metrics(building_id, model_name, metrics):
    """
    Save the model results to a json file.

    :param predictions: DataFrame containing model predictions.
    :param building_id: Identifier for the building.
    :param file_path: Path to save the metrics dict.
    """
    file_name = f"{SAVE_RESULTS}{building_id}_{model_name}_metrics.json"
    with open(file_name, "w") as json_file:
        json.dump(metrics, json_file, indent=4)


def save_models_results(building_id, metrics):
    """
    Save models results in a CSV file for a specific building.
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
            # df = df.append(pd.Series(metrics, name=model_name))
            print(model_name, value)
            df1 = pd.DataFrame(value, index=[model_name])
            print(df)
            print(df1)
            df = pd.concat([df, df1])

    df.to_csv(results)