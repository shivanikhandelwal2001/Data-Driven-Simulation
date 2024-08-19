import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
from src.models import train_model, predict_model


building_ids = [
    "Building01",
    "Building08",
    "Building11",
]

model_name = "linear_regression"

test_start_date = '2014-12-01'
test_end_date = '2015-03-31'


def make_multi_step_prediction_horizon(model, X_test, scaler_X, scaler_y, y_test_columns, n_steps=None, lags=2):
    """
    Make multi-step predictions for a given prediction horizon.
    """
    predictions_df = pd.DataFrame(columns=y_test_columns, index=X_test.index)
    lagged_features = {"lags_1": [col for col in X_test.columns if col.endswith("_1")],
                    "lags_2": [col for col in X_test.columns if col.endswith("_2")]}
    
    if n_steps is None:
        n_steps = len(X_test)
    
    for i in range(0, len(X_test), n_steps):
        # print("#####################", i)
        actual_steps = min(n_steps, len(X_test) - i)
        for step in range(actual_steps):
            # print("****************", step)
            for l in range(1, lags + 1):
                if step - l >= 0:
                    X_test.loc[X_test.index[i+step], lagged_features[f"lags_{l}"]] = predictions_df.loc[X_test.index[i+step - l], :].values
                    
            x_scale = scaler_X.transform(X_test.loc[X_test.index[i+step], :].values.reshape(1, -1))
            predictions_df.loc[X_test.index[i+step]] = scaler_y.inverse_transform(model.predict(x_scale))
            
    return predictions_df


def train_final_model(building_id, data):
    """
    Train and evaluate the final Linear Regression model for a building.
    """
    target_columns = [
        col
        for col in data.columns
        if any(col.endswith(word) for word in ["Air_temperature", "Relative_humidity"])
    ]
    
    feature_columns = [col for col in data.columns if col not in target_columns]
    
    X, y = data.drop(columns=target_columns), data[target_columns]
    X_columns, y_columns = X.columns, y.columns

    X_train = X[(X.index < test_start_date) | (X.index > test_end_date)]
    y_train = y[(y.index < test_start_date) | (y.index > test_end_date)]
    X_test = X[(X.index >= test_start_date) & (X.index <= test_end_date)]
    y_test = y[(y.index >= test_start_date) & (y.index <= test_end_date)]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    X_train_scale, X_test_scale, scaler_X = train_model.min_max_scale_features(X_train, X_test)
    y_train_scale, y_test_scale, scaler_y = train_model.min_max_scale_features(y_train, y_test)

    model = LinearRegression()
    model.fit(X_train_scale, y_train_scale)

    y_pred_scale = model.predict(X_test_scale)

    y_pred = pd.DataFrame(
            scaler_y.inverse_transform(y_pred_scale),
            index=y_test.index,
            columns=y.columns,
        )

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_s = r2_score(y_test, y_pred)

    print("One step Prediction:")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2 Score: ", r2_s)
    print("*********************\n")
    
    feat_df = predict_model.get_feature_importance(
            model, feature_columns, target_columns
        )
    
    predict_model.save_models(building_id, model_name, model, X_columns, y_columns, scaler_X, scaler_y, feat_df)
    

def get_predictions(building_id, data):
    """
    Make multi-step predictions and evaluate the model for different horizons.
    """
    target_columns = [
        col
        for col in data.columns
        if any(col.endswith(word) for word in ["Air_temperature", "Relative_humidity"])
    ]
    
    X, y = data.drop(columns=target_columns), data[target_columns]

    X_train = X[(X.index < test_start_date) | (X.index > test_end_date)]
    y_train = y[(y.index < test_start_date) | (y.index > test_end_date)]
    X_test = X[(X.index >= test_start_date) & (X.index <= test_end_date)]
    y_test = y[(y.index >= test_start_date) & (y.index <= test_end_date)]
    
    
    horizons = [1, 2+1, 6+1, 12+1, 18+1, 24+1, 48+1, 96+1]
    
    model, X_columns, y_columns, scaler_X, scaler_y = predict_model.load_saved_model(building_id, model_name)

    metrics_df = pd.DataFrame(columns=["Prediction Horizon", "MAE", "MSE", "RMSE", "R2 Score"])

    for h in horizons:
        y_multi_step = make_multi_step_prediction_horizon(model, X_test.copy(), scaler_X, scaler_y, y_test_columns=y_columns, n_steps=h)

        mae = mean_absolute_error(y_test, y_multi_step)
        mse = mean_squared_error(y_test, y_multi_step)
        rmse = np.sqrt(mean_squared_error(y_test, y_multi_step))
        r2_s = r2_score(y_test, y_multi_step)

        print(f"{int(h/2)} Hour Horizon Prediction:")
        print("MAE: ", mae)
        print("MSE: ", mse)
        print("RMSE: ", rmse)
        print("R2 Score: ", r2_s)
        print("*********************\n")
        
        metrics_df.loc[len(metrics_df)] = [int(h/2), mae, mse, rmse, r2_s]
        y_multi_step.to_csv(f"reports/multistep_predictions/{building_id}_{model_name}_predictions_{int(h/2)}h.csv", index=False)
    
    metrics_df.to_csv(f"reports/multistep_predictions/{building_id}_{model_name}_metrics.csv", index=False)
    

for building_id in building_ids:
    data = train_model.load_processed_data(building_id)
    train_final_model(building_id, data)
    # get_predictions(building_id, data)