import json
import os
import time

import pandas as pd
from hyperopt import Trials, fmin, tpe
from joblib import Parallel, delayed

from src.data import make_dataset, pre_processing
from src.features import build_features
from src.models import predict_model, train_model
from src.visualization import visualize

model_names = [
    "LinearRegression",
    "RidgeCV",
    "MultiTaskLassoCV",
    "DecisionTree",
    "RandomForest",
    "XGBoost",
    "MLP",
]

SAVE_MODELS = "models/"


def process_building_data(building_id):
    """Process data for a single building."""
    # Extract, transform, and save raw data
    building_data = make_dataset.extract_and_transform_data(building_id)
    make_dataset.save_raw_data(building_data, building_id)

    # Preprocess data and save final dataframe
    data = pre_processing.preprocess_data(building_id)
    spaces = pre_processing.get_building_info(data)
    space_data = pre_processing.get_space_data(data, spaces)

    final_df = pre_processing.create_final_dataframe(data, space_data)
    result_df = pre_processing.merge_external_data(final_df)
    pre_processing.save_preprocessed_data(result_df, building_id)


def visualize_building_data(building_id, features):
    """Visualize data for a single building."""
    data = build_features.load_preprocess_data(building_id, features)

    for col in data.columns:
        visualize.plot_line(data, col, building_id)
        visualize.plot_autocorrelation(data, col, building_id)
        visualize.plot_partial_autocorrelation(data, col, building_id)
        visualize.two_line_plot(data, "Gas_volume", col, building_id)

    data = build_features.remove_sparse_columns(data, threshold=0.2)
    sub_data = build_features.data_slicing(data, data.columns)

    for col in sub_data.columns:
        dataset = build_features.mark_outliers_chauvenet(sub_data, col)
        sub_data = build_features.remove_outliers(sub_data, col)
        visualize.plot_binary_outliers(
            dataset, col, col + "_outlier", building_id, reset_index=True
        )
        visualize.plot_data_distribution(sub_data, col, building_id)

    return sub_data


def preprocess_and_feature_engineer(building_id, sub_data, additional_features):
    """Preprocess and perform feature engineering on building data."""
    data = build_features.create_lags_features(sub_data)
    if additional_features:
        # data = build_features.add_rolling_features(data)
        data = build_features.add_time_features(data)

    data = build_features.check_nan_values(data, remove=True)
    data_imputed = build_features.fill_missing_values(data)

    cols = [col for col in data_imputed.columns if col != "Gas_volume"]
    cols.append("Gas_volume")

    data_imputed = data_imputed.reindex(columns=cols)

    build_features.save_processed_data(data_imputed, building_id)


def hyperparameter_tuning(building_id, model_name, X, y):
    """
    Perform hyperparameter tuning and save the best parameters for a model.
    """
    space = train_model.define_hyperparameters_space(model_name)

    if not space:
        print(f"No hyperparameters to tune for {model_name}. Using default parameters.")
        best_params = {}
    else:
        trials = Trials()
        best = fmin(
            fn=lambda params: train_model.objective_function(params, model_name, X, y),
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
        )

        best_params = {
            key: int(value)
            if isinstance(value, float) and value.is_integer()
            else value
            for key, value in best.items()
        }

        if model_name == "KNN":
            best_params["weights"] = (
                "uniform" if best_params["weights"] == 0 else "distance"
            )
        elif model_name == "LSTM":
            best_params["optimizer"] = (
                "adam" if best_params["optimizer"] == 0 else "rmsprop"
            )
        elif model_name == "MLP":
            mlp_space = {
                "hidden_layer_sizes": [
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
                "activation": ["relu", "sigmoid", "tanh", "linear"],
                "batch_size": [28, 64, 128],
                "optimizer": ["Adadelta", "Adam", "Adamax"],
                "dropout_rate": [0.05, 0.75],
            }

            best_params["hidden_layer_sizes"] = list(
                mlp_space["hidden_layer_sizes"][best_params["hidden_layer_sizes"]]
            )
            best_params["activation"] = mlp_space["activation"][
                best_params["activation"]
            ]
            best_params["batch_size"] = mlp_space["batch_size"][
                best_params["batch_size"]
            ]
            best_params["optimizer"] = mlp_space["optimizer"][best_params["optimizer"]]
            best_params["dropout_rate"] = mlp_space["dropout_rate"][
                best_params["dropout_rate"]
            ]

    predict_model.save_best_params(building_id, model_name, best_params)
    print(f"Best parameters for {model_name} saved.")


def train_and_evaluate_with_hyperopt(building_id, model_name, tuning=False):
    """
    Train a model using the best parameters stored in a JSON file.
    """
    data = train_model.load_processed_data(building_id)

    target_columns = [
        col
        for col in data.columns
        if any(col.endswith(word) for word in ["Air_temperature", "Relative_humidity"])
    ]

    X, y = train_model.get_input_output_features(data, target_columns)

    if tuning:
        hyperparameter_tuning(building_id, model_name, X, y)

    params = f"{SAVE_MODELS}models_best_params/{building_id}_best_params.json"
    best_params = {}
    if os.path.exists(params):
        with open(params, "r") as json_file:
            best_params = json.load(json_file).get(model_name, {})

    if not best_params:
        print(f"No best parameters found for {model_name}.")

    start_time = time.time()
    metrics = train_model.train_models[model_name](X, y, best_params)
    end_time = time.time()
    compute_time = end_time - start_time

    compute_time_path = f"reports/cv_models_metrics/{building_id}_computation_time.csv"

    if os.path.exists(compute_time_path):
        compute_time_df = pd.read_csv(compute_time_path)
    else:
        compute_time_df = pd.DataFrame(columns=["Model", "Computation Time (s)"])

    if model_name in compute_time_df["Model"].values:
        compute_time_df.loc[
            compute_time_df["Model"] == model_name, "Computation Time (s)"
        ] = compute_time
    else:
        compute_time_df.loc[len(compute_time_df)] = [model_name, compute_time]

    compute_time_df.to_csv(compute_time_path, index=False)

    metrics1 = pd.DataFrame.from_dict(metrics[model_name])
    metrics1.to_csv(
        f"reports/cv_models_metrics/{building_id}_{model_name}_metrics.csv", index=False
    )

    for target, metric in metrics[model_name].items():
        print(f"Metrics for {target} in {model_name}:\n", metric, "\n")


def process_all_buildings(building_id, features, process=False):
    """Process all buildings."""
    print("#" * 20, building_id)
    if process:
        process_building_data(building_id)
        sub_data = visualize_building_data(building_id, features)
        preprocess_and_feature_engineer(
            building_id, sub_data, additional_features=False
        )

    Parallel(n_jobs=-1)(
        delayed(train_and_evaluate_with_hyperopt)(building_id, model, tuning=False)
        for model in model_names
    )


# List of building IDs to process
building_ids = [
    "Building01",
    "Building08",
    "Building11",
]


# Key features and model details
features = [
    "Air_temperature",
    "Relative_humidity",
    "Surface_temperature",
    "Gas_volume",
    "Outside_air_temperature",
    "Outside_relative_humidity",
    "Wind_speed",
    "Total_rainfall",
]

# Process all buildings
Parallel(n_jobs=-1)(
    delayed(process_all_buildings)(building_id, features, process=True)
    for building_id in building_ids
)
