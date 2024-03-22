from src.data import make_dataset
from src.data import pre_processing
from src.visualization import visualize
from src.features import build_features
from src.models import train_model, predict_model
from sklearn.linear_model import LinearRegression, RidgeCV, MultiTaskLassoCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from hyperopt import hp, fmin, tpe, Trials
from joblib import Parallel, delayed
import json


model_names = [
    "LinearRegression",
    "RidgeCV",
    "MultiTaskLassoCV",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "MLP",
    "KNN",
    "DecisionTree"
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

    # for col in data.columns:
    #     visualize.plot_line(data, col)
    #     visualize.plot_autocorrelation(data, col)
    #     visualize.plot_partial_autocorrelation(data, col)

    data = build_features.remove_sparse_columns(data, threshold=0.3)
    sub_data = build_features.data_slicing(data, data.columns)

    # for col in sub_data.columns:
    #     dataset = build_features.mark_outliers_chauvenet(sub_data, col)
    #     visualize.plot_binary_outliers(dataset, col, col+"_outlier", reset_index=True)
    #     visualize.plot_data_distribution(sub_data, col)

    return sub_data


def preprocess_and_feature_engineer(building_id, sub_data):
    """Preprocess and perform feature engineering on building data."""
    data_lags = build_features.create_lags_features(sub_data)
    data_lags = build_features.check_nan_values(data_lags, remove=True)
    data_imputed = build_features.fill_missing_values(data_lags)
    build_features.save_processed_data(data_imputed, building_id)


def hyperparameter_tuning(building_id, model_name, X_train, y_train):
    """
    Perform hyperparameter tuning and save the best parameters for a model.
    """
    space = train_model.define_hyperparameters_space(model_name)
    
    if not space:
        print(f"No hyperparameters to tune for {model_name}. Using default parameters.")
        best_params = {}
    else:
        trials = Trials()
        best = fmin(fn=lambda params: train_model.objective_function(params, model_name, X_train, y_train),
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)
                    
        best_params = {key: int(value) if isinstance(value, float) and value.is_integer() else value for key, value in best.items()}
        if "weights" in best_params:
            best_params['weights'] = "uniform" if best_params['weights'] == 0 else "distance"

    predict_model.save_best_params(building_id, model_name, best_params)
    print(f"Best parameters for {model_name} saved.")


def train_and_evaluate_with_hyperopt(building_id, model_name, tuning=False):
    """
    Train a model using the best parameters stored in a JSON file.
    """
    data = train_model.load_processed_data(building_id)
    train, test = train_model.split_data(data)

    target_columns = [
        col for col in data.columns if any(col.endswith(word) for word in ["Air_temperature", "Relative_humidity"])
    ]
    
    feature_columns = [col for col in data.columns if col not in target_columns]

    X_train, y_train = train_model.get_input_output_features(train, target_columns)
    X_test, y_test = train_model.get_input_output_features(test, target_columns)
    
    if tuning:
        hyperparameter_tuning(building_id, model_name, X_train, y_train)

    params = f"{SAVE_MODELS}models_best_params/{building_id}_best_params.json"
    with open(params, "r") as json_file:
        best_params = json.load(json_file).get(model_name, {})
    
    if not best_params:
        print(f"No best parameters found for {model_name}.")

    trained_model, metrics = train_model.train_models[model_name](X_train, y_train, X_test, y_test, best_params)
    predict_model.save_models(building_id, model_name, trained_model)
    predict_model.save_models_results(building_id, metrics)
    
    for target, metric in metrics[model_name].items():
        print(f"Metrics for {target} in {model_name}:\n", metric, "\n")
        
    if model_name not in ["KNN", "MLP"]:
        feat_df = predict_model.get_feature_importance(trained_model, feature_columns, target_columns)

def process_all_buildings(building_id, features, process=False):
    """Process all buildings."""
    print("#"*20, building_id)
    if process:
        process_building_data(building_id)
        sub_data = visualize_building_data(building_id, features)
        preprocess_and_feature_engineer(building_id, sub_data)
        
    Parallel(n_jobs=-1)(delayed(train_and_evaluate_with_hyperopt)(building_id, model, tuning=True) for model in model_names)


# List of building IDs to process
building_ids = [
    "Building01",
    "Building03",
    "Building04",
    "Building06",
    "Building07",
    "Building08",
    "Building09",
    "Building10",
    "Building11",
    "Building12",
    "Building13",
    "Building14",
    "Building15",
    "Building16",
    "Building17",
    "Building18",
    "Building19",
    "Building20",
]


# Key features and model details
features = [
    "Air_temperature",
    "Relative_humidity",
    "Surface_temperature",
    "Gas_volume",
    "Outside_air_temperature",
    "Outside_relative_humidity",
    "Wind speed",
    "Total rainfall",
]

# Process all buildings
Parallel(n_jobs=-1)(delayed(process_all_buildings)(building_id, features, process=False) for building_id in building_ids[:1])
