from src.data import make_dataset
from src.data import pre_processing
from src.visualization import visualize
from src.features import build_features
from src.models import train_model, predict_model


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

    data = build_features.remove_sparse_columns(data)
    sub_data = build_features.data_slicing(data, data.columns)

    # for col in sub_data.columns:
    #     dataset = build_features.mark_outliers_chauvenet(sub_data, col)
    #     visualize.plot_binary_outliers(dataset, col, col+"_outlier", reset_index=True)
    #     visualize.plot_data_distribution(sub_data, col)

    return sub_data


def preprocess_and_feature_engineer(building_id, sub_data):
    """Preprocess and perform feature engineering on building data."""
    sub_data = build_features.check_nan_values(sub_data, remove=True)
    data_lags = build_features.create_lags_features(sub_data)
    data_lags = build_features.check_nan_values(data_lags)
    data_imputed = build_features.fill_missing_values(data_lags)
    build_features.save_processed_data(data_imputed, building_id)


def train_and_evaluate(building_id, model_name):
    """Train and evaluate model for a single building."""
    data = train_model.load_processed_data(building_id)
    train, test = train_model.split_data(data)

    target_columns = [
        col
        for col in data.columns
        if any(col.endswith(word) for word in ["Air_temperature", "Relative_humidity"])
    ]

    X_train, y_train = train_model.get_input_output_features(train, target_columns)
    X_test, y_test = train_model.get_input_output_features(test, target_columns)

    if model_name in ["LinearRegression", "RidgeCV", "MultiTaskLassoCV"]:
        X_train1, X_test1 = train_model.scale_features(X_train, X_test)
        trained_model = train_model.train_model(model_name, X_train1, y_train)
        predictions = predict_model.make_prediction(
            trained_model, X_test1, y_test, X_test.index
        )
    else:
        trained_model = train_model.train_model(model_name, X_train, y_train)
        predictions = predict_model.make_prediction(
            trained_model, X_test, y_test, X_test.index
        )

    metrics = predict_model.calculate_metrics(model_name, y_test, predictions)
    predict_model.save_predictions(building_id, model_name, predictions)
    predict_model.save_metrics(building_id, model_name, metrics)

    if model_name not in ["KNN", "MLP"]:
        feat_df = predict_model.get_feature_importance(trained_model, X_test, y_test)

    for target, metric in metrics.items():
        print(f"Metrics for {target} in {model_name}:\n", metric, "\n")

    # for col in y_test.columns:
    #     visualize.plot_predictions(model_name, y_test, predictions, col)


def process_all_buildings(building_ids, features, model_name):
    """Process all buildings."""
    for building_id in building_ids:
        print("#" * 20, building_id)
        process_building_data(building_id)
        sub_data = visualize_building_data(building_id, features)
        preprocess_and_feature_engineer(building_id, sub_data)
        train_and_evaluate(building_id, model_name)


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

model_name = "LinearRegression"
# Process all buildings
process_all_buildings(building_ids, features, model_name)
