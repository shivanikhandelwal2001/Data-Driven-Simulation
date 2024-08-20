import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
from plotly.subplots import make_subplots

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "lines.linewidth": 3,
    }
)
FIGURES_PATH = "reports/figures/"


# Helper function to ensure directory exists
def ensure_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)


# Function to save figures
def save_figure(fig, building_id, subdirectory, file_name):
    full_path = os.path.join(FIGURES_PATH, building_id, subdirectory)
    ensure_dir(full_path)
    file_path = os.path.join(full_path, f"{file_name}.png")
    fig.tight_layout()

    if isinstance(fig, go.Figure):
        fig.write_image(file_path)
    else:
        fig.savefig(file_path, bbox_inches="tight")

    plt.close(fig)


# Plot line
def plot_line(data, column, building_id):
    fig, ax = plt.subplots(figsize=(20, 10))
    data[column].plot(ax=ax, color="tab:blue", title=f"{column} over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    save_figure(fig, building_id, "line_plots", f"{column}")


# Plot autocorrelation
def plot_autocorrelation(data, column, building_id, lags=40):
    fig, ax = plt.subplots(figsize=(12, 6))
    sm.graphics.tsa.plot_acf(data[column].dropna(), lags=lags, ax=ax)
    ax.set_title(f"Autocorrelation for {column}")
    save_figure(fig, building_id, "autocorrelation_plots", f"autocorr_{column}")


# Plot partial autocorrelation
def plot_partial_autocorrelation(data, column, building_id, lags=40):
    fig, ax = plt.subplots(figsize=(12, 6))
    sm.graphics.tsa.plot_pacf(data[column].dropna(), lags=lags, ax=ax)
    ax.set_title(f"Partial Autocorrelation for {column}")
    save_figure(
        fig, building_id, "partial_autocorrelation_plots", f"part_autocorr_{column}"
    )


# Two line plot
def two_line_plot(data, col1, col2, building_id):
    """Create a two-line plot for features relationship"""
    fig, ax1 = plt.subplots(figsize=(20, 10))

    color1 = "tab:blue"
    color2 = "tab:orange"

    ax1.plot(data.index, data[col1], label=col1, color=color1)
    ax1.set_xlabel("DateTime")
    ax1.set_ylabel(col1, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(data.index, data[col2], label=col2, color=color2, linestyle="--")
    ax2.set_ylabel(col2, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(f"{col1} and {col2}")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save_figure(
        fig, building_id, "two_line_plots", f"Relationship between {col1} {col2}"
    )


# Plot binary outliers
def plot_binary_outliers(dataset, col, outlier_col, building_id, reset_index=False):
    dataset = dataset.dropna(subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")
    if reset_index:
        dataset = dataset.reset_index()
    fig, ax = plt.subplots()
    ax.plot(dataset[col], marker="o", linestyle="", alpha=0.7)
    ax.plot(
        dataset.loc[dataset[outlier_col], col],
        marker="x",
        linestyle="",
        color="r",
        label="Outlier",
    )
    ax.set_title(f"Outliers in {col}")
    ax.legend()
    save_figure(fig, building_id, "binary_outliers_plots", f"{col}_outliers")


# Plot data distribution
def plot_data_distribution(dataset, col, building_id):
    fig, ax = plt.subplots()
    sns.violinplot(data=dataset, y=col, ax=ax)
    ax.set_title(f"Distribution of {col}")
    save_figure(fig, building_id, "data_distribution_plots", f"{col}_distribution")


# Plot metrics
def plot_metrics(metrics, model_name, building_id):
    fig, ax = plt.subplots()
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax)
    ax.set_title(f"Metrics for {model_name}")
    save_figure(fig, building_id, "metrics_plots", f"{model_name}_metrics")


# Plot predictions
def plot_predictions(model_name, y_test, y_pred, col, building_id):
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(y_test.index, y_test[col], label="Actual", color="tab:blue")
    ax.plot(
        y_test.index,
        y_pred[f"{col}_pred"],
        label="Predicted",
        linestyle="--",
        color="tab:orange",
    )
    ax.set_title(f"{model_name} Predictions for {col}")
    ax.legend()
    save_figure(
        fig, building_id, "predictions_plots", f"{model_name}_{col}_predictions"
    )


# Feature importance plot
def feature_importance_plot(model_name, feat_df, col, building_id):
    sub_df = feat_df[[col]].copy()
    sub_df["_abs_imp"] = np.abs(sub_df[col])
    sub_df = sub_df.sort_values("_abs_imp", ascending=False).drop(columns="_abs_imp")
    sub_df = sub_df.head(10)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=sub_df[col], y=sub_df.index, ax=ax, orient="h")
    ax.set_title(f"Feature Importance for {model_name} - {col}")
    save_figure(
        fig, building_id, "feature_importance_plots", f"{model_name}_{col}_importance"
    )


# Plot LSTM History
def plot_training_history(history, building_id):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history.history["loss"], label="Training Loss", color="tab:blue")
    ax.plot(history.history["val_loss"], label="Validation Loss", color="tab:orange")

    ax.set_title(f"Training and Validation Loss for Building {building_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    save_figure(
        fig,
        building_id,
        "training_history_plots",
        f"training_validation_loss_{building_id}",
    )
