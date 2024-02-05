import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FiGURES_PATH = "reports/figures/"

# Define standard colors for plots
COLORS = [
    "blue",
    "r",
    "g",
    "cyan",
    "k",
    "m",
    "y",
    "darkcyan",
    "navy",
    "magenta",
    "purple",
    "peru",
]


# Set standard style for matplotlib plots.

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


def save_figure(fig, file_name):
    """Save the figure to a file."""
    full_path = f"{FiGURES_PATH}{file_name}.png"
    if isinstance(fig, go.Figure):
        fig.write_image(full_path)
    else:
        fig.savefig(full_path)


def plot_pre_processed_raw_data(data, space_data):
    """Plots data for each space and variable from the pre-processed raw data."""
    for space in space_data:
        for var in space_data[space]:
            plt.figure(figsize=(20, 10))
            plot_data = data.loc[:, (slice(None), var, slice(None), slice(None), space)]
            ax = plot_data.plot(color=COLORS)

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [label.split(", ")[0] for label in labels]
            ax.legend(handles, new_labels)

            plt.title(f"{space} {var}")
            plt.xlabel("DateTime")
            plt.ylabel(var)
            plt.show()


def plot_pre_processed_final_data(result_df):
    """Plots final processed data for each column in the DataFrame."""
    for df in result_df.columns:
        plt.figure(figsize=(20, 10))
        plt.plot(result_df.index, result_df[df], color=COLORS[0])

        plt.title(df)
        plt.xlabel("DateTime")
        plt.ylabel(df)
        plt.show()


def plot_line(data, column):
    """Create a line plot for a specific column in the dataset."""
    fig = px.line(data, y=column, title=column)
    fig.update_layout(title_x=0.5)
    fig.show()


def plot_autocorrelation(data, column, lags=40):
    """Plot autocorrelation for a specified column."""
    autocorrelation = sm.tsa.acf(data[column], nlags=lags, fft=False)
    plt.figure(figsize=(12, 5))
    plt.stem(autocorrelation)
    plt.title("Autocorrelation Plot " + column)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.show()


def plot_partial_autocorrelation(data, column, lags=40):
    """Plot partial autocorrelation for a specified column."""
    autocorrelation = sm.tsa.pacf(data[column], nlags=lags)
    plt.figure(figsize=(12, 5))
    plt.stem(autocorrelation)
    plt.title("Partial Autocorrelation Plot " + column)
    plt.xlabel("Lag")
    plt.ylabel("Partial Autocorrelation")
    plt.show()


def two_line_plot_secondary_axis(
    x, y1, y2, y1_name="y1", y2_name="y2", title="", dash_secondary=False
):
    """Create a two-line plot with a secondary y-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=y1, name=y1_name), secondary_y=False)
    fig.add_trace(
        go.Scatter(
            x=x, y=y2, name=y2_name, line=dict(dash="dash") if dash_secondary else None
        ),
        secondary_y=True,
    )
    fig.update_layout(title_text=title)
    fig.show()


def two_line_plot(x, y1, y2, title, y1_name="y1", y2_name="y2"):
    """Create a two-line plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, name=y1_name))
    fig.add_trace(go.Scatter(x=x, y=y2, name=y2_name, line=dict(dash="dash")))
    fig.update_layout(title_text=title)
    return fig


def plot_binary_outliers(dataset, col, outlier_col, reset_index=False):
    """Plot binary outliers in the dataset."""
    dataset = dataset.dropna(subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()
    ax.plot(
        dataset.index[~dataset[outlier_col]], dataset[col][~dataset[outlier_col]], "+"
    )
    ax.plot(
        dataset.index[dataset[outlier_col]], dataset[col][dataset[outlier_col]], "r+"
    )

    plt.legend(["no outlier", "outlier"], loc="upper right")
    plt.xlabel("Samples")
    plt.ylabel(col)
    plt.show()


def plot_data_distribution(dataset, col):
    """Plot the data distribution as a violin plot."""
    sns.violinplot(y=col, data=dataset, inner="box")
    plt.title(f"Distribution of {col}")
    plt.show()


def plot_metrics(metrics, model_name):
    """Plot model metrics of predictions."""
    fig = px.bar(
        x=list(metrics.keys()),
        y=list(metrics.values()),
        title=f"{model_name} Model Metrics",
    )
    fig.show()


def plot_predictions(model_name, y_test, y_pred, col):
    """Plot actual and predicted values graph."""
    fig = px.line()
    fig.add_scatter(x=y_test.index, y=y_test[col], mode="lines", name="Actual")
    fig.add_scatter(
        x=y_test.index, y=y_pred[col + "_pred"], mode="lines", name="Predicted"
    )
    fig.update_layout(
        title=model_name + " " + col, xaxis_title="Date", yaxis_title="Value"
    )
    fig.show()


def feature_importance_plot(model_name, feat_df, col):
    """Plot features importance graph for a target variable."""
    sub_df = feat_df[[col]]
    sub_df["_abs_imp"] = np.abs(sub_df[col])
    sub_df = sub_df.sort_values("_abs_imp", ascending=False).drop(columns="_abs_imp")
    fig = px.bar(
        sub_df.head(15), x=sub_df.index[:15], y=col, title=model_name + " " + col
    )
    fig.show()
