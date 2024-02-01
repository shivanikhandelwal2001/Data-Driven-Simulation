import pandas as pd
from src.visualization import visualize

# Constants for file paths
DATA_CSV_PATH = "data/external/REFIT_TIME_SERIES_VALUES.csv"
RAW_DATA_PATH = "data/raw/"
INTERIM_DATA_PATH = "data/interim/"


def preprocess_data(building_id):
    """
    Preprocess data for a given building ID.

    :param building_id: Identifier for the building.
    :return: Preprocessed DataFrame.
    """
    data = pd.read_pickle(f"{RAW_DATA_PATH}{building_id}.pkl")
    data.rename_axis("Date", inplace=True)
    print(f"Data info:{data.info()}")
    return data


def get_building_info(data):
    """
    Extracts and prints building information such as room types and space mapping.

    :param data: DataFrame containing building data.
    :return: List of spaces in the building.
    """
    multi_index_columns = data.columns
    room_types = set(multi_index_columns.get_level_values(5))
    print(f"Total no. and types of rooms - {len(room_types)} {room_types}")

    space_room_dict = {}
    space_room = (
        multi_index_columns.get_level_values(4)
        .str.cat(multi_index_columns.get_level_values(5), sep=" - ")
        .unique()
    )
    for j in space_room:
        if isinstance(j, str):
            space, room = j.split(" - ")
            space_room_dict.setdefault(space, []).append(room)

    print(f"Space wise room mapping - {space_room_dict}")
    spaces = space_room_dict.keys()
    print(f"Total spaces in the building - {len(spaces)}, {spaces}")
    return spaces


def get_space_data(data, spaces):
    """
    Extracts and organizes data for each space in the building.

    :param data: DataFrame containing the data.
    :param spaces: List of spaces in the building.
    :return: Dictionary of data for each space.
    """
    var_data = set()
    space_data = {}
    for i in spaces:
        cols = data.loc[
            :, (slice(None), slice(None), slice(None), slice(None), i)
        ].columns
        var_ids = set(
            cols.get_level_values(1).str.cat(cols.get_level_values(0), sep=" - ")
        )

        sub_var_id = {}
        for var_id in var_ids:
            var, id1 = var_id.split(" - ")
            sub_var_id.setdefault(var, []).append(id1)

        var_data.update(set(cols.get_level_values(1)))
        space_data[i] = sub_var_id
    return space_data


def create_final_dataframe(data, space_data):
    """
    Creates the final dataframe consolidating data from all spaces.

    :param data: DataFrame containing the data.
    :param space_data: Dictionary containing data for each space.
    :return: Final consolidated DataFrame.
    """
    final_df = pd.DataFrame()
    for space in space_data:
        for var in space_data[space]:
            sub_data = data.loc[:, (slice(None), var, slice(None), slice(None), space)]
            df_name = f"{space}_{var.replace(' ', '_')}"
            final_df[df_name] = sub_data.mean(axis=1)

    final_df["Gas_volume"] = data.loc[:, (slice(None), "Gas volume")]

    return final_df


def merge_external_data(final_df):
    """
    Merges external data sources into the final dataframe.

    :param final_df: Final DataFrame to merge external data into.
    :return: DataFrame with external data merged.
    """
    main_data = pd.read_csv(DATA_CSV_PATH)
    main_data["dateTime"] = pd.to_datetime(main_data["dateTime"])

    external_data = {
        "TimeSeriesVariable1573": ["Outside_air_temperature"],
        "TimeSeriesVariable1574": ["Outside_relative_humidity"],
        "TimeSeriesVariable1575": ["Wind_speed"],
        "TimeSeriesVariable1582": ["Total_rainfall"],
    }

    for id1, columns in external_data.items():
        ext_df = main_data[main_data["TimeSeriesVariable/@id"] == id1][
            ["dateTime", "data"]
        ]
        ext_df.columns = ["Date", columns[0]]
        ext_df.set_index("Date", inplace=True)
        final_df = final_df.merge(
            ext_df.resample("30T").mean(),
            left_index=True,
            right_index=True,
            how="outer",
        )

    return final_df


def save_preprocessed_data(data, building_id):
    """
    Saves the preprocessed DataFrame as a pickle file.

    :param data: DataFrame to be saved.
    :param building_id: Identifier for the building corresponding to the data.
    """
    filepath = f"{INTERIM_DATA_PATH}{building_id}_Preprocessed.pkl"
    data.to_pickle(filepath)
