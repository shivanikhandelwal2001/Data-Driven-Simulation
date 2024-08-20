import collections

import pandas as pd
from lxml import etree

# Constants for file paths
DATA_CSV_PATH = "data/external/REFIT_TIME_SERIES_VALUES.csv"
BUILDING_SURVEY_XML_PATH = "data/external/REFIT_BUILDING_SURVEY.xml"
RAW_DATA_PATH = "data/raw/"
NS = {"a": "http://www.refitsmarthomes.org"}

# Load external CSV and XML data
data = pd.read_csv(DATA_CSV_PATH)
tree = etree.parse(BUILDING_SURVEY_XML_PATH)


def tsv_info(tsv):
    """
    Extracts detailed information from a time series variable (TSV) element.

    :param tsv: lxml element representing a time series variable.
    :return: Dictionary with details about the TSV.
    """
    sensor = tsv.getparent()
    entity = sensor.getparent()
    info = {
        "timeSeriesVariableId": tsv.get("id"),
        "variableType": tsv.get("variableType"),
        "units": tsv.get("units"),
        "sensorManufacturerModel": f"{sensor.get('manufacturer')} {sensor.get('model')}",
        "entityId": (
            entity.get("spaceIdRef")
            if entity.get("id").startswith("Radiator")
            else entity.get("id")
        ),
        "roomType": (
            entity.get("id")
            if entity.get("id").startswith("Radiator")
            else entity.get("roomType")
        ),
    }
    return info


def extract_and_transform_data(building_id):
    """
    Extracts and transforms data for a given building.

    :param building_id: ID of the building for which data is extracted.
    :return: Transformed DataFrame for the specified building.
    """
    building = tree.xpath(f'.//a:Building[@id="{building_id}"]', namespaces=NS)[0]
    tsvs = building.xpath(".//a:TimeSeriesVariable", namespaces=NS)
    tsv_ids = [tsv.get("id") for tsv in tsvs]

    building_df = data[data["TimeSeriesVariable/@id"].isin(tsv_ids)]
    print(f"No. of rows in {building.get('id')} - {building_df.shape}")

    building_df = building_df.drop_duplicates(
        subset=["TimeSeriesVariable/@id", "dateTime"]
    )
    print(
        f"No. of rows in {building.get('id')} after removing duplicated rows - {building_df.shape}"
    )

    building_df2 = building_df.pivot(
        index="dateTime", columns="TimeSeriesVariable/@id", values="data"
    )
    building_df2.index = pd.to_datetime(building_df2.index)

    tsvs = [
        tree.xpath('.//a:TimeSeriesVariable[@id="%s"]' % tsv_id, namespaces=NS)[0]
        for tsv_id in building_df2.columns
    ]

    d = collections.defaultdict(list)

    for tsv in tsvs:
        for k, v in tsv_info(tsv).items():
            d[k].append(v)

    d = dict(d)

    arrays = list(d.values())
    tuples = list(zip(*arrays))
    names = list(d.keys())

    building_df2.columns = pd.MultiIndex.from_tuples(tuples)
    
    print(
        f"No. of rows in {building.get('id')} after pivoting - {building_df2.shape}"
    )
    print(f"Variables types - {set(d['variableType'])}")

    building_df3 = building_df2.resample("30T").mean()
    print(
        f"No. of rows in {building.get('id')} after resampling by half an hour - {building_df3.shape}"
    )

    return building_df3


def save_raw_data(data, building_id):
    """
    Saves the DataFrame as a pickle file for a given building ID.

    :param data: DataFrame to be saved.
    :param building_id: ID of the building corresponding to the DataFrame.
    """
    filepath = f"{RAW_DATA_PATH}{building_id}.pkl"
    data.to_pickle(filepath)
