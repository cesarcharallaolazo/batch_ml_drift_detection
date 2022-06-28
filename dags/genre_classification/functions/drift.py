import os
import json

import numpy as np
import pandas as pd

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection


def _detect_dataset_drift(reference, production, column_mapping, get_ratio=False):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns ration of drifted features.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    Data Drift for the dataset is detected
        if share of the drifted features is above the selected threshold (default value is 0.5).
    """

    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)

    if get_ratio:
        n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
        n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
        return n_drifted_features / n_features

    else:
        return json_report["data_drift"]["data"]["metrics"]["dataset_drift"]


def drift_analysis_execute(**context):
    features = ["acousticness", "danceability", "duration_ms", "energy",
                "instrumentalness", "liveness", "loudness", "speechiness",
                "tempo", "valence", "key", "time_signature"]
    target = "genre"
    features_target = features + [target]

    # evidently Column Mapping
    data_columns = ColumnMapping()
    data_columns.target = target
    data_columns.numerical_features = features
    data_columns.categorical_features = []

    # evidently Dashboard
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])

    # read reference & production datasets
    reference_df = pd.read_csv("./dags/genre_classification/ml_pipeline_genre_classfication/"
                               "mlflow_retraining_pipeline/reference_data/preprocessed_data.csv",
                               low_memory=False)
    production_df = pd.read_csv("./dags/genre_classification/ml_pipeline_genre_classfication/"
                                "mlflow_prediction_pipeline/predicted_daily_data/predicted_daily_data_20220601.csv",
                                low_memory=False)

    # evaluate if there is Drift
    dataset_drift = _detect_dataset_drift(reference_df[features_target],
                                          production_df[features_target],
                                          column_mapping=data_columns)
    # dataset_drift = False
    # dataset_drift = True
    # dataset_drift = not dataset_drift

    print("**** current drift -->", dataset_drift, "****", flush=True)
    print("**** psi -->", compute_psi(), flush=True)

    if dataset_drift:
        data_drift_dashboard.calculate(reference_df[features_target],
                                       production_df[features_target],
                                       column_mapping=data_columns)
        dir_path = "reports"
        file_path = "genre_classification_data_and_target_drift.html"
        data_drift_dashboard.save(os.path.join(dir_path, file_path))

    context["ti"].xcom_push(key="dataset_drift", value=dataset_drift)


def compute_psi():
    target = "genre"

    production_df = pd.read_csv(
        "./dags/genre_classification/ml_pipeline_genre_classfication/"
        "mlflow_prediction_pipeline/predicted_daily_data/predicted_daily_data_20220601.csv",
        low_memory=False)

    reference_df = pd.read_csv(
        "./dags/genre_classification/ml_pipeline_genre_classfication/"
        "mlflow_retraining_pipeline/reference_data/preprocessed_data.csv",
        low_memory=False)

    production_df = production_df[target].value_counts().reset_index() \
        .rename({target: "frequency"}, axis=1) \
        .rename({"index": target}, axis=1) \
        .sort_values(target) \
        .reset_index().drop("index", axis=1)

    reference_df = reference_df[target].value_counts().reset_index() \
        .rename({target: "frequency"}, axis=1) \
        .rename({"index": target}, axis=1) \
        .sort_values(target) \
        .reset_index().drop("index", axis=1)

    production_df["percent_prod"] = production_df["frequency"] / production_df["frequency"].sum()
    reference_df["percent_ref"] = reference_df["frequency"] / reference_df["frequency"].sum()

    df_psi = pd.concat([production_df[[target, "percent_prod"]],
                        reference_df[["percent_ref"]]],
                       axis=1)
    df_psi["percent_dif"] = df_psi["percent_prod"] - df_psi["percent_ref"]
    df_psi["percent_log_div"] = np.log(df_psi["percent_prod"] / df_psi["percent_ref"])
    df_psi["psi"] = df_psi["percent_dif"] * df_psi["percent_log_div"]

    final_psi = df_psi["psi"].sum()

    return final_psi
