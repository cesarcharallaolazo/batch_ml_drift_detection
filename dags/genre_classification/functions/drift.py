import os
import json

import numpy as np
import pandas as pd

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options.data_drift import DataDriftOptions


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


def create_drift_dashboard(file_date):
    print("path dashboard --> ",
          f"./dags/genre_classification/ml_pipeline_genre_classfication/"
          f"mlflow_prediction_pipeline/predicted_daily_data/{file_date}")

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

    # ***** evidently Dashboard

    # options = DataDriftOptions(
    #     num_target_stattest_func=anderson_stat_test,
    #     confidence=0.99,
    #     nbinsx={'MedInc': 15, 'HouseAge': 25, 'AveRooms': 20})
    # options = DataDriftOptions(
    #     feature_stattest_func="psi"
    # )
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab(), CatTargetDriftTab()])

    # read reference & production datasets
    reference_df = pd.read_csv("./dags/genre_classification/ml_pipeline_genre_classfication/"
                               "mlflow_retraining_pipeline/reference_data/preprocessed_data.csv",
                               low_memory=False)
    production_df = pd.read_csv(f"./dags/genre_classification/ml_pipeline_genre_classfication/"
                                f"mlflow_prediction_pipeline/predicted_daily_data/{file_date}",
                                low_memory=False)

    # # evaluate if there is Drift
    # dataset_drift = _detect_dataset_drift(reference_df[features_target],
    #                                       production_df[features_target],
    #                                       column_mapping=data_columns)

    data_drift_dashboard.calculate(reference_df[features_target],
                                   production_df[features_target],
                                   column_mapping=data_columns)
    dir_path = "reports"
    file_path = f"genre_classification_data_and_target_drift_{file_date}.html"
    data_drift_dashboard.save(os.path.join(dir_path, file_path))


def drift_analysis_execute(file_date, **context):
    # dataset_drift = False
    # dataset_drift = True
    # dataset_drift = not dataset_drift

    drift_psi = compute_psi(file_date)
    drift_csi, csi_drifted_features = compute_csi(file_date, drift_ratio=0.1)

    print("**** psi -->", drift_psi, flush=True)
    print("**** csi -->", drift_csi, flush=True)

    dataset_drift = drift_psi or drift_csi  # custom dataset drift detection logic
    # dataset_drift = not dataset_drift

    if dataset_drift:
        create_drift_dashboard(file_date)

    context["ti"].xcom_push(key="dataset_drift", value=dataset_drift)


def compute_psi(file_date):
    target = "genre"

    print("path psi",
          f"./dags/genre_classification/ml_pipeline_genre_classfication/"
          f"mlflow_prediction_pipeline/predicted_daily_data/{file_date}")

    production_df = pd.read_csv(
        f"./dags/genre_classification/ml_pipeline_genre_classfication/"
        f"mlflow_prediction_pipeline/predicted_daily_data/{file_date}",
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

    print("final_psi -->", final_psi)

    if final_psi >= 0.2:
        drift_detected = True
    elif final_psi >= 0.1:
        drift_detected = False
    else:
        drift_detected = False

    return drift_detected


def compute_csi(file_date, drift_ratio=0.5):
    features = ["acousticness", "danceability", "duration_ms", "energy",
                "instrumentalness", "liveness", "loudness", "speechiness",
                "tempo", "valence", "key", "time_signature"]

    print("path csi",
          f"./dags/genre_classification/ml_pipeline_genre_classfication/"
          f"mlflow_prediction_pipeline/predicted_daily_data/{file_date}")

    current_production_df = pd.read_csv(
        f"./dags/genre_classification/ml_pipeline_genre_classfication/"
        f"mlflow_prediction_pipeline/predicted_daily_data/{file_date}",
        low_memory=False)

    current_reference_df = pd.read_csv(
        "./dags/genre_classification/ml_pipeline_genre_classfication/"
        "mlflow_retraining_pipeline/reference_data/preprocessed_data.csv",
        low_memory=False)

    drifted_features = []
    slight_changed_features = []
    unchanged_features = []

    for feature in features:
        production_df = current_production_df.copy()
        reference_df = current_reference_df.copy()
        try:
            try:
                ref_feature, bins = pd.qcut(reference_df[feature], 10, retbins=True)
            except:
                ref_feature, bins = pd.qcut(reference_df[feature], 3, retbins=True)
        except:
            ref_feature, bins = pd.qcut(reference_df[feature], 2, retbins=True)
        production_df = pd.cut(production_df[feature], bins=bins, include_lowest=True).value_counts().reset_index() \
            .rename({feature: "frequency"}, axis=1) \
            .rename({"index": feature}, axis=1) \
            .sort_values(feature, ascending=False) \
            .reset_index().drop("index", axis=1)
        reference_df = ref_feature.value_counts().reset_index() \
            .rename({feature: "frequency"}, axis=1) \
            .rename({"index": feature}, axis=1) \
            .sort_values(feature, ascending=False) \
            .reset_index().drop("index", axis=1)
        production_df["percent_prod"] = production_df["frequency"] / production_df["frequency"].sum()
        reference_df["percent_ref"] = reference_df["frequency"] / reference_df["frequency"].sum()

        df_csi = pd.concat([production_df[[feature, "percent_prod"]],
                            reference_df[["percent_ref"]]],
                           axis=1)

        df_csi["percent_dif"] = df_csi["percent_prod"] - df_csi["percent_ref"]
        df_csi["percent_log_div"] = np.log(df_csi["percent_prod"] / df_csi["percent_ref"])
        df_csi["csi"] = df_csi["percent_dif"] * df_csi["percent_log_div"]

        final_csi = df_csi["csi"].sum()

        if final_csi >= 0.2:
            drifted_features.append(feature)
        elif final_csi >= 0.1:
            slight_changed_features.append(feature)
        else:
            unchanged_features.append(feature)

    print("drifted_features ->", drifted_features)
    print("slight_changed_features ->", slight_changed_features)
    print("unchanged_features ->", unchanged_features)
    print("current drift ratio -->", len(drifted_features) / len(features))

    drift_detected = True if len(drifted_features) / len(features) >= drift_ratio else False

    return drift_detected, drifted_features
