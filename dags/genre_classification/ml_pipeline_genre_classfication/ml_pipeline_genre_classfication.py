from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import ShortCircuitOperator
from airflow.operators.python_operator import BranchPythonOperator

from datetime import datetime, timedelta
import os
import json

import mlflow

default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def drift_analysis_execute(**context):
    dataset_drift = False

    context["ti"].xcom_push(key="dataset_drift", value=dataset_drift)


def mlflow_run(**context):
    print("init mlflow run ..")
    _ = mlflow.run(
        "./dags/genre_classification/ml_pipeline_genre_classfication/mlflow_pipeline",
        "main",
        parameters={
            "hydra_options": "-m main.experiment_name=airflow_prod_all_genre_classification "
                             "random_forest_pipeline.random_forest.n_estimators=60,90,150"
        }
    )
    print("finish mlflow run.")


def detect_drift_execute(**context):
    # print("detect_drift_execute   ")
    drift = context.get("ti").xcom_pull(key="dataset_drift")
    print("drift --> ", drift)
    if drift:
        return "_"


def no_detect_drift_execute(**context):
    # print("detect_drift_execute   ")
    drift = context.get("ti").xcom_pull(key="dataset_drift")
    if not drift:
        return "_"


mlflow_base_path = os.path.join(os.path.dirname(__file__), 'mlflow_pipeline')
os.environ["LOGNAME"] = "airflow"

with DAG(
        dag_id="ml_pipeline_genre_classfication",
        schedule_interval="@once",
        default_args=default_args,
        template_searchpath=mlflow_base_path,
        catchup=False,
) as dag:
    drift_analysis = PythonOperator(
        task_id="drift_analysis",
        python_callable=drift_analysis_execute,
        provide_context=True,
    )

    detect_drift = ShortCircuitOperator(
        task_id="detect_drift",
        python_callable=detect_drift_execute,
        provide_context=True,
        do_xcom_push=False,
    )

    no_detect_drift = ShortCircuitOperator(
        task_id="no_detect_drift",
        python_callable=no_detect_drift_execute,
        provide_context=True,
        do_xcom_push=False,
    )

    retraining = PythonOperator(
        task_id="mlflow_run_retraining",
        python_callable=mlflow_run,
        provide_context=False,
    )

    prediction = PythonOperator(
        task_id="mlflow_run_prediction",
        python_callable=mlflow_run,
        provide_context=True
    )

    retraining_prediction = PythonOperator(
        task_id="retraining_mlflow_run_prediction",
        python_callable=mlflow_run,
        provide_context=True
    )

drift_analysis >> [detect_drift, no_detect_drift]
detect_drift >> retraining >> retraining_prediction
no_detect_drift >> prediction
