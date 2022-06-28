from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator

from datetime import datetime, timedelta
import os

from genre_classification.functions import pipelines, drift

default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def drift_branch_execution(**context):
    # print("detect_drift_execute   ")
    drift = context.get("ti").xcom_pull(key="dataset_drift")
    if drift:
        return "run_retraining"
    else:
        return "run_prediction"


mlflow_base_path = os.path.join(os.path.dirname(__file__), 'mlflow_all_pipeline')
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
        python_callable=drift.drift_analysis_execute,
        provide_context=True,
    )

    branching_drift_detection = BranchPythonOperator(
        task_id="detect_drift",
        python_callable=drift_branch_execution,
        provide_context=True,
        do_xcom_push=False,
    )

    retraining = PythonOperator(
        task_id="run_retraining",
        python_callable=pipelines.mlflow_retraining_pipeline_run,
        provide_context=False,
    )

    prediction = PythonOperator(
        task_id="run_prediction",
        python_callable=pipelines.mlflow_prediction_pipeline_run,
        provide_context=True,
        trigger_rule="none_failed"
    )

drift_analysis >> branching_drift_detection >> [retraining, prediction]
retraining >> prediction
