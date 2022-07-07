from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator

from datetime import datetime, timedelta
import os

from genre_classification.functions import pipelines, drift

current_date = datetime.utcnow().strftime('%Y%m%d%H%M')
current_date_less1 = (datetime.utcnow() - timedelta(minutes=1)).strftime('%Y%m%d%H%M')

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


def verify_file_existence(file_date, file_date_predicted_less1):
    print(f"verify existence : {file_date}", flush=True)
    print(f"verify existence : {file_date_predicted_less1}", flush=True)
    if os.path.exists(f"./dags/genre_classification/ml_pipeline_genre_classfication/"
                      f"mlflow_prediction_pipeline/daily_data/{file_date}"):
        if os.path.exists(f"./dags/genre_classification/ml_pipeline_genre_classfication/"
                          f"mlflow_prediction_pipeline/predicted_daily_data/{file_date_predicted_less1}"):
            return "drift_analysis"
        else:
            return "run_prediction"
    else:
        return "nothing"


os.environ["LOGNAME"] = "airflow"

with DAG(
        dag_id="ml_pipeline_genre_classfication",
        schedule_interval="@once",
        # schedule_interval="* * * * *",
        default_args=default_args,
        concurrency=1,
        catchup=False,
) as dag:
    dummy = DummyOperator(
        task_id='nothing',
        trigger_rule='all_success'
    )

    branching_file_existence = BranchPythonOperator(
        task_id="branching_file_existence",
        python_callable=verify_file_existence,
        provide_context=False,
        op_kwargs={'file_date': f'daily_data_{current_date}.csv',
                   'file_date_predicted_less1': f'predicted_daily_data_{current_date_less1}.csv'}
    )

    drift_analysis = PythonOperator(
        task_id="drift_analysis",
        python_callable=drift.drift_analysis_execute,
        provide_context=True,
        op_kwargs={'file_date': f'predicted_daily_data_{current_date_less1}.csv'}
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
        op_kwargs={'daily_path': f'./../daily_data/daily_data_{current_date}.csv',
                   'prediction_daily_path': f'./../predicted_daily_data/predicted_daily_data_{current_date}.csv'},
        trigger_rule="none_failed"
    )

    branching_file_existence >> [dummy, drift_analysis, prediction]

    drift_analysis >> branching_drift_detection >> [retraining, prediction]
    retraining >> prediction
