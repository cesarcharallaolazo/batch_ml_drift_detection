import os
from random import random
from datetime import datetime, timedelta

import pandas as pd

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator

# datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
current_date = (datetime.utcnow() - timedelta(minutes=7)).strftime('%Y%m%d%H%M')
# last_day_year_month = get_last_day_year_month(year_month)

default_args = {
    "start_date": datetime(2020, 1, 1),
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def batch_ground_truth_data(date):
    df = pd.read_csv("./dags/genre_classification/ml_pipeline_genre_classfication/"
                     "mlflow_retraining_pipeline/ground_truth_data/ground_truth_data_20220601.csv")

    # df = df.iloc[:, 2:].reset_index().rename({"index": "id"}, axis=1).drop("genre", axis=1)
    # print(df.head(10))
    # print("current_date -->", date)

    df.to_csv("./dags/genre_classification/ml_pipeline_genre_classfication/"
              f"mlflow_retraining_pipeline/ground_truth_data/ground_truth_data_{date}.csv", index=False)


with DAG(
        dag_id="simulate_ground_truth_data",
        schedule_interval="@once",
        # schedule_interval="* * * * *",
        default_args=default_args,
        catchup=False,
) as dag:
    create_ground_truth_data = PythonOperator(
        task_id="batch_daily_ground_truth_data",
        python_callable=batch_ground_truth_data,
        provide_context=False,
        op_kwargs={"date": current_date}
    )

    dag >> create_ground_truth_data
