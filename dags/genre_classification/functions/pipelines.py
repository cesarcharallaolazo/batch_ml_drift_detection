import mlflow


def mlflow_all_pipeline_run(**context):
    print("init mlflow all pipeline run ..")
    _ = mlflow.run(
        "./dags/genre_classification/ml_pipeline_genre_classfication/mlflow_all_pipeline",
        "main",
        parameters={
            "hydra_options": "-m main.experiment_name=airflow_prod_all_genre_classification "
                             "random_forest_pipeline.random_forest.n_estimators=60,90,150"
        }
    )
    print("finish mlflow run.")


def mlflow_prediction_pipeline_run(**context):
    print("init mlflow prediction pipeline run ..")
    _ = mlflow.run(
        "./dags/genre_classification/ml_pipeline_genre_classfication/mlflow_prediction_pipeline",
        "main",
        # parameters={
        #     "hydra_options": "-m main.experiment_name=airflow_prod_all_genre_classification "
        #                      "random_forest_pipeline.random_forest.n_estimators=60,90,150"
        # }
    )
    print("finish mlflow run.")


def mlflow_retraining_pipeline_run(**context):
    print("init mlflow retraining pipeline run ..")
    _ = mlflow.run(
        "./dags/genre_classification/ml_pipeline_genre_classfication/mlflow_retraining_pipeline",
        "main",
        # parameters={
        #     "hydra_options": "-m main.experiment_name=airflow_prod_all_genre_classification "
        #                      "random_forest_pipeline.random_forest.n_estimators=60,90,150"
        # }
    )
    print("finish mlflow run.")
