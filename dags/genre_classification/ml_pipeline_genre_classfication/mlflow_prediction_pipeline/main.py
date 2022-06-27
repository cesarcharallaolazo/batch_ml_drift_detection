import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    # Setup the mlflow experiment. All runs will be grouped under this experiment
    if config["main"]["mlflow_tracking_url"] != "null":
        mlflow.set_tracking_uri(config["main"]["mlflow_tracking_url"])

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:

        steps_to_execute = list(config["main"]["execute_steps"])

    if "ml" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "ml"),
            "main",
            parameters={
                "step": "prediction",
                "input_model_step": "random_forest",
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}",
                "data_path": "./../data_example/daily_data.csv",
                "prediction_path": "./../data_example/predicted_daily_data.csv"
            },
            experiment_name=config["main"]["experiment_name"],
            run_name="airflow_daily_ml_prediction"
        )


if __name__ == "__main__":
    go()
