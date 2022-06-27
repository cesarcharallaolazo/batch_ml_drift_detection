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

    if "collect_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "collect_data"),
            "main",
            parameters={
                "step": "collect_data",
                "data_collection_path": "./../ground_truth_data",
                "artifact_name": "preprocessed_data.csv",
                "artifact_description": "Data collected for retraining"
            },
            experiment_name=config["main"]["experiment_name"],
            run_name="collect_data"
        )

    if "check_data" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "check_data"),
            "main",
            parameters={
                "step": "check_data",
                "input_step": "collect_data",
                "reference_artifact": "./../reference_data/preprocessed_data.csv",
                "sample_artifact": "preprocessed_data.csv",
                "ks_alpha": config["data"]["ks_alpha"]
            },
            experiment_name=config["main"]["experiment_name"],
            run_name="check_data"
        )

    if "segregate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "segregate"),
            "main",
            parameters={
                "step": "segregate",
                "input_step": "collect_data",
                "input_artifact": "preprocessed_data.csv",
                "artifact_root": "data",
                "test_size": config["data"]["test_size"],
                "stratify": config["data"]["stratify"]
            },
            experiment_name=config["main"]["experiment_name"],
            run_name="segregate"
        )

    if "random_forest" in steps_to_execute:
        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        _ = mlflow.run(
            os.path.join(root_path, "random_forest"),
            "main",
            parameters={
                "step": "random_forest",
                "input_step": "segregate",
                "train_data": "data/data_train.csv",
                "model_config": model_config,
                "export_artifact": config["random_forest_pipeline"]["export_artifact"],
                "random_seed": config["main"]["random_seed"],
                "val_size": config["data"]["val_size"],
                "stratify": config["data"]["stratify"]
            },
            experiment_name=config["main"]["experiment_name"],
            run_name="random_forest"
        )

    if "evaluate" in steps_to_execute:
        _ = mlflow.run(
            os.path.join(root_path, "evaluate"),
            "main",
            parameters={
                "step": "evaluate",
                "input_model_step": "random_forest",
                "model_export": f"{config['random_forest_pipeline']['export_artifact']}",
                "input_data_step": "segregate",
                "test_data": "data/data_test.csv"
            },
            experiment_name=config["main"]["experiment_name"],
            run_name="evaluate"
        )


if __name__ == "__main__":
    go()
