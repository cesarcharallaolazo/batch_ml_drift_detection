import argparse
import itertools
import logging
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def use_artifact(args):
    query = f"tag.step='{args.input_data_step}' and tag.current='1'"
    retrieved_run = mlflow.search_runs(experiment_ids=[mlflow.active_run().info.experiment_id],
                                       filter_string=query,
                                       order_by=["attributes.start_time DESC"],
                                       max_results=1)["run_id"][0]
    logger.info("retrieved run: " + retrieved_run)
    local_path = mlflow.tracking.MlflowClient().download_artifacts(retrieved_run, args.test_data)
    logger.info("input_artifact: " + args.test_data + " at " + local_path)
    return local_path


def use_model_artifact(args):
    query = f"tag.step='{args.input_model_step}' and tag.current='1'"
    retrieved_run = mlflow.search_runs(experiment_ids=[mlflow.active_run().info.experiment_id],
                                       filter_string=query,
                                       order_by=["attributes.start_time DESC"],
                                       max_results=1)["run_id"][0]
    logger.info("retrieved run: " + retrieved_run)
    model = mlflow.sklearn.load_model(f"runs:/{retrieved_run}/{args.model_export}")
    logger.info("retrieved model artifact: " + args.model_export)
    return model


def go(args):
    logger.info("Reading data to be predicted")
    df = pd.read_csv(args.data_path, low_memory=False)

    logger.info("Downloading and reading the exported model")
    pipe = use_model_artifact(args)

    used_columns = list(itertools.chain.from_iterable([x[2] for x in pipe['preprocessor'].transformers]))

    # ml predictions
    df["genre"] = pipe.predict(df[used_columns])

    df.to_csv(args.prediction_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--step", type=str, help="Current Step Name", required=True
    )

    parser.add_argument(
        "--input_model_step", type=str, help="Input Model Step Name", required=True
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path from data to be predicted by the model",
        required=True,
    )

    parser.add_argument(
        "--prediction_path",
        type=str,
        help="Data Path of predictions",
        required=True,
    )

    args = parser.parse_args()

    with mlflow.start_run() as run:
        go(args)
        mlflow.set_tag("pipeline", args.step)
