import argparse
import logging
import os

import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    logger.info("Collecting Ground Truth Data for Retraining")

    df = pd.DataFrame()

    for file in os.listdir(args.data_collection_path):
        if file.startswith("ground_truth_data") and file.endswith(".csv"):
            _ = pd.read_csv(os.path.join(args.data_collection_path, file))
            df = pd.concat([df, _])

    df.to_csv(args.artifact_name, index=False)

    logger.info("Logging artifact")
    mlflow.log_artifact(args.artifact_name)

    os.remove(args.artifact_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset"
    )

    parser.add_argument(
        "--step", type=str, help="Current Step Name", required=True
    )

    parser.add_argument(
        "--data_collection_path",
        type=str,
        help="Collected data path for retraining",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    with mlflow.start_run() as run:
        go(args)
        mlflow.set_tag("step", args.step)
        mlflow.set_tag("current", "1")
