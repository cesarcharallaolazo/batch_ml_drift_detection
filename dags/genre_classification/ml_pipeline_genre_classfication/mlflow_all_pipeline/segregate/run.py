#!/usr/bin/env python
import argparse
import logging
import os
import tempfile

import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def use_artifact(args):
    query = f"tag.step='{args.input_step}' and tag.current='1'"
    retrieved_run = mlflow.search_runs(experiment_ids=[mlflow.active_run().info.experiment_id],
                                       filter_string=query,
                                       order_by=["attributes.start_time DESC"],
                                       max_results=1)["run_id"][0]
    logger.info("retrieved run: " + retrieved_run)
    local_path = mlflow.tracking.MlflowClient().download_artifacts(retrieved_run, args.input_artifact)
    # MlflowClient().download_artifacts(retrieved_run, input_artifact, "./")
    logger.info("input_artifact: " + args.input_artifact + " at " + local_path)
    return local_path


def go(args):
    logger.info("Downloading and reading artifact")
    artifact_path = use_artifact(args)

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split first in model_dev/test, then we further divide model_dev in train and validation
    logger.info("Splitting data into train, val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != 'null' else None,
    )

    # Save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:
        for split, df in splits.items():
            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"{args.artifact_root}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            df.to_csv(temp_path)

            logger.info("Logging artifact")
            mlflow.log_artifact(temp_path, artifact_path=args.artifact_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--step", type=str, help="Current Step Name", required=True
    )

    parser.add_argument(
        "--input_step", type=str, help="Input Step Name", required=True
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
             "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
             "splitting",
        type=int,
        required=False,
        default=42
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default='null'  # unfortunately mlflow does not support well optional parameters
    )

    args = parser.parse_args()

    with mlflow.start_run() as run:
        go(args)
        mlflow.set_tag("step", args.step)
        mlflow.set_tag("current", "1")
