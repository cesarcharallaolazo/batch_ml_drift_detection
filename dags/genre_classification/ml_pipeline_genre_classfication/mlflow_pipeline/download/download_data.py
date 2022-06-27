import os
import argparse
import logging
import requests
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    # Download file
    logger.info(f"Downloading {args.file_url} ...")
    # with open(basename, 'wb+') as fp:
    with open(args.artifact_name, 'wb+') as fp:
        try:
            # Download the file streaming and write to open temp file
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

            fp.flush()

            logger.info("Logging artifact")
            mlflow.log_artifact(args.artifact_name)
            os.remove(args.artifact_name)
        except Exception as e:
            logger.error("-- Failed --")
            logger.error(e.args)
            os.remove(args.artifact_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to MLFlow"
    )

    parser.add_argument(
        "--step", type=str, help="Current Step Name", required=True
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
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
