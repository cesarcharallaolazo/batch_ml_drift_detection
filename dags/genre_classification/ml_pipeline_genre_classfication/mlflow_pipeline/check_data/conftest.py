import os
import logging
import pytest
import pandas as pd
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

mlflow.start_run()


def use_artifact(input_step, input_artifact):
    query = f"tag.step='{input_step}' and tag.current='1'"
    retrieved_run = mlflow.search_runs(experiment_ids=[mlflow.active_run().info.experiment_id],
                                       filter_string=query,
                                       order_by=["attributes.start_time DESC"],
                                       max_results=1)["run_id"][0]
    logger.info("retrieved run: " + retrieved_run)
    local_path = mlflow.tracking.MlflowClient().download_artifacts(retrieved_run, input_artifact)
    logger.info("input_artifact: " + input_artifact + " at " + local_path)
    return local_path


def pytest_addoption(parser):
    parser.addoption("--step", action="store")
    parser.addoption("--input_step", action="store")
    parser.addoption("--reference_artifact", action="store")
    parser.addoption("--sample_artifact", action="store")
    parser.addoption("--ks_alpha", action="store")


@pytest.fixture(scope="session")
def data(request):
    mlflow.set_tag("step", request.config.option.step)
    mlflow.set_tag("current", "1")

    reference_artifact = request.config.option.reference_artifact
    input_step = request.config.option.input_step

    if reference_artifact is None:
        pytest.fail("--reference_artifact missing on command line")

    sample_artifact = request.config.option.sample_artifact

    if sample_artifact is None:
        pytest.fail("--sample_artifact missing on command line")

    local_path = use_artifact(input_step, reference_artifact)
    sample1 = pd.read_csv(local_path)

    local_path = use_artifact(input_step, sample_artifact)
    sample2 = pd.read_csv(local_path)

    return sample1, sample2


@pytest.fixture(scope='session')
def ks_alpha(request):
    ks_alpha = request.config.option.ks_alpha

    if ks_alpha is None:
        pytest.fail("--ks_threshold missing on command line")

    return float(ks_alpha)
