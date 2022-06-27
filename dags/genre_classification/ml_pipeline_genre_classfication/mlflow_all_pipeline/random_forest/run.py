import argparse
import itertools
import logging
import os
import shutil

import yaml
import tempfile
import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def use_artifact(args):
    query = f"tag.step='{args.input_step}' and tag.current='1'"
    retrieved_run = mlflow.search_runs(experiment_ids=[mlflow.active_run().info.experiment_id],
                                       filter_string=query,
                                       order_by=["attributes.start_time DESC"],
                                       max_results=1)["run_id"][0]
    logger.info("retrieved run: " + retrieved_run)
    local_path = mlflow.tracking.MlflowClient().download_artifacts(retrieved_run, args.train_data)
    # MlflowClient().download_artifacts(retrieved_run, input_artifact, "./")
    logger.info("input_artifact: " + args.train_data + " at " + local_path)
    return local_path


def go(args):
    logger.info("Downloading and reading train artifact")
    train_data_path = use_artifact(args)
    df = pd.read_csv(train_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X = df.copy()
    y = X.pop("genre")

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=df[args.stratify] if args.stratify != "null" else None,
        random_state=args.random_seed,
    )

    logger.info("Setting up pipeline")

    pipe, used_columns = get_training_inference_pipeline(args)

    logger.info("Fitting")
    pipe.fit(X_train[used_columns], y_train)

    # Evaluate
    pred = pipe.predict(X_val[used_columns])
    pred_proba = pipe.predict_proba(X_val[used_columns])

    logger.info("Scoring")
    score = roc_auc_score(y_val, pred_proba, average="macro", multi_class="ovo")

    mlflow.log_metric("AUC", score)

    # Export if required
    if args.export_artifact != "null":
        export_model(pipe, used_columns, X_val, pred, args.export_artifact)

    base_name_img = "./img/"
    if not os.path.exists(base_name_img):
        os.mkdir(base_name_img)
    # Some useful plots
    fig_feat_imp = plot_feature_importance(pipe)
    fig_feat_imp.savefig(base_name_img + 'feature_importance.png')

    fig_cm, sub_cm = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(
        pipe,
        X_val[used_columns],
        y_val,
        ax=sub_cm,
        normalize="true",
        values_format=".1f",
        xticks_rotation=90,
    )
    fig_cm.tight_layout()
    fig_cm.savefig(base_name_img + 'confusion_matrix.png')

    mlflow.log_artifacts(base_name_img, "img")
    shutil.rmtree(base_name_img)


def export_model(pipe, used_columns, X_val, val_pred, export_artifact):
    # Infer the signature of the model

    # Get the columns that we are really using from the pipeline
    signature = infer_signature(X_val[used_columns], val_pred)

    with tempfile.TemporaryDirectory() as temp_dir:
        mlflow.sklearn.log_model(
            pipe,
            export_artifact,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=X_val.iloc[:2],
        )


def plot_feature_importance(pipe):
    # We collect the feature importance for all non-nlp features first
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1]
        + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["classifier"].feature_importances_[len(feat_names):])
    feat_imp = np.append(feat_imp, nlp_importance)
    feat_names = np.append(feat_names, "title + song_name")
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp[idx], color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_training_inference_pipeline(args):
    # Get the configuration for the pipeline
    with open(args.model_config) as fp:
        model_config = yaml.safe_load(fp)
    mlflow.log_params(model_config["random_forest"])

    # We need 3 separate preprocessing "tracks":
    # - one for categorical features
    # - one for numerical features
    # - one for textual ("nlp") features
    # Categorical preprocessing pipeline
    categorical_features = sorted(model_config["features"]["categorical"])
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0), OrdinalEncoder()
    )
    # Numerical preprocessing pipeline
    numeric_features = sorted(model_config["features"]["numerical"])
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    # Textual ("nlp") preprocessing pipeline
    nlp_features = sorted(model_config["features"]["nlp"])
    # This trick is needed because SimpleImputer wants a 2d input, but
    # TfidfVectorizer wants a 1d input. So we reshape in between the two steps
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    nlp_transformer = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=True, max_features=model_config["tfidf"]["max_features"]
        ),
    )
    # Put the 3 tracks together into one pipeline using the ColumnTransformer
    # This also drops the columns that we are not explicitly transforming
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("nlp1", nlp_transformer, nlp_features),
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    # Get a list of the columns we used
    used_columns = list(itertools.chain.from_iterable([x[2] for x in preprocessor.transformers]))

    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**model_config["random_forest"])),
        ]
    )
    return pipe, used_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Random Forest",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--step", type=str, help="Current Step Name", required=True
    )

    parser.add_argument(
        "--input_step", type=str, help="Input Step Name", required=True
    )

    parser.add_argument(
        "--train_data",
        type=str,
        help="Fully-qualified name for the training data artifact",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to a YAML file containing the configuration for the random forest",
        required=True,
    )

    parser.add_argument(
        "--export_artifact",
        type=str,
        help="Name of the artifact for the exported model. Use 'null' for no export.",
        required=False,
        default="null",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for the random number generator.",
        required=False,
        default=42
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size for the validation set as a fraction of the training set",
        required=False,
        default=0.3
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Name of a column to be used for stratified sampling. Default: 'null', i.e., no stratification",
        required=False,
        default="null",
    )

    args = parser.parse_args()

    with mlflow.start_run() as run:
        go(args)
        mlflow.set_tag("step", args.step)
        mlflow.set_tag("current", "1")
