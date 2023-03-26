import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from starter.starter.ml.data import process_data

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    cls = DecisionTreeClassifier(random_state=1234)
    cls.fit(X_train, y_train)
    return cls


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model :
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model_or_encoder(entity, file_path):
    """Save a model or an encoder

    Args:
        path (Path): path to the saved file
    """
    with open(Path(f"{file_path}.pkl"), "wb") as f:
        pickle.dump(entity, f)


def load_model_or_encoder(file_path):
    """Load a sklearn model or encoder

    Args:
        file_path (str): path to the entity

    Returns:
        sklearn model or encoder: model or encoder
    """
    with open(Path(file_path), "rb") as f:
        entity = pickle.load(f)
    return entity


def evaluate_model_on_data_slices(model, df, feature, encoder, lb, saving_path=None):
    """Evaluate a given model performance on slices of data

    Args:
        model (sklearn model): a trained sklearn model
        df (pd.DataFrame): pandas dataframe, cleaned data.
        feature (str): the feature of interest on which the slices will be made.
        encoder (OneHotEncoder): fitted one hot encoder
        lb (LabelBinarizer): fitted label binarizer
        saving_path (str, optional): if provided, save a csv file containing the results
            of the evaluation . Defaults to None.
    """

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    precision_list, recall_list, fbeta_list, feat_list = [], [], [], []
    for feat in df[feature].unique():
        slice = df[df[feature] == feat]
        X, y, _, _ = process_data(
            slice,
            categorical_features=cat_features,
            label="salary",
            training=False,
            encoder=encoder,
            lb=lb,
        )

        preds = inference(model, X)
        precision, recall, fbeta = compute_model_metrics(y, preds)
        precision_list.append(precision)
        recall_list.append(recall)
        fbeta_list.append(fbeta)
        feat_list.append(feat)

    report = pd.DataFrame(
        data={
            "feature": feat_list,
            "precision": precision_list,
            "recall": recall_list,
            "fbeta": fbeta_list,
        }
    )
    logger.info(f"Evaluation result on feature slice {feature}: \n {report}")
    if saving_path:
        report.to_csv(saving_path, index=False)
