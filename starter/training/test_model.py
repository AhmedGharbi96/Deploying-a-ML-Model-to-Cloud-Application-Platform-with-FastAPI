import os
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from starter.training.ml.data import process_data
from starter.training.ml.model import (
    compute_model_metrics,
    inference,
    save_model_or_encoder,
    train_model,
)

# get root of the project
root = os.path.abspath(Path(__file__).parent.parent.parent)


@pytest.fixture()
def data():
    """Load clean data for test

    Returns:
        pd.DataFrame: clean data
    """
    # Add code to load in the data.
    data = pd.read_csv(Path(root, "starter", "data", "census_cleaned.csv"))
    return data


def test_train_model(data):
    # Split dataset into training and testing set
    train, _ = train_test_split(data, test_size=0.20, random_state=1234)

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

    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    model = train_model(X_train, y_train)
    preds = model.predict(X_train)
    assert preds.shape[0] == y_train.shape[0]


def test_inference(data, loaded_model, loaded_encoder, loaded_lb):

    # Split dataset into training and testing set
    _, test = train_test_split(data, test_size=0.20, random_state=1234)

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
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        encoder=loaded_encoder,
        lb=loaded_lb,
        label="salary",
        training=False,
    )

    preds = inference(model=loaded_model, X=X_test)
    assert preds.shape[0] == y_test.shape[0]
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert pytest.approx(precision, 0.001) == 0.911
    assert pytest.approx(recall, 0.001) == 0.928
    assert pytest.approx(fbeta, 0.001) == 0.92


def test_save_model_or_encoder(tmp_path, loaded_model):
    save_model_or_encoder(loaded_model, Path(tmp_path, "classifier"))

    assert Path(tmp_path, "classifier.pkl").is_file()
