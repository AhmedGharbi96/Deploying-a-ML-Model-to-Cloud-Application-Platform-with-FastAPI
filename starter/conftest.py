import os
from pathlib import Path

import pytest
from starter.training.ml.model import load_model_or_encoder

root = os.path.abspath(Path(__file__).parent.parent)


@pytest.fixture(scope="session")
def loaded_model():
    return load_model_or_encoder(Path(root, "starter", "model", "classifier.pkl"))


@pytest.fixture()
def loaded_encoder(scope="session"):
    return load_model_or_encoder(Path(root, "starter", "model", "encoder.pkl"))


@pytest.fixture(scope="session")
def loaded_lb():
    return load_model_or_encoder(Path(root, "starter", "model", "lb.pkl"))


@pytest.fixture(scope="session")
def json_sample_1():
    payload = {
        "age": 79,
        "workclass": " Self-emp-not-inc",
        "fnlgt": 103684,
        "education": " HS-grad",
        "education-num": 9,
        "marital-status": " Married-civ-spouse",
        "occupation": " Farming-fishing",
        "relationship": " Husband",
        "race": " White",
        "sex": " Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": " United-States",
    }

    return payload


@pytest.fixture(scope="session")
def json_sample_2():
    payload = {
        "age": 61,
        "workclass": " Private",
        "fnlgt": 181219,
        "education": " Bachelors",
        "education-num": 13,
        "marital-status": " Married-civ-spouse",
        "occupation": " Prof-specialty",
        "relationship": " Husband",
        "race": " White",
        "sex": " Male",
        "capital-gain": 0,
        "capital-loss": 1848,
        "hours-per-week": 40,
        "native-country": " United-States",
    }

    return payload


@pytest.fixture(scope="session")
def json_sample_with_error():
    # contains extra field: salary
    payload = {
        "age": 61,
        "workclass": " Private",
        "fnlgt": 181219,
        "education": " Bachelors",
        "education-num": 13,
        "marital-status": " Married-civ-spouse",
        "occupation": " Prof-specialty",
        "relationship": " Husband",
        "race": " White",
        "sex": " Male",
        "capital-gain": 0,
        "capital-loss": 1848,
        "hours-per-week": 40,
        "native-country": " United-States",
        "salary": " >50K",
    }

    return payload
