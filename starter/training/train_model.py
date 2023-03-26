# Script to train machine learning model.

import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from starter.training.ml.data import process_data
from starter.training.ml.model import (
    compute_model_metrics,
    evaluate_model_on_data_slices,
    inference,
    save_model_or_encoder,
    train_model,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# get root of the project
root = os.path.abspath(Path(__file__).parent.parent.parent)

# Add code to load in the data.
data = pd.read_csv(Path(os.getcwd(), "starter", "data", "census_cleaned.csv"))
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    encoder=encoder,
    lb=lb,
    label="salary",
    training=False,
)

# Train and save a model.
model = train_model(X_train, y_train)

# model performance on the test set
preds = inference(model=model, X=X_test)
precision, recall, fbeta = compute_model_metrics(y=y_test, preds=preds)
logger.info(
    "Model performance on the test set: \n"
    + f"precision: {precision} \n"
    + f"recall: {recall} \n"
    + f"fbeta: {fbeta}"
)
# saving the model and the encoders
save_model_or_encoder(model, Path(root, "starter", "model", "classifier"))
save_model_or_encoder(encoder, Path(root, "starter", "model", "encoder"))
save_model_or_encoder(lb, Path(root, "starter", "model", "lb"))

# Evaluating the performance of the model on slices of the data
performance_saving_folder_path = Path(
    root, "starter", "starter", "model", "slice_performance"
)
performance_saving_folder_path.mkdir(parents=True, exist_ok=True)
for feat in cat_features:
    saving_path = Path(performance_saving_folder_path, f"{feat}_slice_performance.csv")
    evaluate_model_on_data_slices(model, data, feat, encoder, lb, saving_path)
