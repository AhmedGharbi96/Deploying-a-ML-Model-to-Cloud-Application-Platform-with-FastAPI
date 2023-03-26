import logging

import joblib
import pandas as pd
import uvicorn

# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Extra, Field
from starter.training.ml.data import load_data_from_s3, process_data
from starter.training.ml.model import inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
app = FastAPI()
BUCKET_NAME = "mlops-mini-project-udacity"


# Load the models on the startup of the app
@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    files_to_download = ["classifier", "encoder", "lb"]
    for file in files_to_download:
        load_data_from_s3(
            bucket_name=BUCKET_NAME,
            remote_path=f"trained_model_for_deployment/{file}.pkl",
            local_path=f"starter/model/{file}.pkl",
        )
    model = joblib.load("starter/model/classifier.pkl")
    encoder = joblib.load("starter/model/encoder.pkl")
    lb = joblib.load("starter/model/lb.pkl")


@app.get("/")
async def get_root() -> dict:
    return {"message": "Welcome to the app of census data classification"}


def replace_underscore_with_hyphen(string: str) -> str:
    return string.replace("_", "-")


# Class definition of the data that will be provided as POST request
class CensusData(BaseModel):
    age: int = Field(..., example=43)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=339814)
    education: str = Field(..., example="Some-college")
    education_num: int = Field(..., example=10)
    marital_status: str = Field(..., example="Never-married")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=5178)
    capital_loss: int = Field(..., example=0)
    hours_per_week: int = Field(..., example=40)
    native_country: str = Field(..., example="United-States")

    class Config:
        alias_generator = replace_underscore_with_hyphen
        extra = Extra.forbid


@app.post("/predict")
async def predict(input: CensusData):
    """
    POST request that will provide sample census data and expect a prediction
    Output:
        0 or 1
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

    # Read data sent as POST
    input_data = input.dict(by_alias=True)
    input_df = pd.DataFrame(input_data, index=[0])
    logger.info(f"Input data: {input_df}")

    # Process the data
    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X)
    preds = int(preds[0])
    logger.info(f"Predictions: {preds}")
    return {"prediction": preds}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
