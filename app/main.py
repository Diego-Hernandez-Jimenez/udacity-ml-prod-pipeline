from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from src.utils import load_config


class InputData(BaseModel):
    age: int = Field(gt=0, description="Age of the person in years.")
    workclass: str = Field(description="Employment classification.")
    fnlgt: int = Field(
        description="Final sampling weight assigned by the U.S. Census Bureau, representing the number of people in the population that this record represents."
    )
    education: str = Field(description="Highest level of education attained.")
    education_num: int = Field(
        ge=1,
        le=16,
        alias="education-num",
        description="Numerical code representing the highest level of education attained.",
    )
    marital_status: str = Field(alias="marital-status", description="Marital status.")
    occupation: str = Field(description="Occupation.")
    relationship: str = Field(
        description="Relationship to the household reference person."
    )
    race: str = Field(description="Race.")
    sex: Literal["Male", "Female"] = Field(description="Sex.")
    capital_gain: int = Field(
        alias="capital-gain", description="Annual capital gains in U.S. dollars."
    )
    capital_loss: int = Field(
        alias="capital-loss", description="Annual capital losses in U.S. dollars."
    )
    hours_per_week: int = Field(
        alias="hours-per-week", description="Average number of hours worked per week."
    )
    native_country: str = Field(
        alias="native-country", description="Country of origin."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        }
    )


class OutputPrediction(BaseModel):
    salary: Literal["<=50K", ">50K"] = Field(description="Predicted annual income.")


MODEL_DIR = Path(__file__).parent / "model"
cfg = load_config()
cat_features = cfg["data"]["categorical_features"]

model = joblib.load(MODEL_DIR / "model.pkl")
encoder = joblib.load(MODEL_DIR / "encoder.pkl")
lb = joblib.load(MODEL_DIR / "lb.pkl")

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the salary predictions API!"}


@app.post("/predict", response_model=OutputPrediction)
async def predict(data: InputData) -> OutputPrediction:
    record = pd.DataFrame(
        [
            {
                "age": data.age,
                "workclass": data.workclass,
                "fnlgt": data.fnlgt,
                "education": data.education,
                "education-num": data.education_num,
                "marital-status": data.marital_status,
                "occupation": data.occupation,
                "relationship": data.relationship,
                "race": data.race,
                "sex": data.sex,
                "capital-gain": data.capital_gain,
                "capital-loss": data.capital_loss,
                "hours-per-week": data.hours_per_week,
                "native-country": data.native_country,
            }
        ]
    )

    cat = record[cat_features]
    num = record[[c for c in record.columns if c not in cat_features]].values
    X = np.concatenate([num, encoder.transform(cat)], axis=1)

    prediction = model.predict(X)
    salary = lb.inverse_transform(prediction.reshape(-1, 1))[0]

    return OutputPrediction(salary=salary)
