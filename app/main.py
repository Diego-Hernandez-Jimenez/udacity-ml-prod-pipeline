from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


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


app = FastAPI()


# GET on the root giving a welcome message.
@app.get("/")
async def root():
    return {"message": "Welcome to the salary predictions API!"}


# POST that does model inference
@app.post("/predict", response_model=OutputPrediction)
async def predict(data: InputData) -> OutputPrediction:
    return OutputPrediction(salary="<=50K")
