from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import pandas as pd
import joblib

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="API for predicting employee attrition using the trained Random Forest model.",
    version="1.0"
)

# Load trained model
model = joblib.load("best_model.pkl")


class EmployeeData(BaseModel):
    data: Dict[str, Any]


@app.get("/")
def home():
    return {
        "message": "Employee Attrition Prediction API is running"
    }


@app.get("/features")
def get_features():
    try:
        features = list(model.feature_names_in_)
        return {
            "feature_count": len(features),
            "required_features": features
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve model features: {str(e)}"
        )


@app.post("/predict")
def predict(employee: EmployeeData):
    try:
        input_df = pd.DataFrame([employee.data])

        prediction = model.predict(input_df)[0]

        result = "Left" if int(prediction) == 1 else "Stayed"

        response = {
            "prediction": int(prediction),
            "turnover_status": result
        }

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
            response["attrition_probability"] = round(
                float(probability), 4
            )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
