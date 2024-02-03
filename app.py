import os
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from src.models.predict_model import MLflowModel


# Initialize the FastAPI application
app = FastAPI(title="Hotel Booking Cancellation Predictor",
              description="An API for predicting hotel booking cancellations using advanced \
                           Machine Learning models."
              )

# Initialize the MLflow model
model = MLflowModel()


# Root endpoint
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Hotel Booking Cancellation Prediction API!"}


# POST endpoint with path '/predict'
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
    if file.filename.endswith(".csv"):
        # Create a temporary file with the same name as the uploaded
        # CSV file to load the data into a pandas Dataframe
        with open(file.filename, "wb") as f:
            f.write(file.file.read())
        input_data = pd.read_csv(file.filename)
        os.remove(file.filename)
        prediction = model.predict(input_data)
        # Return a JSON object containing the model predictions
        return {"prediction": str(prediction)}
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")

