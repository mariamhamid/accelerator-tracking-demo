# api.py
# This file exposes our ML training logic as a web API using FastAPI


# -------------------------
# IMPORTS
# -------------------------

# FastAPI is used to create the web API
from fastapi import FastAPI

# Pydantic is used to validate and structure request/response data
from pydantic import BaseModel, Field

# Used for type hints (lists, optional values)
from typing import List, Optional

# Import the training function from train.py
# This is where the ML model training actually happens
from train import run_training

# MLflow is used to track experiments and models
import mlflow


# -------------------------
# CREATE FASTAPI APP
# -------------------------

# Create the FastAPI application
# The title appears in Swagger UI (/docs)
app = FastAPI(title="Iris Trainer API")


# -------------------------
# REQUEST SCHEMA
# -------------------------

# This class defines what data the client must send
# when calling the /train endpoint
class TrainRequest(BaseModel):

    # List of values for number of trees in Random Forest
    # If the user does not provide it, these defaults are used
    n_values: List[int] = Field(default_factory=lambda: [10, 50, 100])

    # Maximum depth of each tree
    max_depth: int = 5

    # Name of the MLflow experiment
    experiment: str = "mlops-course"

    # Name under which the model will be registered
    model_name: str = "mlops-demo-model"

    # Whether to register the model in MLflow Model Registry
    use_registry: bool = True


# -------------------------
# RESPONSE SCHEMA
# -------------------------

# This defines the structure of the response returned by /train
class TrainResponse(BaseModel):

    # Shows where MLflow is tracking experiments (local or remote)
    tracking_uri: str

    # Results of training (metrics, parameters, registry status)
    results: List[dict]



# -------------------------
# HEALTH CHECK ENDPOINT
# -------------------------

# This endpoint is used to check if the API is alive
# Commonly used by Docker, Kubernetes, or monitoring tools
@app.get("/health")
def health_check():
    return {"status": "API is running"}


# -------------------------
# TRAINING ENDPOINT
# -------------------------

# This endpoint triggers model training
# It receives parameters, runs training, and returns results
@app.post("/train", response_model=TrainResponse)
def train_model(request: TrainRequest):

    # Call the training function with values from the request
    results = run_training(
        n_values=request.n_values,
        max_depth=request.max_depth,
        experiment_name=request.experiment,
        model_name=request.model_name,
        use_registry=request.use_registry
    )

    # Return MLflow tracking URI and training results
    return TrainResponse(
        tracking_uri=mlflow.get_tracking_uri(),
        results=results
    )


# -------------------------
# RUN SERVER (LOCAL ONLY)
# -------------------------

# This block allows the API to be run directly using:
# python api.py
# In production, uvicorn is usually run from the command line
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
