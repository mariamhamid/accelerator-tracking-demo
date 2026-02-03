# -------------------------
# BASIC IMPORTS
# -------------------------

# Used to read environment variables (like secrets or repo names)
import os

# MLflow is used to track experiments, metrics, parameters, and models
import mlflow
import mlflow.sklearn

# Used to automatically infer model input/output format (important for deployment)
from mlflow.models.signature import infer_signature


# -------------------------
# MACHINE LEARNING TOOLS
# -------------------------

# Load a built-in dataset (Iris dataset) for training and testing
from sklearn.datasets import load_iris

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

# Random Forest model (ensemble learning model)
from sklearn.ensemble import RandomForestClassifier

# Metric to evaluate how good the model is
from sklearn.metrics import accuracy_score


# -------------------------
# DAGSHUB INTEGRATION
# -------------------------

# DagsHub allows us to track MLflow experiments remotely (online)
import dagshub

# GitHub / DagsHub repository owner
DAGSHUB_OWNER = "MariamHamid"

# Repository name where experiments will be tracked
DAGSHUB_REPO = "accelerator-tracking-demo"

# Initialize DagsHub + MLflow integration
# This tells MLflow to log everything to DagsHub instead of only locally
dagshub.init(
    repo_owner=DAGSHUB_OWNER,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)


# -------------------------
# TRAINING FUNCTION
# -------------------------

# This function trains multiple Random Forest models
# using different hyperparameters and logs everything to MLflow
def run_training(
    n_values=(10, 50, 100),        # Different values for number of trees
    max_depth=5,                  # Maximum depth of each tree
    experiment_name="mlops-demo", # Name of the MLflow experiment
    model_name="mlops-demo-model",# Name for model registry
    use_registry=True             # Whether to register the model or not
):

    # Set (or create) an MLflow experiment
    # All runs will be grouped under this name
    mlflow.set_experiment(experiment_name)

    # -------------------------
    # LOAD & PREPARE DATA
    # -------------------------

    # Load Iris dataset (features X and labels y)
    X, y = load_iris(return_X_y=True)

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # List to store results for comparison later
    results = []

    # -------------------------
    # TRAIN MULTIPLE MODELS
    # -------------------------

    # Loop over different values of n_estimators
    for n in n_values:

        # Start an MLflow run (one run = one experiment attempt)
        with mlflow.start_run(
            run_name=f"rf_n{n}",
            tags={"model": "RandomForest", "dataset": "iris"}
        ):

            # Create the Random Forest model
            model = RandomForestClassifier(
                n_estimators=int(n),   # Number of trees
                max_depth=int(max_depth),
                random_state=42
            )

            # Train the model using training data
            model.fit(X_train, y_train)

            # Make predictions on test data
            preds = model.predict(X_test)

            # Calculate accuracy (how many predictions were correct)
            acc = accuracy_score(y_test, preds)

            # -------------------------
            # LOG PARAMETERS & METRICS
            # -------------------------

            # Log hyperparameters to MLflow
            mlflow.log_param("n_estimators", int(n))
            mlflow.log_param("max_depth", int(max_depth))

            # Log model performance metric
            mlflow.log_metric("accuracy", float(acc))

            # -------------------------
            # MODEL SIGNATURE
            # -------------------------

            # Infer input/output schema for the model
            # This is important for reproducibility and deployment
            signature = infer_signature(
                X_train,
                model.predict(X_train)
            )

            # -------------------------
            # LOG & REGISTER MODEL
            # -------------------------

            try:
                # Log the model to MLflow
                # Optionally register it in the model registry
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=model_name if use_registry else None,
                )
                registered = use_registry

            except Exception as e:
                # If model registry fails, still log the model
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                )
                registered = False

            # -------------------------
            # SAVE RESULTS
            # -------------------------

            # Store results for later analysis
            results.append({
                "n_estimators": int(n),
                "accuracy": float(acc),
                "registered": registered
            })

    # Return all experiment results
    return results
