import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# MLflow setup
#dagshub integration is used here for logging and tracking  remotly
import dagshub
dagshub_owner="MariamHamid"
dagshub_repo="accelerator-tracking-demo"
dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)

#tracking locally
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("mlops-course")

MODEL_NAME = "mlops-course-model"

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameter sweep
for n in [10, 50, 100]:

    with mlflow.start_run(run_name=f"rf_n={n}"):

        # Tags (metadata)
        mlflow.set_tags({
            "model": "random_forest",
            "dataset": "iris",
            "experiment": "mlflow_testing"
        })

        max_depth = 5

        model = RandomForestClassifier(
            n_estimators=n,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro")
        rec = recall_score(y_test, y_pred, average="macro")

        # Log params & metrics
        mlflow.log_param("n_estimators", n)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log & register model
        info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random-forest-model",
            registered_model_name=MODEL_NAME,
            signature=signature
        )

        print(
            f"Logged run_id={info.run_id} | "
            f"Registered model={MODEL_NAME}"
        )
