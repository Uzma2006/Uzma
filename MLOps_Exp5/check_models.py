
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiment = client.get_experiment_by_name(
    "Student_Performance_Experiment"
)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'Linear_Regression'",
    order_by=["attributes.start_time DESC"],
    max_results=3
)

print("\nLINEAR REGRESSION RUNS")

for run in runs:
    print("\nRun ID:", run.info.run_id)
    print("Artifact URI:", run.info.artifact_uri)

    models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id]
    )

    for model in models:
        if model.source_run_id == run.info.run_id:
            print("Logged Model ID:", model.model_id)
            print("Source Run ID:", model.source_run_id)
print("Model Name:", model.name)
print("Model Status:", model.status)


print("\nDiagnostic completed!")

