import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

model_id = "m-e5a8efa9647b4fc28d9ad28079c365a9"
model_uri = f"models:/{model_id}"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name="StudentPerformanceModel"
)

print("Model registered successfully!")
print("Model Name:", registered_model.name)
print("Model Version:", registered_model.version)
print("Model ID:", model_id)