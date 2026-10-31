import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/student_performance.csv")

# Select input and output
X = df[["study_hours", "attendance"]]
y = df["marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow experiment
mlflow.set_experiment("Student_Performance_Experiment")

with mlflow.start_run(run_name="Linear_Regression"):

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters and metrics
    mlflow.log_param("model", "Linear Regression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2_Score", r2)

    # Log model
    mlflow.sklearn.log_model(
        model,
        name="student_marks_model"
    )

    print("Model trained successfully!")
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)