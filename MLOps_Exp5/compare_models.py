
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/student_performance.csv")

X = df[["study_hours", "attendance"]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Student_Performance_Experiment")

models = {
    "Linear_Regression": LinearRegression(),
    "Decision_Tree": DecisionTreeRegressor(random_state=42),
    "Random_Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2_Score", r2)

        trusted_types = []

    if name in ["Decision_Tree", "Random_Forest"]:
     trusted_types = ["sklearn.tree._tree.Tree"]

    mlflow.sklearn.log_model(
           model,
           name="model",
           skops_trusted_types=trusted_types
        )

    
    print(f"\nModel: {name}")
    print("MSE:", mse)
    print("R2 Score:", r2)

print("\nAll models compared successfully!")
