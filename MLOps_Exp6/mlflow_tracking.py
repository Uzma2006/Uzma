
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/student_performance.csv")
X = df[["study_hours", "attendance"]]
y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Exp6_Hyperparameter_Tuning")

# Define parameter grid
param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2, 3]
}

# Train three models
baseline = DecisionTreeRegressor(random_state=42)

grid = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error"
)

random = RandomizedSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    n_iter=10,
    cv=3,
    scoring="neg_mean_squared_error",
    random_state=42
)

models = {
    "Baseline": baseline,
    "Grid_Search": grid,
    "Random_Search": random
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

        best_model = (
            model.best_estimator_
            if hasattr(model, "best_estimator_")
            else model
        )

        if hasattr(model, "best_params_"):
            mlflow.log_params(model.best_params_)

        mlflow.sklearn.log_model(
    best_model,
    name="decision_tree_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

        print(name)
        print("MSE:", mse)
        print("R2 Score:", r2)

print("All experiments tracked successfully!")
