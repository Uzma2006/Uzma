
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/student_performance.csv")

X = df[["study_hours", "attendance"]]
y = df["marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = DecisionTreeRegressor(random_state=42)

# Define parameter grid
param_grid = {
    "max_depth": [2, 3, 4, 5, None],
    "min_samples_split": [2, 3, 4],
    "min_samples_leaf": [1, 2, 3]
}

# Apply Grid Search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
predictions = best_model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Grid Search Results")
print("Best Parameters:", grid_search.best_params_)
print("Best CV MSE:", -grid_search.best_score_)
print("Test MSE:", mse)
print("Test R2 Score:", r2)
