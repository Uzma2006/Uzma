
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
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

# Create baseline model
model = DecisionTreeRegressor(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Baseline Model: Decision Tree")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)
