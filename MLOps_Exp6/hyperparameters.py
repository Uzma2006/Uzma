
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state=42)

print("Decision Tree Hyperparameters:")
print(model.get_params())
