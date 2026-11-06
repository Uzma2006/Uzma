import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model():

    data = {
        "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
        "marks": [35, 40, 50, 55, 65, 70, 80, 90]
    }

    df = pd.DataFrame(data)

    X = df[["study_hours"]]
    y = df["marks"]

    model = LinearRegression()

    model.fit(X, y)

    predictions = model.predict(X)

    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    print("Model Training Successful")
    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    return model

if __name__ == "__main__":
    train_model()