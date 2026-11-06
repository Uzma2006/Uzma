from train import train_model
import pandas as pd

def test_model_training():

    model = train_model()

    assert model is not None

def test_model_prediction():

    model = train_model()

    prediction = model.predict(
    pd.DataFrame({"study_hours": [5]})
)

    assert len(prediction) == 1

    assert prediction[0] > 0

def test_model_accuracy():

    model = train_model()

    assert model.score(
        pd.DataFrame({
            "study_hours": [1, 2, 3, 4, 5]
        }),
        [35, 40, 50, 55, 65]
    ) > 0.80