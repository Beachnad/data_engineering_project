import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from titanic_model.train import run, get_data, preprocess_data


# Train model
run()

with open('titanic_model/knn_model.p', 'rb') as f:
    knn_model: KNeighborsClassifier = pickle.load(f)


def test_model_accuracy():
    # Overall, test for a minimum accuracy
    df = get_data()
    df = preprocess_data(df)
    X = df.loc[:, ['sex_male', 'class_first', 'class_second', 'age']]
    Y = df.Survived

    accuracy = np.sum(knn_model.predict(X) == Y) / len(X)
    assert accuracy > 0.70


def test_model_extremes():
    # Make sure that the model is correct at the extremes
    df = pd.DataFrame([
        # elderly male in third class - not survived
        {
            'sex_male': 1,
            'class_first': 0,
            'class_second': 0,
            'age': 0.90,
            'survived': 0
        },
        # young female in first class - survived
        {
            'sex_male': 0,
            'class_first': 1,
            'class_second': 0,
            'age': 0.10,
            'survived': 1
        }
    ])

    X = df.loc[:, ['sex_male', 'class_first', 'class_second', 'age']]
    Y = df.survived

    accuracy = np.sum(knn_model.predict(X) == Y) / len(X)
    assert accuracy == 1
