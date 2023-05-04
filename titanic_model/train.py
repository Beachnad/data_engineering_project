import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import pickle


def get_data():
    return pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')


def validate_data(df: pd.DataFrame):
    """
    There are better solutions for production environments such as
    the excellent, greate expectations platform.

    https://greatexpectations.io/
    """
    # Sex should only contain values 'male' and 'female'
    assert set(df['Sex']) == {'male', 'female'}, 'Invalid value for Sex column'

    # Pclass should only contain values 1, 2, and 3
    assert set(df['Pclass']) == {1, 2, 3}, 'Invalid value for Pclass column'

    # Age should fall on a range between 0 and 100
    assert min(df['Age']) >= 0, 'Age is smaller than 0'
    assert max(df['Age']) <= 100, 'Age is larger than 100'


def preprocess_data(df: pd.DataFrame):
    # One hot encode sex
    df['sex_male'] = (df['Sex'] == 'male').astype(int)

    # One hot passenger class
    df['class_first'] = (df['Pclass'] == 1).astype(int)
    df['class_second'] = (df['Pclass'] == 2).astype(int)

    # Rescale age from 0 to 100 to 0 to 1
    df['age'] = df['Age'] / 100
    df['age'] = df['age'].fillna(df['age'].median())

    return df


def train_model(X, Y):
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(X, Y)
    return knn


def run():
    df = get_data()
    validate_data(df)
    df = preprocess_data(df)

    X = df.loc[:, ['sex_male', 'class_first', 'class_second', 'age']]
    Y = df.Survived

    knn = train_model(X, Y)

    accuracy = np.sum(knn.predict(X) == Y) / len(X)
    print(accuracy)

    with open('knn_model.p', 'wb+') as f:
        pickle.dump(knn, f)


if __name__ == '__main__':
    run()
