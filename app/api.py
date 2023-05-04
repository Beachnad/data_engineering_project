import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, conint
from enum import Enum, IntEnum
import pickle
from sklearn.neighbors import KNeighborsClassifier
from titanic_model.train import preprocess_data


with open('titanic_model/knn_model.p', 'rb') as f:
    knn_model: KNeighborsClassifier = pickle.load(f)


class PassengerClassEnum(IntEnum):
    first = 1
    second = 2
    third = 3


class SexEnum(str, Enum):
    male = 'male'
    female = 'female'


app = FastAPI()


class Passenger(BaseModel):
    passenger_class: PassengerClassEnum
    sex: SexEnum
    age: conint(ge=0, le=100)


@app.post("/predict/")
def predict(passenger: Passenger):
    df = pd.DataFrame([{
        'Sex': passenger.sex,
        'Pclass': passenger.passenger_class,
        'Age': passenger.age
    }])
    df = preprocess_data(df)
    X = df.loc[:, ['sex_male', 'class_first', 'class_second', 'age']]

    prediction = int(knn_model.predict(X)[0])
    return {'survives': prediction}
