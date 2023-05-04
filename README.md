# Data Engineering Demo Project

This repository is a demonstration on how to deploy a k-nearest neighbors model which predicts if a hypothetical passenger on the Titanic would survive or not. The model takes as input a passenger age, sex, and class and outputs a 1 if they are predicted to survive, otherwise returns a 0 if they are predicted to not survive.

## Training the Model

The code used to produce the predictive model is found inside the `titanic_model/train.py` file. Truth be told, this model is not very good, but is serviceable for our purposes. Some basic techniques are applied, like using one hot encoding to convert categorical variables to numerical representations and imputation of missing data.

Training the model is achieved by executing the following from the command line.

```shell
python titanic_model/train.py
```

This script will output a pickled model `titanic_model/knn_model.p`.

> Some basic validation is performed on the incoming training data and could be enhanced by using a dedicated validation library such as [great expectations](https://greatexpectations.io/). There are some easy validations to identify, such as age being constrained between 0 and 100, sex can be male or female, missing values, etc. However, there are also more subtle data quality issues that can occur which may require collaboration with the data science team. For example, bad data can come in the form of bad distributions, where we might expect a few elderly passengers on board, but if the vast majority are reportedly over the age of 90, we would want to question the data we are receiving.

## Deploying the Model

Having a great predictive model is nice, but being able to programmatically access through an API takes it to the next level. For this project, I used [FastApi](https://fastapi.tiangolo.com/) which is easy to stand up, has great built-in data validation, provides an interactive sandbox, and as the name suggests - is very fast as well.

The code handling the api is found in the `apy.py` script and can be easily ran with `uvicorn api:app`.

> uvicorn is not recommended for production applications, but it does offer a very easy low-hassle method to locally test. 

## Running with Docker

Finally, to help this code easily port to other machines, we use Docker to create an image and run the code inside a Docker container built from this image. 

```bash
docker build -t data_engineering_project .
docker run --name mycontainer -p 80:80 data_engineering_project
```
