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

```shell
docker build -t data_engineering_project .
docker run -p 80:80 data_engineering_project
```

The easiest way to begin testing the API is to use the doc page which is automatically included when using FastApi. Assuming that the API is running, the doc page should be accessible to here: `127.0.0.1:80/docs`. To submit a request, expand the POST endpoint click the "Try it out" button on the top right of the window.

## Continuous Testing

In order to catch mistakes before they impact production, some basic testing functionality using GitHub actions is used to run the tests found in the `tests/` folder.

> Since we want to deploy this model inside a Docker container, a production deployment would want to add steps that actually build and use the container for the testing environment. One way to achieve this would be to add a GitHub action to build and upload to a container registration service such as Docker hub or AWS ECR, then pull that image to perform the testing. There also  a [pytest-docker](https://github.com/avast/pytest-docker) package that could help.

As a part of the GitHub actions, there is also a linting step to check for any Python syntax errors and problematic patterns in the codebase.

## Next Steps and Improvements

As mentioned before, this project is far from production ready. Here are some items that should be considered before final deployment.

- Agree on a branching strategy for the project. Popular options are [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow), [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow), and [trunk based development](https://trunkbaseddevelopment.com/)
- Decide on an orchestration platform to regularly perform actions like monitering the performance of the deployed model, sending notifications on detected issues, and updating the predictive model with new data. [AirFlow](https://airflow.apache.org/) is the default choice, but other competitors like [Dagster](https://dagster.io/) and [Prefect](https://www.prefect.io/) are closing the gap in terms of features, community, and stability.
- Develop a data testing suite to validate the inputs to the predictive model's training data. For example, we would want to put in place some hard rules on expected values for age, sex, etc. but also provide a data profile report for the data science which summarizes each of the columns in terms of max, min, median values along with a histogram. My recommendation would be to start with an [established solution](https://github.com/ydataai/ydata-profiling) and build custom only when necessary.
- In order to support an orchestration platform, some additional services will be required: a database, object storage, networking, and security roles. If this were deployed on AWS, that would correlate to AuroraDB, S3, VPC configurations, security groups / iAM roles.
- Add automated deployment workflows to automatically trigger on a set condition such as a change to the main branch.
