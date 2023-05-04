FROM python:3.9

WORKDIR /code

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python titanic_model/train.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
