FROM python:3.8-slim

WORKDIR /app

COPY data/question_sample.csv data/question_sample.csv

COPY App . 

COPY requirements.txt .  

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
