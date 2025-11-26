FROM python:3.14-slim

WORKDIR /app 

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN mkdir -p /app/data && chmod -R 777 /app/data

COPY . . 

CMD ["python", "ml-benchmark.py"] 