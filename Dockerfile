FROM python:3.10-slim

RUN pip install --no-cache-dir \
    polars \
    "deltalake[pyarrow]" \
    mlflow \
    scikit-learn \
    pandas \
    boto3

WORKDIR /app
COPY . .
CMD ["python", "src/main.py"]