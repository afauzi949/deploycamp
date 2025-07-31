# Gunakan image MLflow resmi sebagai base
FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install psycopg2 dan dependensi lain
RUN pip install psycopg2-binary