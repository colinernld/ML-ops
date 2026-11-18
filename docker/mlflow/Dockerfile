# MLflow tracking server + the two drivers this stack needs:
# psycopg2 for the Postgres backend store, boto3 for the MinIO (S3) artifact store.
FROM ghcr.io/mlflow/mlflow:v3.18.0

RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.44.5

EXPOSE 5000
