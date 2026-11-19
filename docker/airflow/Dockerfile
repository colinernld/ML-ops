# Airflow 3 (scheduler + UI in one container: `airflow standalone`) + the project's ML dependencies.
# The ML libraries live in their own virtualenv (/opt/project-venv): the DAG tasks call
# /opt/project-venv/bin/python -m trajclf.pipeline <step>, so Airflow's pinned dependencies and
# MLflow/PyTorch never conflict. The project code itself is mounted at /opt/project (docker-compose.yml).
FROM apache/airflow:3.4.0-python3.12

USER root
RUN mkdir -p /opt/project-venv && chown airflow: /opt/project-venv
USER airflow

RUN python -m venv /opt/project-venv \
    && /opt/project-venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/project-venv/bin/pip install --no-cache-dir \
        "numpy>=1.26" "pandas>=2.1" "pyarrow>=14" "pyyaml>=6" "scikit-learn>=1.4" "mlflow==3.18.0" "boto3>=1.34"

ENV PROJECT_DIR=/opt/project \
    PROJECT_PYTHON=/opt/project-venv/bin/python \
    AIRFLOW__CORE__DAGS_FOLDER=/opt/project/dags \
    AIRFLOW__CORE__LOAD_EXAMPLES=false
