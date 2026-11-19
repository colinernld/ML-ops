#!/usr/bin/env bash
# Create (or update) the `ride-secrets` Secret from the repository's .env file.
# Run once before the first `kubectl apply -k k8s/`, and again if you change a password.
#   ./k8s/create-secrets.sh            # uses ./.env
#   ./k8s/create-secrets.sh path/.env
set -euo pipefail
ENV_FILE="${1:-.env}"
NAMESPACE=ride-mlops
[[ -f "$ENV_FILE" ]] || { echo "No $ENV_FILE: run 'cp .env.example .env' and set the passwords first." >&2; exit 1; }

get() { grep -E "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2-; }
for key in MINIO_ROOT_USER MINIO_ROOT_PASSWORD POSTGRES_USER POSTGRES_PASSWORD; do
  [[ -n "$(get "$key")" ]] || { echo "$key is empty in $ENV_FILE" >&2; exit 1; }
done

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NAMESPACE" create secret generic ride-secrets \
  --from-literal=MINIO_ROOT_USER="$(get MINIO_ROOT_USER)" \
  --from-literal=MINIO_ROOT_PASSWORD="$(get MINIO_ROOT_PASSWORD)" \
  --from-literal=POSTGRES_USER="$(get POSTGRES_USER)" \
  --from-literal=POSTGRES_PASSWORD="$(get POSTGRES_PASSWORD)" \
  --from-literal=RELOAD_TOKEN="$(get RELOAD_TOKEN)" \
  --dry-run=client -o yaml | kubectl apply -f -
echo "Secret ride-secrets ready in namespace $NAMESPACE"
