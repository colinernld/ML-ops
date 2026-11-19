#!/usr/bin/env bash
# Train, register and promote a model INSIDE the Kubernetes cluster (no Python setup on the laptop).
# A one-off Job runs the API image (it already has PyTorch, MLflow and trajclf) on the code of this
# repository and the raw data folder of this Mac, both mounted read-only, then the API is restarted
# to serve the new @production model.
#   ./k8s/train-in-cluster.sh                      # data in ./data/raw
#   ./k8s/train-in-cluster.sh "/path/to/Project"   # folder with GBG500.parquet + GBG500_labels.csv
set -euo pipefail
NAMESPACE=ride-mlops
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$(cd "${1:-$REPO_DIR/data/raw}" && pwd)"
IMAGE="${IMAGE:-docker.io/colinernld/ride-api:latest}"

for f in GBG500.parquet GBG500_labels.csv; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "Missing $DATA_DIR/$f (download Project.zip from Inside)" >&2; exit 1; }
done

kubectl -n "$NAMESPACE" delete job train --ignore-not-found
kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: train
  labels: {app.kubernetes.io/name: train, app.kubernetes.io/part-of: ride-mlops}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    metadata:
      labels: {app.kubernetes.io/name: train}
    spec:
      restartPolicy: Never
      containers:
        - name: train
          image: $IMAGE
          workingDir: /repo
          command: ["sh", "-c"]
          args:
            - >
              python -m trajclf.data --raw-dir /data/raw --out-dir /tmp/processed &&
              python -m trajclf.train --data-dir /tmp/processed --out-dir /tmp/artifacts --register-name ride-classifier &&
              python -m trajclf.promote
          env:
            - {name: PYTHONPATH, value: /repo/src}
            - {name: MLFLOW_TRACKING_URI, value: "http://mlflow:5000"}
            - {name: MLFLOW_DISABLE_AGENT_HINT, value: "1"}
          resources:
            requests: {cpu: 500m, memory: 1Gi}
            limits: {memory: 3Gi}
          volumeMounts:
            - {name: repo, mountPath: /repo, readOnly: true}
            - {name: data, mountPath: /data/raw, readOnly: true}
      volumes:
        - name: repo
          hostPath: {path: "$REPO_DIR", type: Directory}
        - name: data
          hostPath: {path: "$DATA_DIR", type: Directory}
EOF

echo "Waiting for the training pod to start..."
for _ in $(seq 1 60); do   # the Job creates its pod a moment after `apply`
  kubectl -n "$NAMESPACE" get pod -l job-name=train -o name | grep -q . && break
  sleep 2
done
# Ready while running; a pod that already finished never becomes Ready, hence `|| true`.
kubectl -n "$NAMESPACE" wait --for=condition=Ready pod -l job-name=train --timeout=300s || true
kubectl -n "$NAMESPACE" logs -f job/train
if ! kubectl -n "$NAMESPACE" wait --for=condition=Complete job/train --timeout=120s; then
  echo "Training job failed: kubectl -n $NAMESPACE describe job/train" >&2
  exit 1
fi

echo "Restarting the API so it loads models:/ride-classifier@production..."
kubectl -n "$NAMESPACE" rollout restart deployment/api
kubectl -n "$NAMESPACE" rollout status deployment/api --timeout=300s
