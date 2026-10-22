#!/usr/bin/env bash
set -euo pipefail

# Script to re-train the RNN model on the IMDB dataset and save artifacts to models/

DATA_PATH="${1:-IMDB_dataset.csv}"
EPOCHS="${2:-20}"
BATCH_SIZE="${3:-64}"
PATIENCE="${4:-5}"
OUTPUT_DIR="${5:-models}"

echo "=========================================="
echo "Starting IMDB Sentiment RNN Training"
echo "Data:       ${DATA_PATH}"
echo "Epochs:     ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Patience:   ${PATIENCE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

python -m src.train \
  --data "${DATA_PATH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --patience "${PATIENCE}" \
  --output-dir "${OUTPUT_DIR}"

echo "Training complete! Artifacts saved to ${OUTPUT_DIR}/"
