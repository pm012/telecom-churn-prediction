#!/bin/bash
# run_all.sh - Train and evaluate churn models without deleting existing artifacts.

set -e
PYTHON="${PYTHON:-python}"

echo "=========================================="
echo "FULL PROJECT RUN"
echo "=========================================="

# 1. Ensure output directories exist; keep previous artifacts if a later step fails.
echo ""
echo "[1/3] Preparing output directories..."
mkdir -p models models/plots

# 2. Training and fold-local preprocessing
echo ""
echo "[2/3] Running model_training.py..."
"$PYTHON" src/model_training.py
echo "model_training.py completed"

# 3. Model evaluation
echo ""
echo "[3/3] Running model_evaluation.py..."
"$PYTHON" src/model_evaluation.py
echo "model_evaluation.py completed"

# 5. Final check
echo ""
echo "[4/4] Checking results..."
ls -lh models/*.pkl 2>/dev/null || echo "Models not found"
ls -lh models/*.csv 2>/dev/null || echo "Results not found"

echo ""
echo "=========================================="
echo "PROJECT COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Models saved in: models/"
echo "Results: models/model_results.csv"
echo ""
echo "Start the web application:"
echo "streamlit run app/main.py"