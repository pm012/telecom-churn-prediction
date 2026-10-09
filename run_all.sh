#!/bin/bash
# run_all.sh - Повний запуск проекту

echo "=========================================="
echo "FULL PROJECT RUN"
echo "=========================================="

# 1. Cleaning old models
echo ""
echo "[1/5] Cleaning old models..."
rm -rf models/
mkdir -p models
echo "Models folder cleared"

# 2. Data preprocessing
echo ""
echo "[2/5] Running data_preprocessing.py..."
python src/data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Error in data_preprocessing.py"
    exit 1
fi
echo "data_preprocessing.py completed"

# 3. Training models
echo ""
echo "[3/5] Running model_training.py..."
python src/model_training.py
if [ $? -ne 0 ]; then
    echo "Error in model_training.py"
    exit 1
fi
echo "model_training.py completed"

# 4. Model evaluation
echo ""
echo "[4/5] Running model_evaluation.py..."
python src/model_evaluation.py
if [ $? -ne 0 ]; then
    echo "Error in model_evaluation.py"
    exit 1
fi
echo "model_evaluation.py completed"

# 5. Final check
echo ""
echo "[5/5] Checking results..."
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