#!/bin/bash
# run_all.sh - Повний запуск проекту

echo "=========================================="
echo "ПОВНИЙ ЗАПУСК ПРОЕКТУ"
echo "=========================================="

# 1. Очищення
echo ""
echo "[1/5] Очищення старих моделей..."
rm -rf models/
mkdir -p models
echo "Папку models очищено"

# 2. Обробка даних
echo ""
echo "[2/5] Запуск data_preprocessing.py..."
python src/data_preprocessing.py
if [ $? -ne 0 ]; then
    echo "Помилка в data_preprocessing.py"
    exit 1
fi
echo "data_preprocessing.py виконано"

# 3. Навчання моделей
echo ""
echo "[3/5] Запуск model_training.py..."
python src/model_training.py
if [ $? -ne 0 ]; then
    echo "Помилка в model_training.py"
    exit 1
fi
echo "model_training.py виконано"

# 4. Оцінка моделей
echo ""
echo "[4/5] Запуск model_evaluation.py..."
python src/model_evaluation.py
if [ $? -ne 0 ]; then
    echo "Помилка в model_evaluation.py"
    exit 1
fi
echo "model_evaluation.py виконано"

# 5. Фінальна перевірка
echo ""
echo "[5/5] Перевірка результатів..."
ls -lh models/*.pkl 2>/dev/null || echo "Моделі не знайдено"
ls -lh models/*.csv 2>/dev/null || echo "Результати не знайдено"

echo ""
echo "=========================================="
echo "ПРОЕКТ УСПІШНО ЗАВЕРШЕНО!"
echo "=========================================="
echo ""
echo "Моделі збережено в: models/"
echo "Результати: models/model_results.csv"
echo ""
echo "Запустіть веб-додаток:"
echo "streamlit run app/main.py"