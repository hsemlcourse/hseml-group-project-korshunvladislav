[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Прогнозирование пожизненной ценности клиента (CLV)

**Студент:** Коршун В.

**Группа:** БИВ 237


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Регрессия — прогнозирование пожизненной ценности клиента (Customer Lifetime Value, CLV) на основе поведенческих и демографических признаков.

**Датасет:** [Ecommerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset) с Kaggle (50 000 записей, 25 признаков).

**Целевая метрика:** RMSE (основная), дополнительные — MAE и R².


## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   └── cp1.ipynb               # Основной ноутбук CP1 (очистка, EDA, модели)
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-korshunvladislav.git
cd hseml-group-project-korshunvladislav

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt

# 4. Скачать датасет и поместить в data/raw

# 5. Запустить ноутбук
jupyter notebook notebooks/cp1.ipynb
```

## Данные
- `data/raw/` — исходный файл ecommerce_customer_churn_data.csv (скачан с Kaggle)
- `data/processed/` — пока не используется, все преобразования выполняются в ноутбуке


## Результаты
Здесь коротко выпишите результаты.
| Модель | RMSE (val) | MAE (val) | R² (val) | Примечание |
|--------|------------|-----------|----------|------------|
| Baseline (Linear Regression, исходные признаки) | 350.23 | 236.26 | 0.853 | без feature engineering |
| Linear Regression (все признаки) | 348.02 | 235.34 | 0.855 | 	с новыми фичами |
| Ridge Regression (alpha=1.0) | 348.02 | 235.34 | 0.855 | 	регуляризация |
| Lasso Regression (alpha=1.0) | 348.07 | 235.17 | 0.855 | 	регуляризация |
| **Random Forest (n=100)** | **233.56** | **153.86** | **0.935** | **лучшая модель** |
| Random Forest (n=50) | 235.10 | 154.88 | 0.934 | быстрее, чуть хуже |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
