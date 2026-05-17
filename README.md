[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Прогнозирование пожизненной ценности клиента (CLV)

**Студент:** Коршун В.И.

**Группа:** БИВ237


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Регрессия — прогнозирование пожизненной ценности клиента (Customer Lifetime Value, CLV) на основе поведенческих и демографических признаков.

**Датасет:** [Ecommerce Customer Behavior Dataset](https://www.kaggle.com/datasets/dhairyajeetsingh/ecommerce-customer-behavior-dataset) с Kaggle (50000 записей, 25 признаков).

**Целевая метрика:** RMSE (основная), дополнительные — MAE и R².


## Структура репозитория
```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
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

### Локальный запуск (без Docker)

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

# 4. Запустить Jupyter Notebook
jupyter notebook notebooks/
```

### Запуск через Docker

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-korshunvladislav.git
cd hseml-group-project-korshunvladislav

# 2. Собрать образ
docker-compose build

# 3. Запустить контейнер
docker-compose up

# 4. Получить ссылку на запущенный Jupyter
jupyter server list

# 5. Открыть в браузере нужную ссылку из списка
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты

| Модель | RMSE | MAE | R2 | Примечание |
|--------|------|-----|----|------------|
| Baseline (Lasso) | 350.129535 | 236.068940 | 0.853155 | Линейная модель без feature engineering, только исходные закодированные признаки |
| **CatBoost default** | **225.103129** | **151.461374** | **0.939303** | **Лучшая модель** – после feature engineering и кодирования категорий, без перебора гиперпараметров |

**Вывод:** CatBoost default показал снижение RMSE на **36%** по сравнению с Baseline Lasso, что объясняется способностью градиентного бустинга учитывать нелинейные зависимости и взаимодействия признаков. Тестовая оценка CatBoost: RMSE = 224.85, MAE = 150.92, R2 = 0.940 (стабильность подтверждена).

> **Примечание:** Модель CatBoost обучена на данных после полной очистки, feature engineering (54 новых признака) и one‑hot кодирования. Гиперпараметры оставлены по умолчанию, так как уже дали отличный результат. Дополнительный перебор (RandomizedSearchCV) для XGBoost и LightGBM не улучшил качество относительно CatBoost.


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
