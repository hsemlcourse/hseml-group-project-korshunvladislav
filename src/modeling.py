"""
Модуль обучения и оценки моделей.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import joblib
import os

SEED = 42

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name="Model"):
    """Обучает модель и возвращает метрики на валидации."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"{model_name:45} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R2: {r2:.4f}")
    return {'name': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'model': model}

def train_baseline_models(X_train, y_train, X_val, y_val):
    """Обучает линейные модели (Baseline, Ridge, Lasso) и возвращает словарь с метриками."""
    results = []
    lr = LinearRegression()
    results.append(evaluate_model(lr, X_train, y_train, X_val, y_val, "Linear Regression"))

    ridge = Ridge(alpha=1.0, random_state=SEED)
    results.append(evaluate_model(ridge, X_train, y_train, X_val, y_val, "Ridge (alpha=1.0)"))

    lasso = Lasso(alpha=1.0, random_state=SEED)
    results.append(evaluate_model(lasso, X_train, y_train, X_val, y_val, "Lasso (alpha=1.0)"))

    return results

def tune_random_forest(X_train, y_train, X_val, y_val, n_iter=10):
    """Перебор гиперпараметров Random Forest через RandomizedSearchCV."""
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=SEED, n_jobs=-1)
    search = RandomizedSearchCV(rf, param_dist, n_iter=n_iter, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RF tuned RMSE: {rmse:.2f}, best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_

def tune_xgboost(X_train, y_train, X_val, y_val, n_iter=10):
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_model = xgb.XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0)
    search = RandomizedSearchCV(xgb_model, param_dist, n_iter=n_iter, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"XGBoost tuned RMSE: {rmse:.2f}, best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_

def tune_lightgbm(X_train, y_train, X_val, y_val, n_iter=10):
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100]
    }
    lgb_model = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, verbose=-1)
    search = RandomizedSearchCV(lgb_model, param_dist, n_iter=n_iter, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=SEED, n_jobs=-1)
    search.fit(X_train, y_train)
    y_pred = search.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"LightGBM tuned RMSE: {rmse:.2f}, best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_

def train_catboost_default(X_train, y_train, X_val, y_val):
    cat = CatBoostRegressor(random_seed=SEED, verbose=0, iterations=200)
    cat.fit(X_train, y_train)
    y_pred = cat.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"CatBoost default RMSE: {rmse:.2f}")
    return cat

def save_model(model, filepath='models/'):
    """Сохраняет модель в папку models."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Модель сохранена в {filepath}")
