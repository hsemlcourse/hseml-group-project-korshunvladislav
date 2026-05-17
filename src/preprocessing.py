import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

SEED = 42

def load_data(raw_path):
    """Загружает сырые данные и возвращает копию DataFrame."""
    df = pd.read_csv(raw_path)
    df = df.copy()
    return df

def clean_data(df):
    """
    Очистка данных:
    - заполнение пропусков (числовые -> медиана, категориальные -> мода)
    - удаление дубликатов
    - ограничение выбросов 99-м перцентилем для непрерывных признаков
    - приведение категориальных колонок к типу 'category'
    """
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Lifetime_Value' in numeric_cols:
        numeric_cols.remove('Lifetime_Value')
    continuous_cols = [col for col in numeric_cols if df[col].nunique() >= 10]
    for col in continuous_cols:
        cap = df[col].quantile(0.99)
        df[col] = np.where(df[col] > cap, cap, df[col])

    categorical_cols = ['Gender', 'Country', 'City', 'Signup_Quarter', 'Churned']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

def feature_engineering(df, target_col='Lifetime_Value', drop_id_cols=True):
    """
    Создаёт новые признаки, кодирует категориальные переменные.
    Возвращает X (признаки), y (целевая), а также оригинальные признаки до one-hot.
    """
    if drop_id_cols and 'Customer_ID' in df.columns:
        X_raw = df.drop(columns=[target_col, 'Customer_ID'])
    else:
        X_raw = df.drop(columns=[target_col])
    y = df[target_col].copy()

    X = X_raw.copy()

    # Новые признаки
    if 'Login_Frequency' in X.columns and 'Session_Duration_Avg' in X.columns:
        X['Login_Session_Product'] = X['Login_Frequency'] * X['Session_Duration_Avg']
    if 'Days_Since_Last_Purchase' in X.columns:
        X['Recency_Score'] = 1 / (X['Days_Since_Last_Purchase'] + 1)
    if 'Product_Reviews_Written' in X.columns and 'Social_Media_Engagement_Score' in X.columns:
        X['Engagement_Index'] = X['Product_Reviews_Written'] + X['Social_Media_Engagement_Score']
    if 'Returns_Rate' in X.columns and 'Total_Purchases' in X.columns:
        X['Returns_Per_Purchase'] = X['Returns_Rate'] / (X['Total_Purchases'] + 1)
    if 'Age' in X.columns:
        X['Age_Group'] = pd.cut(X['Age'], bins=[0, 25, 35, 50, 100],
                                labels=['Young', 'Adult', 'Middle', 'Senior'])

    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y[X.index]

    return X, y, X_raw

def split_data(X, y, test_size=0.15, val_size=0.1765):
    """
    Делит данные на train, val, test.
    Для статических данных используется случайный сплит.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X, y, output_dir='data/processed'):
    """Сохраняет обработанные признаки и целевую переменную в CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X).to_csv(f'{output_dir}/X_processed.csv', index=False)
    pd.Series(y).to_csv(f'{output_dir}/y_processed.csv', index=False)
    print(f"Сохранено в {output_dir}/")

def load_processed_data(input_dir='data/processed'):
    """Загружает обработанные данные."""
    X = pd.read_csv(f'{input_dir}/X_processed.csv')
    y = pd.read_csv(f'{input_dir}/y_processed.csv')
    y = y.iloc[:, 0]
    return X, y
