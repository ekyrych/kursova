"""
Створення агрегованих ознак для baseline (оптимізовано)
"""
import numpy as np
import pandas as pd


def extract_session_features(clicks, buys):
    """
    Створення ознак для кожної сесії (векторизована версія)

    Parameters
    ----------
    clicks : pd.DataFrame
        Дані кліків з колонками: Session_ID, Timestamp, Item_ID, Category
    buys : pd.DataFrame
        Дані покупок з колонками: Session_ID, Timestamp, Item_ID, Price, Quantity

    Returns
    -------
    pd.DataFrame
        Таблиця з ознаками для кожної сесії
    """

    # 1. ОСНОВНІ ОЗНАКИ (групуванням)
    features = clicks.groupby('Session_ID').agg(
        session_length=('Item_ID', 'count'),
        unique_items=('Item_ID', 'nunique'),
        unique_categories=('Category', 'nunique'),
        session_duration_seconds=(
            'Timestamp',
            lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0
        ),
        first_click_hour=('Timestamp', lambda x: x.min().hour),
        last_click_hour=('Timestamp', lambda x: x.max().hour)
    ).reset_index()

    # 2. ОЗНАКИ ПОКУПОК
    # ВИПРАВЛЕНО: попередня версія передавала None у agg() якщо 'Price' не було —
    # це падало з TypeError. Тепер будуємо dict умовно.
    if len(buys) > 0:
        agg_dict = {'num_purchases': ('Session_ID', 'size')}
        if 'Price' in buys.columns:
            agg_dict['total_price'] = ('Price', 'sum')
            agg_dict['avg_price'] = ('Price', 'mean')

        purchase_features = buys.groupby('Session_ID').agg(**agg_dict).reset_index()
        purchase_features['has_purchase'] = 1

        # Об'єднуємо
        features = features.merge(purchase_features, on='Session_ID', how='left')
    else:
        features['has_purchase'] = 0
        features['num_purchases'] = 0
        features['total_price'] = 0.0
        features['avg_price'] = 0.0

    # Заповнюємо NaN після merge
    features['has_purchase'] = features['has_purchase'].fillna(0).astype(int)
    features['num_purchases'] = features['num_purchases'].fillna(0).astype(int)
    if 'total_price' in features.columns:
        features['total_price'] = features['total_price'].fillna(0.0)
    if 'avg_price' in features.columns:
        features['avg_price'] = features['avg_price'].fillna(0.0)

    # 3. ДОДАТКОВІ ОЗНАКИ
    # Частота кліків (кліки/хвилину)
    features['click_frequency'] = np.where(
        features['session_duration_seconds'] > 0,
        features['session_length'] / (features['session_duration_seconds'] / 60),
        0
    )

    # Різноманітність (унікальні товари / загальна кількість)
    # ДОДАНО: захист від ділення на 0
    features['diversity'] = np.where(
        features['session_length'] > 0,
        features['unique_items'] / features['session_length'],
        0
    )

    # Повторюваність (1 - різноманітність)
    features['repeat_ratio'] = 1 - features['diversity']

    # Чи є повторення товарів
    features['has_repeats'] = (features['unique_items'] < features['session_length']).astype(int)

    # Чи були перегляди в різних категоріях
    features['multi_category'] = (features['unique_categories'] > 1).astype(int)

    return features
