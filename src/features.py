"""
Створення агрегованих ознак для baseline (оптимізовано)
"""
import pandas as pd
import numpy as np

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
        session_duration_seconds=('Timestamp', lambda x: (x.max() - x.min()).total_seconds() if len(x) > 1 else 0),
        first_click_hour=('Timestamp', lambda x: x.min().hour),
        last_click_hour=('Timestamp', lambda x: x.max().hour)
    ).reset_index()
    
    # 2. ОЗНАКИ ПОКУПОК
    if len(buys) > 0:
        purchase_features = buys.groupby('Session_ID').agg(
            has_purchase=('Session_ID', 'size'),
            num_purchases=('Session_ID', 'size'),
            total_price=('Price', 'sum') if 'Price' in buys.columns else None,
            avg_price=('Price', 'mean') if 'Price' in buys.columns else None
        ).reset_index()
        
        # Перейменовуємо
        purchase_features = purchase_features.rename(columns={'has_purchase': 'has_purchase_temp'})
        purchase_features['has_purchase'] = 1
        purchase_features = purchase_features.drop('has_purchase_temp', axis=1)
        
        # Об'єднуємо
        features = features.merge(purchase_features, on='Session_ID', how='left')
    else:
        features['has_purchase'] = 0
        features['num_purchases'] = 0
        if 'total_price' in features.columns:
            features['total_price'] = 0
        if 'avg_price' in features.columns:
            features['avg_price'] = 0
    
    # Заповнюємо NaN
    features['has_purchase'] = features['has_purchase'].fillna(0).astype(int)
    features['num_purchases'] = features['num_purchases'].fillna(0).astype(int)
    
    # 3. ДОДАТКОВІ ОЗНАКИ
    # Частота кліків (кліки/хвилину)
    features['click_frequency'] = np.where(
        features['session_duration_seconds'] > 0,
        features['session_length'] / (features['session_duration_seconds'] / 60),
        0
    )
    
    # Різноманітність (унікальні товари / загальна кількість)
    features['diversity'] = features['unique_items'] / features['session_length']
    
    # Повторюваність (1 - різноманітність)
    features['repeat_ratio'] = 1 - features['diversity']
    
    # Чи є повторення товарів
    features['has_repeats'] = (features['unique_items'] < features['session_length']).astype(int)
    
    # Чи були перегляди в різних категоріях
    features['multi_category'] = (features['unique_categories'] > 1).astype(int)
    
    return features