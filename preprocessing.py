"""
Функції для підготовки даних
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .config import MAX_SEQUENCE_LENGTH, RANDOM_SEED

def create_sessions(clicks):
    """Створення сесій з кліків"""
    sessions = clicks.groupby('Session_ID').agg({
        'Item_ID': list,
        'Timestamp': list,
        'Category': list
    }).reset_index()
    
    sessions['session_length'] = sessions['Item_ID'].apply(len)
    return sessions

def encode_items(clicks):
    """Кодування товарів"""
    le = LabelEncoder()
    clicks['Item_Encoded'] = le.fit_transform(clicks['Item_ID']) + 1  # +1 для резерву 0 (padding)
    return clicks, le

def pad_sequences(sequences, max_len=MAX_SEQUENCE_LENGTH):
    """Вирівнювання послідовностей до однакової довжини"""
    padded = np.zeros((len(sequences), max_len), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > max_len:
            padded[i] = seq[:max_len]  # обрізаємо
        else:
            padded[i, :len(seq)] = seq  # доповнюємо нулями
    return padded

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Розбиття на train/val/test"""
    assert train_ratio + val_ratio + test_ratio == 1
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_ratio), random_state=RANDOM_SEED
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=RANDOM_SEED
    )
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)