"""
Функції для підготовки даних
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
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
    """Розбиття на train/val/test з стратифікацією за класом"""
    # ВИПРАВЛЕНО: надійне порівняння float сум замість ==
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6, f"Сума ratio має дорівнювати 1, зараз {total}"

    # ДОДАНО: stratify=y — щоб класи пропорційно розподілялись між train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(1 - train_ratio),
        random_state=RANDOM_SEED,
        stratify=y
    )

    test_size = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=y_temp
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def compute_class_weights(y_train):
    """
    Обчислення ваг класів для боротьби з дисбалансом.

    Повертає dict у форматі {0: w0, 1: w1}, готовий до передачі у model.fit(class_weight=...).
    """
    y_arr = np.asarray(y_train).ravel()
    classes = np.unique(y_arr)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_arr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"Ваги класів: {cw}")
    return cw
