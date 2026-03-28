"""
Всі шляхи та гіперпараметри в одному місці
"""
from pathlib import Path

# Шляхи
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'yoochoose'
MODELS_DIR = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'

# Створення папок, якщо не існують (ВИПРАВЛЕНО)
for dir_path in [DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)  # exist_ok=True дозволяє пропустити якщо папка вже є

# Параметри даних
MAX_SEQUENCE_LENGTH = 50  # максимальна довжина сесії
MIN_SEQUENCE_LENGTH = 2    # мінімальна довжина сесії
EMBEDDING_DIM = 50         # розмір ембеддінгів

# Параметри моделей
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

# Випадкове зерно для відтворюваності
RANDOM_SEED = 42