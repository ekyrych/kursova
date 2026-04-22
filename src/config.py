"""
Всі шляхи та гіперпараметри в одному місці
"""
from pathlib import Path

# Шляхи
# ВИПРАВЛЕНО: .parent.parent йшло на рівень ВИЩЕ теки проекту.
# Тепер BASE_DIR = тека, в якій лежить config.py (корінь проекту).
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'yoochoose'
MODELS_DIR = BASE_DIR / 'models'
FIGURES_DIR = BASE_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'

# Створення папок, якщо не існують
for dir_path in [DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Параметри даних
MAX_SEQUENCE_LENGTH = 20   # було 9 — обрізало багато сесій. 20 покриває ~90%+
MIN_SEQUENCE_LENGTH = 2    # мінімальна довжина сесії
EMBEDDING_DIM = 50         # розмір ембеддінгів

# Параметри моделей
LSTM_UNITS = 64
DROPOUT_RATE = 0.4         # було 0.3 — збільшено для боротьби з перенавчанням
RECURRENT_DROPOUT = 0.2    # новий: dropout всередині LSTM
L2_REG = 1e-5              # новий: L2-регуляризація на Dense-шарах
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

# Параметри боротьби з дисбалансом класів
USE_CLASS_WEIGHTS = True   # автоматичне зважування класів у .fit()

# Випадкове зерно для відтворюваності
RANDOM_SEED = 42
