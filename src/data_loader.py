"""
Функції для завантаження даних Yoochoose
"""
import pandas as pd
import requests
from .config import DATA_DIR


def download_yoochoose():
    """Завантаження датасету Yoochoose"""
    base_url = "https://s3-eu-west-1.amazonaws.com/yc-rdata/"
    files = {
        "yoochoose-clicks.dat": "yds-data/yoochoose-clicks.dat",
        "yoochoose-buys.dat": "yds-data/yoochoose-buys.dat",
        "yoochoose-test.dat": "yds-data/yoochoose-test.dat"
    }

    for filename, path in files.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"· {filename} вже існує, пропускаємо")
            continue

        print(f"Завантаження {filename}...")
        url = base_url + path
        # ДОДАНО: raise_for_status + timeout замість тихого продовження при помилці
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"✓ {filename} завантажено")
        except requests.RequestException as e:
            print(f"✗ Помилка завантаження {filename}: {e}")
            # Видаляємо частково завантажений файл, якщо є
            if filepath.exists():
                filepath.unlink()


def load_clicks():
    """Завантаження clicks даних"""
    return pd.read_csv(
        DATA_DIR / "yoochoose-clicks.dat",
        names=['Session_ID', 'Timestamp', 'Item_ID', 'Category'],
        header=None,
        parse_dates=['Timestamp'],
        dtype={'Session_ID': 'int64', 'Item_ID': 'int64', 'Category': 'str'}
    )


def load_buys():
    """Завантаження buys даних"""
    return pd.read_csv(
        DATA_DIR / "yoochoose-buys.dat",
        names=['Session_ID', 'Timestamp', 'Item_ID', 'Price', 'Quantity'],
        header=None,
        parse_dates=['Timestamp']
    )
