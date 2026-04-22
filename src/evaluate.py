"""
Функції для оцінки моделей
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve
)
from .config import FIGURES_DIR


def evaluate_model(model, X_test, y_test, model_name='model', threshold=0.5):
    """
    Повна оцінка моделі.
    ДОДАНО: параметр threshold — можна передати оптимальний поріг замість 0.5.
    """
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'threshold': threshold
    }

    print(f"\n=== Метрики для {model_name} (threshold={threshold:.3f}) ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, model_name)
    return metrics, y_pred_proba


def find_best_threshold(y_true, y_pred_proba, metric='f1'):
    """
    Пошук оптимального порогу класифікації.
    НОВА ФУНКЦІЯ — корисно при дисбалансі класів, де 0.5 не є оптимальним.

    Parameters
    ----------
    metric : 'f1' | 'youden'
        'f1'     — максимізувати F1-score
        'youden' — Youden's J = TPR - FPR (баланс чутливості та специфічності)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    if metric == 'f1':
        # F1 для кожного порогу (уникаємо ділення на 0)
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0
        )
        # thresholds має довжину на 1 менше за precision/recall
        best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
        best_threshold = thresholds[best_idx] if len(thresholds) > 0 else 0.5
        best_score = f1_scores[best_idx]
        print(f"Оптимальний поріг за F1: {best_threshold:.4f} (F1={best_score:.4f})")
    else:
        raise ValueError(f"Невідома метрика: {metric}")

    return float(best_threshold), float(best_score)


def plot_precision_recall_curve(y_true, y_pred_proba, model_name='model'):
    """НОВА ФУНКЦІЯ: Precision-Recall крива — інформативніша за ROC при дисбалансі."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'pr_curve_{model_name}.png', dpi=100, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Побудова матриці помилок"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'cm_{model_name}.png', dpi=100, bbox_inches='tight')
    plt.show()


def plot_training_history(history, model_name='model'):
    """Візуалізація історії тренування"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title(f'Model Loss - {model_name}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_title(f'Model Accuracy - {model_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'training_history_{model_name}.png', dpi=100, bbox_inches='tight')
    plt.show()
