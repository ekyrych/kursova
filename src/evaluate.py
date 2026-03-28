"""
Функції для оцінки моделей
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from .config import FIGURES_DIR

def evaluate_model(model, X_test, y_test, model_name='model'):
    """Повна оцінка моделі"""
    # Передбачення
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Метрики
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print(f"\n=== Метрики для {model_name} ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    return metrics, y_pred_proba

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
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Validation')
    axes[0].set_title(f'Model Loss - {model_name}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Validation')
    axes[1].set_title(f'Model Accuracy - {model_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'training_history_{model_name}.png', dpi=100, bbox_inches='tight')
    plt.show()