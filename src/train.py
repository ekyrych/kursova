"""
Функції для тренування моделей
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODELS_DIR, USE_CLASS_WEIGHTS
from .preprocessing import compute_class_weights


def compile_model(model, learning_rate=LEARNING_RATE):
    """Компіляція моделі"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model


def get_callbacks(model_name):
    """Отримання callback функцій. Всі моніторять val_auc."""
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            str(MODELS_DIR / f'{model_name}.keras'),
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1,
        ),
    ]
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val,
                model_name='lstm_model', use_class_weights=USE_CLASS_WEIGHTS):
    """
    Тренування моделі.

    Parameters
    ----------
    use_class_weights : bool
        Якщо True — автоматично обчислює ваги класів для боротьби з дисбалансом.
        Значення за замовчуванням береться з config.USE_CLASS_WEIGHTS.
    """
    callbacks = get_callbacks(model_name)
    class_weight = compute_class_weights(y_train) if use_class_weights else None

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    return history
