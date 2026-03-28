"""
Функції для тренування моделей
"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, MODELS_DIR

def compile_model(model, learning_rate=LEARNING_RATE):
    """Компіляція моделі"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def get_callbacks(model_name):
    """Отримання callback функцій"""
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            MODELS_DIR / f'{model_name}.h5',
            save_best_only=True,
            monitor='val_auc',
            mode='max',
            verbose=1
        )
    ]
    return callbacks

def train_model(model, X_train, y_train, X_val, y_val, model_name='lstm_model'):
    """Тренування моделі"""
    callbacks = get_callbacks(model_name)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history