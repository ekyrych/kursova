import tensorflow as tf
from tensorflow.keras import layers, models
from .config import EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE, RECURRENT_DROPOUT, L2_REG


def create_lstm_model(vocab_size, max_len, lstm_units=LSTM_UNITS, dropout=DROPOUT_RATE,
                      recurrent_dropout=RECURRENT_DROPOUT):
    """Створення LSTM моделі"""
    regularizer = tf.keras.regularizers.l2(L2_REG)

    model = models.Sequential([
        layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_len, mask_zero=True),
        layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
                    return_sequences=False),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_bidirectional_lstm_model(vocab_size, max_len, lstm_units=LSTM_UNITS, dropout=DROPOUT_RATE,
                                    recurrent_dropout=RECURRENT_DROPOUT):
    """Створення двонаправленої LSTM"""
    regularizer = tf.keras.regularizers.l2(L2_REG)

    model = models.Sequential([
        layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_len, mask_zero=True),
        layers.Bidirectional(
            layers.LSTM(lstm_units, dropout=dropout, recurrent_dropout=recurrent_dropout)
        ),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_gru_model(vocab_size, max_len, gru_units=LSTM_UNITS, dropout=DROPOUT_RATE,
                     recurrent_dropout=RECURRENT_DROPOUT):
    """Створення GRU моделі"""
    regularizer = tf.keras.regularizers.l2(L2_REG)

    model = models.Sequential([
        layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_len, mask_zero=True),
        layers.GRU(gru_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
                   return_sequences=False),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(dropout),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
