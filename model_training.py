import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np


def train_model(X, y, actions):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(actions.shape[0], activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=2000,
              callbacks=[TensorBoard(log_dir='logs')])

    return model


if __name__ == "__main__":
    # Load your data and train the model
    pass
