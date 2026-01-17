# train_and_save_preds.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from preprocess import load_all_data


def build_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("Loading data...")
    X, y = load_all_data()
    print(f"Total samples: {X.shape[0]}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    model = build_model((X.shape[1], X.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32, verbose=1)

    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nAccuracy:", np.mean(y_pred == y_val))
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Wake', 'NREM', 'REM']))

    model.save("sleep_model.h5")


    np.savetxt("predictions.csv", np.column_stack([y_val, y_pred]), delimiter=",", header="true,pred", comments="")
    print("Predictions saved to predictions.csv")


if __name__ == "__main__":
    main()