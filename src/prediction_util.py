import numpy as np
import tensorflow as tf
import joblib
import os
import config

def load_model_and_scaler():
    model_path = config.MODEL_PATH
    scaler_path = config.SCALER_PATH
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing model or scaler file.")
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_next_price_from_model(prices, model, scaler, window_size=5):
    if len(prices) <= window_size:
        return None

    scaled_prices = scaler.transform(np.array(prices).reshape(-1, 1))
    input_seq = scaled_prices[-window_size:].reshape(1, window_size, 1)
    pred_scaled = model.predict(input_seq)
    predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

    return predicted_price
