'''
Written by: Robert Palmarino

API Documentation: https://docs.universalis.app/
API's used: https://universalis.app/api/v2/marketable, https://xivapi.com/

'''
import requests
import json
import time
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from config import MODEL_PATH, SCALER_PATH


def fetch_and_save_item_data(output_path="items.json"):
    base_url = "https://xivapi.com/Item"
    page = 1
    item_map = {}

    print("Fetching item data from XIVAPI...")

    while True:
        response = requests.get(f"{base_url}?page={page}")
        if response.status_code != 200:
            print(f"Failed on page {page}")
            break

        data = response.json()
        results = data.get("Results", [])

        for item in results:
            name = item.get("Name")
            item_id = item.get("ID")
            if name and item_id:
                item_map[name] = item_id

        next_page = data.get("Pagination", {}).get("PageNext")
        if not next_page:
            break

        page = next_page
        time.sleep(0.1)  # be polite to the API

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(item_map, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(item_map)} items to '{output_path}'.")


def prepare_timeseries_data(prices, window_size=5):
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_prices) - window_size):
        X.append(scaled_prices[i:i+window_size])
        y.append(scaled_prices[i+window_size])
    return np.array(X), np.array(y), scaler


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def train_or_update_model(prices, window_size=5, epochs=10, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    prices = np.array(prices).reshape(-1, 1)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Model and scaler found at {model_path} and {scaler_path}. Loading...")
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        if len(prices) <= window_size:
            print("Not enough data to train. Skipping training.")
            return model, scaler

        # Refit the scaler on new prices for correct scaling
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(prices)

        X, y = [], []
        for i in range(len(scaled_prices) - window_size):
            X.append(scaled_prices[i:i + window_size])
            y.append(scaled_prices[i + window_size])
        X, y = np.array(X), np.array(y)

        model.fit(X, y, epochs=epochs, verbose=0)

    else:
        print("No model/scaler found. Training new model...")
        X, y, scaler = prepare_timeseries_data(prices, window_size)
        model = create_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=epochs, verbose=0)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return model, scaler


def get_sales_count_for_items(server, item_ids):
    counts = {}
    print(f"Fetching sales counts for {len(item_ids)} items on server '{server}'...")
    for item_id in item_ids:
        url = f"https://universalis.app/api/v2/history/{server}/{item_id}?entries=300"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            counts[item_id] = len(data.get("entries", []))
        else:
            counts[item_id] = 0
        time.sleep(0.05)  # small delay to be polite to API
    return counts


def fetch_prices_for_items(server, item_ids, max_prices=300):
    all_prices = []
    print(f"Fetching up to {max_prices} prices for {len(item_ids)} items on server '{server}'...")
    for item_id in item_ids:
        url = f"https://universalis.app/api/v2/history/{server}/{item_id}?entries={max_prices}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            entries = data.get("entries", [])
            prices = [e["pricePerUnit"] for e in entries]
            all_prices.extend(prices)
        time.sleep(0.05)  # small delay to avoid hammering API
    return all_prices



# Allows the script to be run directly for testing or imported for function call
if __name__ == "__main__":
    fetch_and_save_item_data()

    # Load items from file to get item IDs
    with open("items.json", "r", encoding="utf-8") as f:
        item_map = json.load(f)
    all_item_ids = list(item_map.values())

