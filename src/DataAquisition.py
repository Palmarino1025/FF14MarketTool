'''
Written by: Robert Palmarino
DataAquisition

API Documentation: https://docs.universalis.app/
API's used: https://universalis.app/api/v2/marketable, https://xivapi.com/

'''
import requests
import json
import time
import os
import pandas as pd
import numpy as np
from prediction_util import predict_next_price_from_model
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def fetch_and_save_item_data(output_path="items.json"):
    base_url = "https://xivapi.com/Item"
    page = 1
    item_map = {}

    print("Fetching marketable item IDs from Universalis...")
    marketable_resp = requests.get("https://universalis.app/api/v2/marketable")
    if marketable_resp.status_code != 200:
        print("Failed to fetch marketable items.")
        return
    marketable_ids = set(marketable_resp.json())

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
            if name and item_id and item_id in marketable_ids:
                item_map[name] = item_id

        next_page = data.get("Pagination", {}).get("PageNext")
        if not next_page:
            break

        page = next_page
        time.sleep(0.1)  # be polite to the API

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(item_map, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(item_map)} marketable items to '{output_path}'.")

def train_and_save_model(server_name="Leviathan", item_id=5057):
    print(f"Fetching sales data for item {item_id} on {server_name}...")
    df = fetch_top_sales_data(server_name, item_id, sales_limit=300)

    if df.empty:
        print("No data fetched, aborting training.")
        return

    print("\n--- Training Timestamp Range ---")
    print(f"Min: {df['Timestamp'].min()} -> {pd.to_datetime(df['Timestamp'].min(), unit='s')}")
    print(f"Max: {df['Timestamp'].max()} -> {pd.to_datetime(df['Timestamp'].max(), unit='s')}")
    print("--------------------------------\n")

    # Features and target
    X = df[["ItemID", "Server", "Timestamp"]]
    y = df["Price"]

    X["Timestamp"] = pd.to_numeric(X["Timestamp"])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore'), ["Server"]),
            ("num", "passthrough", ["ItemID", "Timestamp"])
        ]
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    print("Training model...")
    model.fit(X, y)

    model_path = os.path.join(os.path.dirname(__file__), "regression_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def fetch_top_sales_data(server_name: str, item_id: int, sales_limit: int = 300) -> pd.DataFrame:
    history_url = f"https://universalis.app/api/v2/history/{server_name}/{item_id}?entries={sales_limit}"

    print(history_url)
    print(f"Fetching sales for item {item_id} on {server_name}...")

    try:
        history_resp = requests.get(history_url)
        if history_resp.status_code != 200:
            print(f"[Warning] Failed to fetch data for {item_id} from {server_name}")
            return pd.DataFrame()

        sales = history_resp.json().get("entries", [])[:sales_limit]

        sales_records = []
        for sale in sales:
            timestamp = datetime.fromtimestamp(sale['timestamp'])
            sales_records.append({
                "ItemID": int(item_id),
                "Price": sale["pricePerUnit"],
                "Timestamp": sale["timestamp"],
                "Day": timestamp.strftime("%Y-%m-%d"),
                "Time": timestamp.strftime("%H:%M:%S"),
                "Server": server_name
            })

        return pd.DataFrame(sales_records)

    except Exception as e:
        print(f"[Error] Exception fetching data: {e}")
        return pd.DataFrame()


# Allows the script to be run directly for testing or imported for function call
if __name__ == "__main__":
    train_and_save_model(server_name="Leviathan", item_id=5069)
    df = fetch_top_sales_data(server_name="Leviathan", item_id=5069)
    print(predict_next_price_from_model(df))


