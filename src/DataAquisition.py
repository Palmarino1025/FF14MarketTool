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

def train_and_save_model(server_name="Leviathan"):
    print(f"Fetching sales data for {server_name}...")
    df = fetch_top_sales_data(server_name)

    if df.empty:
        print("No data fetched, aborting training.")
        return

    # Features and target
    X = df[["ItemID", "Server", "Timestamp"]]
    y = df["Price"]

    # We will encode Server as categorical, ItemID as categorical or numeric, Timestamp numeric
    # Convert Timestamp to numeric (e.g. seconds since epoch)
    X["Timestamp"] = pd.to_numeric(X["Timestamp"])

    # Create preprocessing pipeline
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

    # Save the model to disk
    model_path = os.path.join(os.path.dirname(__file__), "linear_regression_model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def fetch_top_sales_data(server_name: str, top_n: int = 100, sales_limit: int = 50) -> pd.DataFrame:
    with open('items.json', "r", encoding="utf-8") as f:
        items_dict = json.load(f)

    item_ids = list(items_dict.values())

    sales_records = []

    for item_id in item_ids:
        history_url = f"https://universalis.app/api/v2/{server_name}/{item_id}"
        history_resp = requests.get(history_url)
        if history_resp.status_code != 200:
            print(f"[Warning] Skipping item {item_id} due to fetch error.")
            continue

        sales = history_resp.json().get("recentHistory", [])[:sales_limit]
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

# Allows the script to be run directly for testing or imported for function call
if __name__ == "__main__":
    fetch_and_save_item_data()

    # Load items from file to get item IDs
    with open("items.json", "r", encoding="utf-8") as f:
        item_map = json.load(f)
    all_item_ids = list(item_map.values())
