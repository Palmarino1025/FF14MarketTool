import pandas as pd
import traceback
from joblib import load

def predict_next_price_from_model(prices_df):
    try:
        model = load("regression_model.joblib")
        last_row = prices_df.iloc[-1]
        item_id = last_row['ItemID']
        server = last_row['Server']
        last_timestamp = last_row['Timestamp']

        print("\n--- Prediction Debug Info ---")
        print(f"Last sale timestamp: {last_timestamp} -> {pd.to_datetime(last_timestamp, unit='s')}")

        next_date = pd.to_datetime(last_timestamp, unit='s') + pd.Timedelta(days=1)
        next_timestamp = int(next_date.timestamp())

        print(f"Predicted for next day: {next_timestamp} -> {next_date}")
        print("-----------------------------------\n")

        X_pred = pd.DataFrame([{
            'ItemID': item_id,
            'Timestamp': next_timestamp,
            'Server': server
        }])

        predicted_price = model.predict(X_pred)[0]
        return predicted_price

    except Exception as e:
        print("[Prediction Error]")
        traceback.print_exc()
        raise RuntimeError(f"Prediction failed: {e}")