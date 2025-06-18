import pandas as pd

def predict_next_price_from_model(prices_df, model):
    last_row = prices_df.iloc[-1]
    next_date = pd.to_datetime(last_row['dateOfSale']) + pd.Timedelta(days=1)
    timestamp = next_date.value // 10**9  # UNIX timestamp

    X_pred = pd.DataFrame([{
        'ItemID': last_row['itemID'],
        'Server': last_row['serverID'],
        'Timestamp': timestamp
    }])

    predicted_price = model.predict(X_pred)[0]
    return predicted_price