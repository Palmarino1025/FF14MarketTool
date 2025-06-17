import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_linear_model(df):
    if df.empty:
        raise ValueError("DataFrame is empty, cannot train model.")

    # Ensure required columns exist
    if 'Timestamp' not in df or 'Price' not in df:
        raise ValueError("DataFrame must contain 'Timestamp' and 'Price' columns.")

    X = np.array(df['Timestamp']).reshape(-1, 1)
    y = np.array(df['Price'])

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_next_price_from_model(prices_df, model, le_item, le_server):
    last_row = prices_df.iloc[-1]
    itemID_enc = le_item.transform([last_row['itemID']])[0]
    serverID_enc = le_server.transform([last_row['serverID']])[0]
    next_date = pd.to_datetime(last_row['dateOfSale']) + pd.Timedelta(days=1)
    dateOfSale_num = next_date.value // 10**9

    X_pred = pd.DataFrame([{
        'itemID_enc': itemID_enc,
        'dateOfSale_num': dateOfSale_num,
        'serverID_enc': serverID_enc
    }])

    predicted_price = model.predict(X_pred)[0]
    return predicted_price
