import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Fake data
df = pd.DataFrame({
    'ItemID': [101, 102, 103, 104],
    'Server': ['Leviathan', 'Leviathan', 'Cactuar', 'Cactuar'],
    'Timestamp': [1609459200, 1609545600, 1609632000, 1609718400],
    'Price': [100, 110, 95, 105]
})

X = df[['ItemID', 'Server', 'Timestamp']]
y = df['Price']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Server']),
        ('num', 'passthrough', ['ItemID', 'Timestamp'])
    ]
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# Predict next day for last record
last_row = df.iloc[-1]
next_timestamp = last_row['Timestamp'] + 86400  # add 1 day in seconds

X_pred = pd.DataFrame([{
    'ItemID': last_row['ItemID'],
    'Server': last_row['Server'],
    'Timestamp': next_timestamp
}])

predicted_price = model.predict(X_pred)[0]
print(f"Predicted price: {predicted_price}")
