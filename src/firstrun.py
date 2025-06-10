from DataAquisition import train_or_update_model

# Dummy prices to initialize
prices = [10000, 10200, 10150, 10400, 10300, 10500, 10600]

model, scaler = train_or_update_model(prices)
print("Model and scaler created and saved.")
