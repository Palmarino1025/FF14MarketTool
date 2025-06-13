import os

# Get the directory of the current file (src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to FF14MarketTool
BASE_DIR = os.path.dirname(SRC_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "ffxiv_model.keras")      # Replace with your actual model filename
SCALER_PATH = os.path.join(BASE_DIR, "ffxiv_model.keras_scaler.pkl")   # Replace with your actual scaler filename
