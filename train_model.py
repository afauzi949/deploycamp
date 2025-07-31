import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# --- 1. Load Environment ---
load_dotenv()

# --- 2. Setup Kaggle config ---
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")

# --- 3. Download dataset jika belum ada ---
DATA_PATH = "data/CarPrice_Assignment.csv"
if not os.path.exists(DATA_PATH):
    print("🔽 Downloading dataset dari Kaggle...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('hellbuoy/car-price-prediction', path='data', unzip=True)

# --- 4. Load dataset ---
df = pd.read_csv(DATA_PATH)
X = df[["horsepower", "curbweight", "enginesize"]]
y = df["price"]

# --- 5. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- 6. Train model ---
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = abs(y_test - preds).mean()

# --- 7. MLflow logging ---
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mae)

    # Log model dengan signature & input example
    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(
        model, "model",
        signature=signature,
