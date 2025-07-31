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

# --- 1. Load .env (jika dipakai) ---
load_dotenv()

# --- 2. Setup Kaggle credentials ---
os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")

# --- 3. Download dataset dari Kaggle jika belum ada ---
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

# --- 5. Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- 6. Train model ---
model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
mae = abs(y_test - preds).mean()

# --- 7. Log ke MLflow ---
with mlflow.start_run() as run:
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mae)

    # Log model + signature + contoh input
    signature = infer_signature(X_test, preds)
    mlflow.sklearn.log_model(
        model, "model",
        signature=signature,
        input_example=X_test.iloc[:1]
    )

    # --- 8. Register model ke MLflow Model Registry ---
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    model_name = "car_price_model"

    client = MlflowClient()

    # Jika belum ada, buat model registry-nya
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)

    client.create_model_version(model_name, model_uri, run_id=run_id)

    print(f"✅ Model berhasil dilog & diregister ke MLflow Registry dengan MAE: {mae:.2f}")
