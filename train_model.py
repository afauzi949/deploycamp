import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("CarPrice_Assignment.csv")
X = df[["horsepower", "curbweight", "enginesize"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
mae = abs(y_test - model.predict(X_test)).mean()

with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, "model")
