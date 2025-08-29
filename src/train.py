import pandas as pd, yaml, json, os, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/interim/train_plus_sample.csv", sep=';')

# Asegúrate de que sample_id y average existen y son numéricos
X = df[["sample_id"]]
y = df["average"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["train"]["test_size"], random_state=params["seed"]
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

os.makedirs("metrics", exist_ok=True)
with open("metrics/train.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f, indent=2)

print(f"[train] MSE: {mse:.4f} | R2: {r2:.4f}")