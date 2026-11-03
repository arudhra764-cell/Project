from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("housing.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Please add housing.csv to the repository.")

    data = pd.read_csv(DATA_PATH)

    if "price" not in data.columns:
        raise ValueError("Dataset must contain a 'price' column.")

    X = data.drop(columns=["price"])
    y = data["price"]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42
        )
    }

    results = {}
    best_model = None
    best_mae = float("inf")
    best_name = ""

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        results[name] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4)
        }

        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_name = name

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    metrics = {
        "best_model": best_name,
        "results": results
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=4))

    print("Model training completed!")
    print("Best model:", best_name)
    print("Results:", results)

if __name__ == "__main__":
    main()
