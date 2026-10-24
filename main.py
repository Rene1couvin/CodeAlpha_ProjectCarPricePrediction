
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    # Load and clean the dataset
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    numeric_columns = [
        "Year", "Selling_Price", "Present_Price",
        "Driven_kms", "Owner"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Selling_Price", "Year"])
    df = df[df["Selling_Price"] >= 0]
    df["Car_Age"] = pd.Timestamp.now().year - df["Year"]

    features = [
        "Car_Name", "Car_Age", "Present_Price",
        "Driven_kms", "Fuel_Type", "Selling_type",
        "Transmission", "Owner"
    ]
    target = "Selling_Price"

    X = df[features]
    y = df[target]

    numeric_features = [
        "Car_Age", "Present_Price", "Driven_kms", "Owner"
    ]
    categorical_features = [
        "Car_Name", "Fuel_Type", "Selling_type", "Transmission"
    ]

    # Preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    # One shared split for a fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42
        )
    }

    results = []
    trained_models = {}

    for name, regressor in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        trained_models[name] = pipeline

        print(f"\n{name}")
        print(f"MAE:  {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"R2:   {r2:.3f}")

    # Save the comparison results
    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv", index=False
    )

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    # Select the model with the lowest MAE
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "car_price_model.joblib")


    # Feature importance using permutation importance
    importance_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
    )

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": importance_result.importances_mean,
    }).sort_values("Importance", ascending=False)

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nFEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(9, 6))
    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"],
    )
    plt.xlabel("Increase in MAE when feature is shuffled")
    plt.ylabel("Feature")
    plt.title("Car Price Prediction - Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()

    print("Feature importance chart saved.")


    print(f"\nBest model by test MAE: {best_name}")
    print("Saved best model to models/car_price_model.joblib")

    # Plot model comparison
    plt.figure(figsize=(9, 5))
    plt.bar(results_df["Model"], results_df["MAE"])
    plt.ylabel("Mean Absolute Error (lower is better)")
    plt.title("Car Price Prediction: Model Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png")
    plt.close()

    # Plot actual vs predicted for the selected model
    predictions = best_model.predict(X_test)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.7)

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())

    plt.plot([lower, upper], [lower, upper], "r--")
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Actual vs Predicted: {best_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
    plt.close()

    # Save test predictions for reporting
    pd.DataFrame({
        "Actual_Price": y_test.to_numpy(),
        "Predicted_Price": predictions
    }).to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

    print("\nComparison chart and prediction results saved.")
    print("Task completed successfully!")


if __name__ == "__main__":
    main()
