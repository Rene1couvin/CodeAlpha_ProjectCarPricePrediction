
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.joblib"


def main():
    if not MODEL_PATH.exists():
        print("Trained model not found. Run python main.py first.")
        return

    model = joblib.load(MODEL_PATH)

    print("\n===== CAR PRICE PREDICTOR =====")
    print("Enter the details of the car below.\n")

    car_name = input("Car name (e.g., city): ").strip()

    try:
        year = int(input("Manufacturing year (e.g., 2018): "))
        present_price = float(
            input("Present price in lakh INR (e.g., 8.5): ")
        )
        driven_kms = float(input("Kilometres driven (e.g., 30000): "))
        owner = int(input("Previous owner category (0, 1, 2...): "))

    except ValueError:
        print("Invalid input. Enter numbers for the numeric fields.")
        return

    fuel_type = input(
        "Fuel type (Petrol/Diesel/CNG): "
    ).strip().capitalize()

    selling_type = input(
        "Seller type (Dealer/Individual): "
    ).strip().capitalize()

    transmission = input(
        "Transmission (Manual/Automatic): "
    ).strip().capitalize()

    current_year = pd.Timestamp.now().year
    car_age = current_year - year

    if (
        year < 1900
        or year > current_year
        or present_price <= 0
        or driven_kms < 0
        or owner < 0
    ):
        print("Please check the values entered and try again.")
        return

    car = pd.DataFrame([{
        "Car_Name": car_name,
        "Car_Age": car_age,
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner,
    }])

    predicted_price = model.predict(car)[0]

    print("\n===== PREDICTION RESULT =====")
    print(f"Car: {car_name}")
    print(f"Estimated selling price: {predicted_price:.2f} lakh INR")
    print("\nThis is a machine learning estimate, not a guaranteed market price.")


if __name__ == "__main__":
    main()

