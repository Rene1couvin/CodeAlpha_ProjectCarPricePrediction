# Car Price Prediction with Machine Learning

## Project Overview
This project predicts the selling price of a used car using machine learning. It compares Linear Regression, Random Forest Regression, and Gradient Boosting to evaluate their performance.

## Objectives
- Clean and prepare used-car data.
- Explore the factors associated with car selling prices.
- Train and compare multiple regression models.
- Evaluate model performance using MAE, RMSE, and R².
- Save the selected model for future predictions.

## Dataset
The dataset contains used-car information, including:
- Car name
- Year of manufacture
- Present price
- Kilometres driven
- Fuel type
- Seller type
- Transmission
- Previous owner information

The target variable is `Selling_Price`. Prices in the supplied dataset are expressed in lakh Indian rupees.

## Machine Learning Models
1. Linear Regression
2. Random Forest Regression
3. Gradient Boosting Regression

Models are evaluated using a shared training and testing split. The model with the lowest test Mean Absolute Error (MAE) is selected by the training script.

## Evaluation Metrics
- **MAE:** Average absolute prediction error.
- **RMSE:** Gives greater weight to larger prediction errors.
- **R²:** Measures how much variation in the target is explained by the model.

## Project Structure
```text
CodeAlpha_Car_Price_Prediction/
├── data/
│   └── car data.csv
├── models/
│   └── car_price_model.joblib
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   ├── prediction_results.csv
│   ├── feature_importance.csv
│   └── feature_importance.png
├── main.py
├── predict.py
├── requirements.txt
└── README.md
```

## How to Run
1. Install Python.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Train and compare models:
   `python main.py`
4. Make a sample prediction:
   `python predict.py`

## Results
The model comparison script reports MAE, RMSE, and R² for all three algorithms. The selected model and evaluation charts are saved in the project folders.

## Limitations
Predictions depend on the supplied dataset and may not reflect current market prices. Results may be less reliable for unfamiliar car names or unusual input values.

## Author
BANA RUSIZANA Rene Francis

## Internship
CodeAlpha — Machine Learning Project
