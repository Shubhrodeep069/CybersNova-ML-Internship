# House Price Prediction

A machine learning project that predicts residential house prices using property-related features. The project compares multiple regression algorithms and selects the best-performing model based on RMSE, MAE, and R².

## Project Overview

House prices depend on several factors such as overall quality, living area, garage capacity, basement area, year built, and other property characteristics.

In this project, the Kaggle House Prices dataset was used to build and evaluate regression models for predicting the `SalePrice` of houses.

## Objectives

* Load and explore the house price dataset
* Handle missing values
* Preprocess numerical and categorical features
* Train multiple regression models
* Compare model performance using RMSE, MAE, and R²
* Select the best-performing model
* Save the trained model for future predictions

## Dataset

The project uses the **House Prices: Advanced Regression Techniques** dataset.

The dataset contains:

* 1,460 training records
* 1,459 test records
* 79 input features in the training data
* `SalePrice` as the target variable

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Jupyter Notebook

## Project Structure

```text
House-Price-Prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── house_price_prediction.ipynb
├── house_price_model.pkl
├── preprocessor.pkl
├── requirements.txt
└── README.md
```

## Data Preprocessing

The following preprocessing techniques were applied:

* Removed the `Id` column because it does not provide useful predictive information.
* Separated the `SalePrice` target from the input features.
* Split the training data into training and validation sets using an 80:20 ratio.
* Identified numerical and categorical features.
* Filled missing numerical values using the median.
* Filled missing categorical values using the most frequent value.
* Applied one-hot encoding to categorical features.
* Used `handle_unknown="ignore"` to safely process unseen categorical values.

After preprocessing, the 79 original features were transformed into 285 model-ready features.

## Models Used

Three regression algorithms were trained and compared:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

### Model Performance

| Model             |      RMSE |       MAE |     R² |
| ----------------- | --------: | --------: | -----: |
| Gradient Boosting | 27,784.52 | 16,624.09 | 0.8994 |
| Random Forest     | 28,376.65 | 17,350.76 | 0.8950 |
| Linear Regression | 31,294.99 | 20,466.13 | 0.8723 |

### Best Model

**Gradient Boosting Regressor** performed the best on the validation dataset.

It achieved:

* **RMSE:** 27,784.52
* **MAE:** 16,624.09
* **R²:** 0.8994

The model was selected because it produced the lowest RMSE and MAE and the highest R² among the three tested models.

## Example Prediction

For one validation sample:

```text
Actual Price:    $154,500.00
Predicted Price: $146,763.89
```

The prediction was approximately $7,736 lower than the actual sale price.

## Saved Models

The project saves the trained model and preprocessing pipeline using Joblib:

* `house_price_model.pkl`
* `preprocessor.pkl`

These files can be loaded later to make predictions on new house data.

## How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd House-Price-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
house_price_prediction.ipynb
```

and run the cells sequentially.

## Conclusion

The project demonstrates how machine learning regression techniques can be used to estimate house prices from property features. Among the tested models, Gradient Boosting provided the strongest validation performance with an R² score of 0.8994.
