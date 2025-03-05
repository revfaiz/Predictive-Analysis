import pandas as pd
from utils import (
    read_csv_file,
    fill_missing_values,
    split_data,
    train_model,
    evaluate_model,
    predict,
    train_prophet_model,
    predict_with_prophet,
    rename_and_filter_columns
)

# Load the dataset
file_path = 'Currency_dataset_2011_2024.csv'
df = read_csv_file(file_path)

# Fill missing values
df = fill_missing_values(df, column='PKR', method='ffill')

# Split the data
X_train, X_test, y_train, y_test = split_data(df, target_column='PKR', test_size=0.2)

# Train the Linear Regression model
linear_model = train_model(X_train, y_train, model_type='LinearRegression')

# Evaluate the Linear Regression model
linear_metrics = evaluate_model(linear_model, X_test, y_test)
print("Linear Regression Model Evaluation Metrics:", linear_metrics)

# Train the Ridge Regression model
ridge_model = train_model(X_train, y_train, model_type='Ridge')

# Evaluate the Ridge Regression model
ridge_metrics = evaluate_model(ridge_model, X_test, y_test)
print("Ridge Regression Model Evaluation Metrics:", ridge_metrics)

# Predict future values with Linear Regression model
X_future = pd.DataFrame({
    'Date': pd.to_datetime(['2025-02-18', '2025-02-19', '2025-02-20'])  # Replace with actual future values
})
X_future['Date'] = X_future['Date'].map(pd.Timestamp.toordinal)
linear_predictions = predict(linear_model, X_future)
print("Linear Regression Predictions:", linear_predictions)

# Predict future values with Ridge Regression model
ridge_predictions = predict(ridge_model, X_future)
print("Ridge Regression Predictions:", ridge_predictions)

# Train Prophet model
prophet_data = df[['Date', 'PKR']].rename(columns={'Date': 'ds', 'PKR': 'y'})
prophet_response = train_prophet_model(prophet_data)
print(prophet_response)

# Predict with Prophet model
prophet_prediction_data = {'periods': 30}
prophet_forecast = predict_with_prophet(prophet_prediction_data)
print("Prophet Forecast:", prophet_forecast)
