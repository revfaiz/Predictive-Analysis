from flask import Flask, request, jsonify
import pandas as pd
from exchange_rates.project_backend.utils import read_csv_file, calculate_missing_values, fill_missing_values, split_data, train_model, evaluate_model, predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

df = None  # Global variable to hold dataset
model = None  # Global variable to hold trained model

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    file = request.files['file']
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)
    df = read_csv_file(file_path)
    return jsonify({"file_path": file_path, "columns": df.columns.tolist()})

@app.route('/columns', methods=['POST'])
def get_columns():
    global df
    if df is None:
        return jsonify({"error": "No dataset loaded"}), 400
    columns = df.columns.tolist()
    column_types = df.dtypes.astype(str).to_dict()
    return jsonify({"columns": columns, "column_types": column_types})

@app.route('/fill_missing', methods=['POST'])
def fill_missing():
    global df
    if df is None:
        return jsonify({"error": "No dataset loaded"}), 400
    
    data = request.json
    column = data.get('column')
    method = data.get('method')
    custom_value = data.get('custom_value', None)
    
    df = fill_missing_values(df, column, method, custom_value)
    return jsonify({"message": "Missing values filled successfully"})

@app.route('/missing_values', methods=['POST'])
def missing_values():
    global df
    if df is None:
        return jsonify({"error": "No dataset loaded"}), 400
    missing_values_count = calculate_missing_values(df)
    return jsonify({"missing_values": missing_values_count})

@app.route('/split_data', methods=['POST'])
def split_data_route():
    global df
    if df is None:
        return jsonify({"error": "No dataset loaded"}), 400
    
    data = request.json
    target_column = data.get('target_column')
    test_size = data.get('test_size', 0.2)
    
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size)
    return jsonify({"message": "Data split successfully", "X_train_shape": X_train.shape, "X_test_shape": X_test.shape})

@app.route('/train_model', methods=['POST'])
def train_model_route():
    global df, model
    if df is None:
        return jsonify({"error": "No dataset loaded"}), 400
    
    data = request.json
    target_column = data.get('target_column')
    test_size = data.get('test_size', 0.2)
    model_type = data.get('model_type', 'LinearRegression')
    
    X_train, X_test, y_train, y_test = split_data(df, target_column, test_size)
    model = train_model(X_train, y_train, model_type)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Calculate MAE and MSE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    # Plot results
    plot_results(y_test, y_pred, mae, mse)
    
    return jsonify({"message": "Model trained successfully", "metrics": metrics})

@app.route('/predict', methods=['POST'])
def predict_route():
    global model
    if model is None:
        return jsonify({"error": "No model trained"}), 400
    
    data = request.json
    X_future = pd.DataFrame(data.get('X_future'))
    
    predictions = predict(model, X_future)
    return jsonify({"predictions": predictions.tolist()})

def plot_results(y_test, y_pred, mae, mse):
    """
    Visualize actual vs. predicted exchange rates and display MAE and MSE scores.

    :param y_test: Actual values of exchange rates.
    :param y_pred: Predicted values from the model.
    :param mae: Mean Absolute Error of the model.
    :param mse: Mean Squared Error of the model.
    """
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=y_test.index, y=y_test, label="Actual", linestyle='dashed')
    sns.lineplot(x=y_test.index, y=y_pred, label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate (USD to PKR)")
    plt.title(f"Model Prediction - MAE: {mae:.4f}, MSE: {mse:.4f}")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    app.run(debug=True, port=4432)
