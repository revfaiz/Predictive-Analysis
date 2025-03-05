from flask import Flask, request, jsonify
import pandas as pd
import logging
from processor import DataProcessor

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
processor = DataProcessor()

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles dataset upload.
    """
    file = request.files['file']
    file_path = f"{file.filename}"
    file.save(file_path)
    df = processor.load_file(file_path)
    return jsonify({"file_path": file_path, "columns": df.columns.tolist()})

@app.route('/columns', methods=['POST'])
def get_columns():
    """
    Returns the columns and their data types.
    """
    try:
        columns, column_types = processor.get_columns()
        return jsonify({"columns": columns, "column_types": column_types})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/fill_missing', methods=['POST'])
def fill_missing():
    """
    Fills missing values in the specified column using the given method.
    """
    try:
        data = request.json
        column = data.get('column')
        method = data.get('method')
        custom_value = data.get('custom_value', None)
        df = processor.fill_missing(column, method, custom_value)
        return jsonify({"message": "Missing values filled successfully"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/missing_values', methods=['POST'])
def missing_values():
    """
    Returns the count of missing values in the dataset.
    """
    try:
        missing_values_count = processor.calculate_missing()
        return jsonify({"missing_values": int(missing_values_count)})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/split_data', methods=['POST'])
def split_data_route():
    """
    Splits the dataset into training and testing sets.
    """
    try:
        data = request.json
        target_column = data.get('target_column')
        test_size = data.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = processor.split_data(target_column, test_size)
        return jsonify({"message": "Data split successfully", "X_train_shape": X_train.shape, "X_test_shape": X_test.shape})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/train_model', methods=['POST'])
def train_model_route():
    """
    Trains a machine learning model using the specified parameters.
    """
    try:
        data = request.json
        target_column = data.get('target_column')
        test_size = data.get('test_size', 0.2)
        model_type = data.get('model_type', 'LinearRegression')
        X_train, X_test, y_train, y_test = processor.split_data(target_column, test_size)
        model = processor.train_model(X_train, y_train, model_type)
        metrics = processor.evaluate_model(X_test, y_test)
        return jsonify({"message": "Model trained successfully", "metrics": metrics})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Predicts future values using the trained model.
    """
    try:
        data = request.json
        X_future = pd.DataFrame(data.get('X_future'))
        predictions = processor.predict(X_future)
        return jsonify({"predictions": predictions.tolist()})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/prophet/train', methods=['POST'])
def train_prophet_model():
    """
    Trains a Prophet model using the provided data.
    """
    data = request.get_json()
    response = processor.train_prophet_model(data)
    return jsonify(response), 200

@app.route('/prophet/predict', methods=['POST'])
def make_prophet_prediction():
    """
    Makes predictions using the trained Prophet model.
    """
    data = request.get_json()
    response = processor.predict_with_prophet(data)
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, port=4432)
