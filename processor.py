import logging
import pandas as pd
from utils import (
    read_csv_file,
    calculate_missing_values,
    fill_missing_values,
    split_data,
    train_model,
    evaluate_model,
    predict,
    train_prophet_model,
    predict_with_prophet,
    rename_and_filter_columns
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.df = None
        self.model = None

    def load_file(self, file_path):
        """
        Load a file and return the DataFrame.

        :param file_path: Path to the file.
        :return: DataFrame containing the file data.
        """
        logger.info(f"Loading file from {file_path}")
        self.df = read_csv_file(file_path)
        return self.df

    def get_columns(self):
        """
        Get the columns and their data types from the DataFrame.

        :return: List of columns and their data types.
        """
        if self.df is None:
            raise ValueError("No dataset loaded")
        return self.df.columns.tolist(), self.df.dtypes.astype(str).to_dict()

    def fill_missing(self, column, method, custom_value=None):
        """
        Fill missing values in the specified column using the given method.

        :param column: The column to fill missing values in.
        :param method: The method to use for filling missing values.
        :param custom_value: The custom value to use if method is "Custom".
        :return: DataFrame with filled values.
        """
        if self.df is None:
            raise ValueError("No dataset loaded")
        self.df = fill_missing_values(self.df, column, method, custom_value)
        return self.df

    def calculate_missing(self):
        """
        Calculate and return the sum of missing values in the DataFrame.

        :return: The sum of missing values.
        """
        if self.df is None:
            raise ValueError("No dataset loaded")
        return calculate_missing_values(self.df)

    def split_data(self, target_column, test_size=0.2):
        """
        Split the data into training and testing sets.

        :param target_column: The column to be predicted.
        :param test_size: The proportion of the dataset to include in the test split.
        :return: X_train, X_test, y_train, y_test
        """
        if self.df is None:
            raise ValueError("No dataset loaded")
        return split_data(self.df, target_column, test_size)

    def train_model(self, X_train, y_train, model_type='LinearRegression'):
        """
        Train the specified machine learning model.

        :param X_train: Training features.
        :param y_train: Training target.
        :param model_type: The type of model to train.
        :return: Trained model
        """
        self.model = train_model(X_train, y_train, model_type)
        return self.model

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model and return performance metrics.

        :param X_test: Testing features.
        :param y_test: Testing target.
        :return: Dictionary of performance metrics
        """
        if self.model is None:
            raise ValueError("No model trained")
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        self.plot_results(y_test, y_pred, mae, mse)
        return evaluate_model(self.model, X_test, y_test)

    def predict(self, X_future):
        """
        Predict future values using the trained model.

        :param X_future: Future features.
        :return: Predicted values
        """
        if self.model is None:
            raise ValueError("No model trained")
        return predict(self.model, X_future)

    @staticmethod
    def plot_results(y_test, y_pred, mae, mse):
        """
        Plot the actual vs predicted results.

        :param y_test: Actual values.
        :param y_pred: Predicted values.
        :param mae: Mean Absolute Error.
        :param mse: Mean Squared Error.
        """
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=y_test.index, y=y_test, label="Actual", linestyle='dashed')
        sns.lineplot(x=y_test.index, y=y_pred, label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("Exchange Rate (USD to PKR)")
        plt.title(f"Model Prediction - MAE: {mae:.4f}, MSE: {mse:.4f}")
        plt.legend()
        plt.show()

    def train_prophet_model(self, data):
        """
        Train a Prophet model using the provided data.

        :param data: Data for training the Prophet model.
        :return: Success message
        """
        return train_prophet_model(data)

    def predict_with_prophet(self, data):
        """
        Make predictions using the trained Prophet model.

        :param data: Data for making predictions.
        :return: Forecast in JSON format
        """
        return predict_with_prophet(data)

    def rename_and_filter_columns(self, data, date_column, target_column):
        """
        Rename and filter columns in the DataFrame.

        :param data: The DataFrame.
        :param date_column: The name of the date column.
        :param target_column: The name of the target column.
        :return: The DataFrame with renamed and filtered columns.
        """
        return rename_and_filter_columns(data, date_column, target_column)
