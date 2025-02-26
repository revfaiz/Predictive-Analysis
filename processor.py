import logging
import pandas as pd
from utils import (
    read_csv_file,
    calculate_missing_values,
    fill_missing_values,
    split_data,
    train_model,
    evaluate_model,
    predict
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
        logger.info(f"Loading file from {file_path}")
        self.df = read_csv_file(file_path)
        return self.df

    def get_columns(self):
        if self.df is None:
            raise ValueError("No dataset loaded")
        return self.df.columns.tolist(), self.df.dtypes.astype(str).to_dict()

    def fill_missing(self, column, method, custom_value=None):
        if self.df is None:
            raise ValueError("No dataset loaded")
        self.df = fill_missing_values(self.df, column, method, custom_value)
        return self.df

    def calculate_missing(self):
        if self.df is None:
            raise ValueError("No dataset loaded")
        return calculate_missing_values(self.df)

    def split_data(self, target_column, test_size=0.2):
        if self.df is None:
            raise ValueError("No dataset loaded")
        return split_data(self.df, target_column, test_size)

    def train_model(self, X_train, y_train, model_type='LinearRegression'):
        self.model = train_model(X_train, y_train, model_type)
        return self.model

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("No model trained")
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        self.plot_results(y_test, y_pred, mae, mse)
        return evaluate_model(self.model, X_test, y_test)

    def predict(self, X_future):
        if self.model is None:
            raise ValueError("No model trained")
        return predict(self.model, X_future)

    @staticmethod
    def plot_results(y_test, y_pred, mae, mse):
        plt.figure(figsize=(10, 5))
        sns.lineplot(x=y_test.index, y=y_test, label="Actual", linestyle='dashed')
        sns.lineplot(x=y_test.index, y=y_pred, label="Predicted")
        plt.xlabel("Date")
        plt.ylabel("Exchange Rate (USD to PKR)")
        plt.title(f"Model Prediction - MAE: {mae:.4f}, MSE: {mse:.4f}")
        plt.legend()
        plt.show()
