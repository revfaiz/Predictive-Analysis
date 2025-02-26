import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read the CSV file and return a DataFrame.

    :param file_path: Path to the CSV file.
    :return: DataFrame containing the CSV data.
    """
    return pd.read_csv(file_path, parse_dates=['Date'])

def calculate_missing_values(df: pd.DataFrame) -> int:
    """
    Calculate and return the sum of missing values in the DataFrame.

    :param df: The DataFrame.
    :return: The sum of missing values.
    """
    return df.isnull().sum().sum()

def fill_missing_values(df: pd.DataFrame, column: str, method: str, custom_value=None) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame column using the specified method.

    :param df: The DataFrame.
    :param column: The column to fill missing values in.
    :param method: The method to use for filling missing values.
    :param custom_value: The custom value to use if method is "Choose".
    :return: The DataFrame with filled values.
    """
    if method == "Choose":
        df[column].fillna(custom_value, inplace=True)
    elif method == "Mean":
        df[column].fillna(df[column].mean(), inplace=True)
    elif method == "Median":
        df[column].fillna(df[column].median(), inplace=True)
    elif method == "Mode":
        df[column].fillna(df[column].mode()[0], inplace=True)
    elif method == "Zero":
        df[column].fillna(0, inplace=True)
    elif method == "Forward":
        df[column].fillna(method='ffill', inplace=True)
    elif method == "Backward":
        df[column].fillna(method='bfill', inplace=True)
    return df

def split_data(df: pd.DataFrame, target_column: str, test_size: float):
    """
    Split the data into training and testing sets.

    :param df: The DataFrame.
    :param target_column: The column to be predicted.
    :param test_size: The proportion of the dataset to include in the test split.
    :return: X_train, X_test, y_train, y_test
    """
    features = df.drop(columns=[target_column])
    target = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type="LinearRegression"):
    """
    Train the specified machine learning model.

    :param X_train: Training features.
    :param y_train: Training target.
    :param model_type: The type of model to train.
    :return: Trained model
    """
    if model_type == "LinearRegression":
        model = LinearRegression()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return performance metrics.

    :param model: Trained model.
    :param X_test: Testing features.
    :param y_test: Testing target.
    :return: Dictionary of performance metrics
    """
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "f1_score": f1_score(y_test, y_pred, average='macro')
    }
    return metrics

def predict(model, X_future):
    """
    Predict future values using the trained model.

    :param model: Trained model.
    :param X_future: Future features.
    :return: Predicted values
    """
    return model.predict(X_future)
