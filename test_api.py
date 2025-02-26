import pytest
import logging
import os
from flask import Flask
from app import app

# Configure logger
logging.basicConfig(filename='test_api_responses.txt', level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def log_response(response, endpoint):
    print('got the hit')

    response_data = f"Endpoint: {endpoint}\nStatus Code: {response.status_code}\nResponse JSON: {response.json}\n\n"
    with open('Currency_dataset_2011_2024.csv', 'a') as file:
        file.write(response_data)
    logger.info(response_data)

def test_api(client):
    # Test upload file with valid input
    data = {
        'file': (open('Currency_dataset_2011_2024.csv', 'rb'), 'data.csv')
    }
    response = client.post('/upload', data=data)
    log_response(response, '/upload')
    assert response.status_code == 200
    assert 'file_path' in response.json
    assert 'columns' in response.json

    # Test upload file with invalid input
    data = {
        'file': (open('invalid_file.txt', 'rb'), 'invalid_file.txt')
    }
    response = client.post('/upload', data=data)
    log_response(response, '/upload')
    assert response.status_code != 200

    # Test get columns
    response = client.post('/columns')
    log_response(response, '/columns')
    assert response.status_code == 200
    assert 'columns' in response.json
    assert 'column_types' in response.json

    # Test fill missing values with valid input
    data = {
        'column': 'USD',  # Use an existing column
        'method': 'Mean'
    }
    response = client.post('/fill_missing', json=data)
    log_response(response, '/fill_missing')
    assert response.status_code == 200
    assert response.json['message'] == 'Missing values filled successfully'

    # Test fill missing values with invalid input
    data = {
        'column': 'invalid_column',
        'method': 'InvalidMethod'
    }
    response = client.post('/fill_missing', json=data)
    log_response(response, '/fill_missing')
    assert response.status_code != 200

    # Test calculate missing values
    response = client.post('/missing_values')
    log_response(response, '/missing_values')
    assert response.status_code == 200
    assert 'missing_values' in response.json

    # Test split data with valid input
    data = {
        'target_column': 'PKR',  # Use an existing column
        'test_size': 0.2
    }
    response = client.post('/split_data', json=data)
    log_response(response, '/split_data')
    assert response.status_code == 200
    assert response.json['message'] == 'Data split successfully'
    assert 'X_train_shape' in response.json
    assert 'X_test_shape' in response.json

    # Test split data with invalid input
    data = {
        'target_column': 'invalid_column',
        'test_size': 1.5
    }
    response = client.post('/split_data', json=data)
    log_response(response, '/split_data')
    assert response.status_code != 200

    # Test train model with valid input
    data = {
        'target_column': 'PKR',  # Use an existing column
        'test_size': 0.2,
        'model_type': 'LinearRegression'
    }
    response = client.post('/train_model', json=data)
    log_response(response, '/train_model')
    assert response.status_code == 200
    assert response.json['message'] == 'Model trained successfully'
    assert 'metrics' in response.json

    # Test train model with invalid input
    data = {
        'target_column': 'PKR',  # Use an existing column
        'test_size': 0.2,
        'model_type': 'InvalidModel'
    }
    response = client.post('/train_model', json=data)
    log_response(response, '/train_model')
    assert response.status_code != 200

    # Test predict with valid input
    data = {
        'X_future': [{'feature1': 1, 'feature2': 2}]
    }
    response = client.post('/predict', json=data)
    log_response(response, '/predict')
    assert response.status_code == 200
    assert 'predictions' in response.json

    # Test predict with invalid input
    data = {
        'X_future': [{'invalid_feature': 'invalid_value'}]
    }
    response = client.post('/predict', json=data)
    log_response(response, '/predict')
    assert response.status_code != 200

if __name__ == '__main__':
    pytest.main()