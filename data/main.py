import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


file_url = 'https://raw.githubusercontent.com/T-Rex-chess/OMSBACapstone/main/data/sales_data_sample.csv'
sales_data = pd.read_csv(file_url, encoding='ISO-8859-1') #Handles wider range of special char.
#print(sales_data.head())


# BaseManager class: Parent class.
class BaseManager:
    def __init__(self, data):
        self.data = data

# DataLoader class: Responsible for loading and preprocessing data, including handling NaN values and non-numeric columns.
class DataLoader(BaseManager):
    def load_data(self):
        print("Data loaded successfully.")
    
    def handle_missing_values(self):
        # Handle missing values by filling with the median
        self.data.fillna(self.data.median(), inplace=True)
        print("Missing values handled successfully.")
    
    def preprocess_data(self):
        # Drop non-numeric columns
        self.data = self.data.select_dtypes(include=[np.number])
        print("Non-numeric columns dropped successfully.")
    
    def split_data(self, target_column, test_size=0.2):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

### Add next class 


# Call the methods from the DataLoader class
data_loader = DataLoader(sales_data)
data_loader.preprocess_data()
data_loader.handle_missing_values()

# Split the data (assuming 'SALES' is the target column)
X_train, X_test, y_train, y_test = data_loader.split_data(target_column='SALES')

# Output first few rows of training data for verification
print("First few rows of training data:", X_train.head())

# Dataset is preprocessed and cleaned ready for testing and predictions !!!