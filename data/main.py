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
        # Handle missing values for numeric columns by filling with the median
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())
        # Flag which datapoints had any missing values (for numeric colunms)
        for col in numeric_cols:
            self.data[f"{col}_missing"] = self.data[col].isna().astype(int)
   
        # Fill non-numeric missing values with "Unknown" or a similar placeholder
        self.data.fillna({'STATE': 'Unknown', 'TERRITORY': 'Unknown'}, inplace=True)
        print("Missing values handled successfully.")
    
    def preprocess_data(self):
        # Drop irrelevant columns
        columns_to_drop = ['ORDERNUMBER', 'PHONE', 'ADDRESSLINE1', 'ADDRESSLINE2', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME']
        self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Encode categorical variables using one-hot encoding
        categorical_cols = ['STATUS', 'PRODUCTLINE', 'DEALSIZE', 'COUNTRY']
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)
        
        # Drop non-numeric columns after encoding
        self.data = self.data.select_dtypes(include=[np.number])
        print("Non-numeric columns dropped and categorical variables encoded successfully.")
    
    def split_data(self, target_column, test_size=0.2):
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test


# TESTING ---

# Call the methods from the DataLoader class
data_loader = DataLoader(sales_data)
data_loader.preprocess_data()
data_loader.handle_missing_values()

# Split the data (assuming 'SALES' is the target column)
X_train, X_test, y_train, y_test = data_loader.split_data(target_column='SALES')

# Output first few rows of training data for verification
print("First few rows of training data:", X_train.head())


# Save pre-processed data to csv 
output_file_path = 'cleaned_data.csv'
data_loader.data.to_csv(output_file_path, index=False)
print(f"Pre-processed data saved to {output_file_path}")

