
# Imports -----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, r2_score


file_url = 'https://raw.githubusercontent.com/T-Rex-chess/OMSBACapstone/main/data/sales_data_sample.csv'
sales_data = pd.read_csv(file_url, encoding='ISO-8859-1') #Handles wider range of special char.
sales_data_df = pd.DataFrame(sales_data) # create a dataframe of the sales data
#print(sales_data.head())
print("\n Sales Dataframe Head:")
print(sales_data_df.head())



# BaseManager Parent Class -----------------------------------------------------------------------------------------------
# BaseManager class: Parent class.
class BaseManager:
    def __init__(self, data):
        self.data = data



# DataLoader Class -------------------------------------------------------------------------------------------------------
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
    

    def get_summary_stats(self):
        """
        This function returns summary statistics for a pandas DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame to analyze.
        Returns:
            pd.DataFrame: A DataFrame containing summary statistics.
        """
        return self.data.describe()




# ModelManager Class ---------------------------------------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        """
        Fit the linear regression model.
        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.
        Args:
            X (array-like): The feature matrix for which to make predictions.
        Returns:
            array-like: The predicted values.
        """
        return self.model.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model's performance.
        Args:
            X (array-like): The feature matrix.
            y (array-like): The target vector.
        Returns:
            tuple: (Mean squared error, R-squared score)
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred)
        return mse, r2, mape

    def get_coefficients(self):
        """
        Get the coefficients of the linear regression model.
        Returns:
            array-like: The coefficients.
        """
        return self.model.coef_

    def get_intercept(self):
        """
        Get the intercept of the linear regression model.
        Returns:
            float: The intercept.
        """
        return self.model.intercept_
    
# How to Use the ModelManager Class ----------------------
# Create an instance of the class
'''
model = ModelManager()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)

# Get coefficients and intercept
coefficients = model.get_coefficients()
intercept = model.get_intercept()
'''


# Ready for ScoreManager Class !!! ------------------------------------------------------------------------------





# Ready for ResultVisualizer Class !!! --------------------------------------------------------------------------




# ----- RUN THE PROGRAM !!! -------------------------------------------------------------------------------------
# Call the DataLoader Class -------------------------------------------------------------------------------------
print("\n")
print("Beginning program execution \n")
print("Calling the DataLoader Class \n")
# Call the methods from the DataLoader class
data_loader = DataLoader(sales_data)
data_loader.preprocess_data()
data_loader.handle_missing_values()

# Split the data (assuming 'SALES' is the target column)
X_train, X_test, y_train, y_test = data_loader.split_data(target_column='SALES')

# Output first few rows of training data for verification
print("First few rows of training data: \n", X_train.head())

# Display summary statistics
print("Here are the summary statistics of the Sales Dataframe: \n")
summ_stats_df = DataLoader(sales_data_df)
summ_stats_df.preprocess_data()
summ_stats_df.handle_missing_values()
summary_stats = summ_stats_df.get_summary_stats()
print("\n")
print("Summary Statistics on the DataFrame:")
print(summary_stats)

# Dataset is preprocessed and cleaned ready for testing and predictions !!!


# Call the ModelManager Class -----------------------------------------------------------------------------------
# Call the methods from the ModelManager class
print("\nCalling the ModelManager Class to Fit the LinearRegression Model")
regr = ModelManager() #create instance of the ModelManager class
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
mse, r2, mape = regr.evaluate(X_test, y_test)
coefficients = regr.get_coefficients()
intercept = regr.get_intercept()

print("Here are the results of the fitted linear model: ")
print("MSE: ", mse)
print("r2: ", r2)
print("MAPE: ", mape)
print("\n")
print("Program Executed Successfully \n")

# END -----------------------------------------------------------------------------------------------------------


