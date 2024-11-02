# Xtrapolate 

- [Project Overview](#project-overview)
- [Data](#data)
- [Installation](#installation)
- [Model Metrics](#model-metrics)
- [Results](#results)
- [License](#license)
- [Disclaimer](#disclaimer)

## Project Overview

Data science understanding among the general public is insufficient to address the increasing impact the field has on life in the 21st century. The project aims to leverage people's love for solving puzzles and recognizing patterns by providing a fun, engaging way to learn data science and machine learning concepts through a game. The online gaming market offers an untapped opportunity for data science-related games, and this project seeks to bridge that gap.

The game allows users to interact with a sales dataset by making predictions about future trends, visualizing their results, and comparing their predictions with those generated by machine learning algorithms. The goal is to make machine learning concepts more accessible and enjoyable.

## Data

We are using a **Sales Dataset** from [Kaggle](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data?resource=download). It contains monthly sales data, including variables such as order numbers, products sold, prices, and geographic regions.

- **Dataset Variables**:
    - `ORDERNUMBER`
    - `QUANTITYORDERED`
    - `PRICEEACH`
    - `ORDERLINENUMBER`
    - `SALES`
    - `ORDERDATE`
    - `STATUS`
    - `QTR_ID`, `MONTH_ID`, `YEAR_ID`
    - `ADDRESSLINE1`, `ADDRESSLINE2`, `CITY`, `STATE`, `POSTALCODE`, `COUNTRY`, `TERRITORY`
    - `CONTACTLASTNAME`, `CONTACTFIRSTNAME`

## Installation

### Prerequisites

- **Python 3.8+**
- **Packages** (to be listed in the `requirements.txt` file):
    - `pandas` (1.3.3)
    - `numpy` (1.21.2)
    - `matplotlib` (3.4.3)
    - `scikit-learn` (0.24.2)
    - `seaborn` (0.11.2)
    - `flask` (2.0.1)

### Setup Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-repository/xtrapolate.git
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    python app.py
    ```

## Model Metrics

### Linear Regression Model

The project uses a **simple linear regression model** to predict sales based on various features like the quantity ordered, price per unit, and order date. Below is a breakdown of the regression equation used in our model:

```math
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \epsilon
```


- **Y**: Dependent variable (the response variable we are predicting), in this case, `SALES`.
- **β0**: Y-intercept, the baseline sales value when all independent variables are zero.
- **X1, X2**, ...: Independent variables (predictors) that affect the outcome, such as:
  - `QUANTITYORDERED`: Number of units ordered.
  - `PRICEEACH`: Price per unit.
  - `MONTH_ID`, `YEAR_ID`: Time-related variables.
  - `COUNTRY`: Geographic location (encoded as a categorical variable).
  - `PRODUCT`: Specific products (encoded as categorical variables).
- **β1, β2**, ...: Coefficients representing the effect of each independent variable on the dependent variable (i.e., how much the SALES change when each predictor increases by one unit).
- **ε:** Error term accounting for the variability in sales not explained by the model.

This regression equation allows us to model the relationship between the sales data and the predictors, giving us a framework to predict future sales based on historical patterns and product-specific features.

### MAPE (Mean Absolute Percentage Error)

The Mean Absolute Percentage Error (MAPE) formula is:

```math
\text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{A_i - F_i}{A_i} \right| \times 100
```


Where:
- **`A_i`**: Actual value
- **`F_i`**: Forecasted value
- **`n`**: Total number of observations

MAPE is used to evaluate the accuracy of the model by measuring the percentage error between the predicted and actual values.

## Results

### Key Insights

- **User Engagement**: The game displays user-predicted sales trends alongside the machine learning model’s predictions. The results are evaluated using MAPE, showing how close the user’s predictions are to the actual data.
- **Scoring System**: Based on MAPE, users receive a score between 0 and 100, with 100 representing a perfect match with the actual data.

### Anticipated Results:

The primary goal is to help users compare their predictions against the model’s predictions and true values. By doing so, users can learn how close their estimates are and see by what percentage their predictions differ from both the actual values and the machine learning model’s predictions.

## Game User Interface

### Gameplay/Player Values
<img src="./pics/Player.png" width="600"/>

### Results and Details
<img src="./pics/Results.png" width="600"/>

### Gameplay Loop
<img src="./pics/Loop.png" width="600"/>





## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

The analysis and predictions made by the game are based on the dataset provided from [Kaggle](https://www.kaggle.com/datasets/kyanyoga/sample-sales-data?resource=download) and are intended for educational purposes. The data used is publicly available, and all efforts have been made to ensure its accuracy. However, no guarantees are provided, and any conclusions drawn from this data should be considered with the dataset's limitations in mind.

