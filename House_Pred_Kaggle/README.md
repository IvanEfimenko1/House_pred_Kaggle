
# House Price Prediction

This project uses a dataset of house features to predict house prices. The implementation is done using various machine learning models in a Jupyter Notebook.

## Table of Contents

- [Installation](#installation)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.7 or higher
- Jupyter Notebook
- Docker (optional, for containerized execution)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/YourUsername/House_Price_Prediction.git
    cd House_Price_Prediction
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) If you prefer to use Docker, build the Docker container:

    ```bash
    docker-compose build
    ```

## Running the Notebook

1. Activate the virtual environment if not already activated:

    ```bash
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. Start Jupyter Notebook:

    ```bash
    jupyter notebook
    ```

3. Open the `House_predict_kagle.ipynb` notebook and run the cells sequentially to perform the analysis and model training.

## Project Structure

- `.dockerignore`: Specifies files and directories to be ignored by Docker.
- `House_predict_kagle.ipynb`: Jupyter notebook containing the implementation and analysis of the house price data.
- `docker-compose.yml`: Docker Compose file for managing multi-container Docker applications.
- `Dockerfile`: Instructions for building the Docker image.
- `README.md`: This file, provides information about the project.
- `requirements.txt`: List of project dependencies.
- `processed_dataset_ul.csv`: The dataset containing house features and target variable.

## Data Description

The dataset `processed_dataset_ul.csv` contains the following columns (among others):

- `id`: Unique identifier for each house
- `house_lot_frontage`: Lot frontage of the house
- `house_lot_area`: Lot area of the house
- `house_lot_shape`: Shape of the lot
- `house_land_contour`: Flatness of the property
- `house_land_slope`: Slope of the property
- `house_overall_quality`: Overall quality of the house
- `house_overall_condition`: Overall condition of the house
- `house_year_built`: Year the house was built
- `house_remodeling_year`: Year the house was remodeled
- ... (226 columns in total)

## Data Preprocessing

The data preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Normalizing numerical features

Example code snippet for data preprocessing:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('processed_dataset_ul.csv')

# Handle missing values
df = df.dropna()

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Normalize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Further preprocessing steps...
```

## Model Training and Evaluation

The model training process includes:
- Splitting the data into training and testing sets
- Training various machine learning models
- Evaluating the models using metrics like RMSE and R-squared

Example code snippet for model training and evaluation:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the data
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')
```
