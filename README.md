# Stock Market Prediction and Forecasting Using Stacked LSTM

This project implements a **Stock Market Prediction** system using a **Stacked LSTM (Long Short-Term Memory)** model for forecasting future stock prices. The model is trained on historical stock data (in this case, Tesla's stock prices) to predict future prices and assist in stock market analysis.

## Project Overview:

- **Data Collection**: Stock market data is collected for Tesla (AAPL stock) and preprocessed for model training.
- **Data Preprocessing**: The dataset is cleaned, missing values are handled, and the data is normalized using MinMaxScaler for model compatibility.
- **Model**: A stacked LSTM model is used for training on historical stock price data, with the goal of forecasting future prices.
- **Prediction**: The model is used to predict the stock prices for the next 30 days.

## Key Features:

1. **Data Collection**: Fetch stock data and visualize it.
2. **Preprocessing**: Clean the data and normalize it.
3. **Stacked LSTM Model**: Create and train the LSTM model.
4. **Model Evaluation**: Measure the performance using RMSE and visualize training and test predictions.
5. **Forecasting**: Predict stock prices for the next 30 days.

## Libraries Used:

- **Keras**
- **TensorFlow**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**


## Setup:

### Step 1: Install Required Libraries

Install the necessary dependencies by running:

```bash
pip install -r requirements.txt
Step 2: Data Collection
The dataset (Tesla.csv) contains historical stock data for Tesla. It includes columns such as Open, High, Low, Close, Volume, and Adj Close. The data is preprocessed for use in the model.

Step 3: Model Training
The stock data is used to train a Stacked LSTM model. The model's architecture consists of three LSTM layers followed by a Dense layer for prediction.

python
Copy code
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
Step 4: Evaluation
The model's performance is evaluated using Root Mean Squared Error (RMSE), and the predictions are plotted for both training and test data.

Step 5: Forecasting
The model is used to predict stock prices for the next 30 days, and the predicted values are plotted for visualization.

Running the Project:
Clone the project or download the files.
Install the necessary dependencies (pip install -r requirements.txt).
Open the stock_prediction.ipynb notebook in Jupyter Notebook.
Run the cells to preprocess the data, train the model, and generate predictions.
Review the results of stock price predictions and visualize them using matplotlib.
Contributing:
Feel free to fork the project, submit issues, or pull requests for any improvements!

License:
This project is licensed under the MIT License.
