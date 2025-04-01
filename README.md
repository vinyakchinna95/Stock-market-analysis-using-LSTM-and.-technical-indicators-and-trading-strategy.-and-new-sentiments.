# 📌 Overview

This project focuses on predicting the closing, high, and low prices of stocks in the NSE (National Stock Exchange) market using LSTM (Long Short-Term Memory) networks. Instead of analyzing all available stocks, a filtration process is applied using technical indicators and trading strategies to identify the top-performing sectors. From these sectors, the best-performing companies are selected for training the predictive model.

The prediction is based on historical stock prices obtained via web scraping and the Yahoo Finance API. The project includes data preprocessing, model training, evaluation, and visualization.

# 🔧 Tech Stack & Libraries Used

This project is built using Python and relies on the following libraries:

## 📊 Data Collection & Processing

* pandas - Handling and processing structured data
* numpy - Efficient numerical computations
* yfinance - Fetching historical stock data from Yahoo Finance
* BeautifulSoup - Web scraping for sectoral stock data

# ⚙️ Data Preprocessing

* sklearn.preprocessing.MinMaxScaler - Normalization of stock prices
* matplotlib & seaborn - Data visualization

# 🔥 Deep Learning (LSTM Model)
# 📈 Model Evaluation

* sklearn.metrics - Performance evaluation using R² score and MSE

# 📂 Project Structure

* 📁 nse-stock-prediction/
* │── 📄 README.md  # Project Documentation
* │── 📄 requirements.txt  # List of required dependencies
* │── 📄 data_preprocessing.py  # Data cleaning & feature engineering
* │── 📄 lstm_model.py  # Model training and evaluation
* │── 📄 predict.py  # Future stock price predictions
* │── 📄 web_scraping.py  # Fetching stock & sectoral data
* │── 📂 datasets/  # Historical stock data
* │── 📂 models/  # Saved LSTM model
* │── 📂 results/  # Predictions & evaluation reports

# 📊 Data Collection & Filtration

## Web Scraping & API Calls
* Stock data is collected from Yahoo Finance.
* Sectoral data is extracted using web scraping.

## Filtration Process

* Technical indicators are used to filter top-performing sectors.
* Within these sectors, best-performing companies are selected.

## Data Preprocessing

* Scaling data using MinMaxScaler.
* Splitting data into training (80%) and testing (20%) sets.
* Creating sequences for LSTM model training.

# 🏗️ Model Architecture

### The LSTM model is designed with multiple stacked layers to capture sequential dependencies:
### 3 LSTM Layers (32 neurons each, return sequences)
### 1 Dense Layer (for output prediction)
### Loss Function: Mean Squared Error (MSE)
### Optimizer: Adam

# 📌 Results & Evaluation

* The model is evaluated using R² score and Mean Squared Error (MSE).
* Predicted stock prices are plotted alongside actual prices.
* Future predictions are made for 5 days ahead using the last known stock prices.

# 👥 Contributors

### Manoj Kumar(218r1a66j1@gmail.com)
### Sai Pranith(218r1a66d9@gmail.com)
### Dindi Vinayak(218r1a66i7@gmail.com)
### Kaushik Anand(218r1a66h4@gmail.com)

# 🤝 Contributions

* We welcome contributions! Feel free to fork the repo, submit pull requests, or report issues.

# 📞 Contact

For any queries, reach out via GitHub Issues or email your vinayaktejavath9@gmail.com
