# M3---Deep-Learning-Applied-AI
M3 - Deep Learning &amp; Applied AI is a repository focused on deep learning techniques and their real-world applications. It includes implementations of neural networks, AI models, and data processing workflows using frameworks like PyTorch. The repository covers topics such as time series forecasting, NLP, and model optimization.
# 📈 LSTM Stock Price Prediction 🚀

## 🔍 Overview

This project implements a Long Short-Term Memory (LSTM) neural network to predict stock prices using historical data. The model is trained on Siemens Energy (ENR.DE) stock prices retrieved from Yahoo Finance.

## ✨ Features

✅ Data collection from Yahoo Finance using `yfinance`
✅ Data preprocessing with lag and rolling statistical features
✅ Normalization using `MinMaxScaler`
✅ LSTM model for time series prediction
✅ Training and validation with PyTorch
✅ Visualization of actual vs. predicted stock prices
✅ Model saving and reloading for future use

## 📦 Dependencies

Ensure you have the following libraries installed:

```bash
pip install pandas numpy yfinance torch scikit-learn matplotlib
```

## 🔄 Project Workflow

### 📊 Step 1: Load and Preprocess Data

- Fetch historical stock prices for Siemens Energy from Yahoo Finance.
- Select only the closing price for analysis.
- Sort data by date.

### 🔁 Step 2: Create Lag and Rolling Features

- Generate lagged price features (5-day lag).
- Compute 10-day rolling mean and standard deviation.
- Handle missing values caused by rolling calculations.

### 🔄 Step 3: Normalize the Data

- Scale the features using `MinMaxScaler` for better model performance.

### 📌 Step 4: Prepare Data for LSTM Input

- Create sequences of past `n_timesteps` observations as input.
- Define the target variable as the next day's closing price.

### ✂️ Step 5: Split Data into Training and Testing Sets

- Allocate 80% for training, 20% for testing.
- Further split training data into train and validation sets (80%-20%).
- Convert data into PyTorch tensors and create `DataLoader` instances.

### 🏗️ Step 6: Build the LSTM Model

- Implement an LSTM model with:
  - Two LSTM layers
  - Dropout for regularization
  - Fully connected output layer
- Use Mean Squared Error (MSE) loss and Adam optimizer.

### 🏋️ Step 7: Train the Model

- Train for 200 epochs.
- Perform validation to track model performance.
- Print training and validation loss every 2 epochs.

### 📈 Step 8: Evaluate the Model

- Generate predictions on the test set.
- Denormalize predictions to original price values.
- Plot actual vs. predicted stock prices.

### 💾 Step 9: Save and Load the Model

- Save the trained model as `lstm_model.pth`.
- Load the model for future predictions.

## 🚀 Usage

Run the script to train and evaluate the model:

```bash
python Assignment_1_M3_LSTM_Stock_prediction.ipynb
```

## 📉 Results

The model predicts stock price trends based on historical data. Predictions are visualized through Matplotlib plots to compare actual vs. predicted prices.

## 🔮 Future Improvements

🔧 Hyperparameter tuning for better accuracy.
📊 Adding external financial indicators.
🔬 Experimenting with different LSTM architectures.
🤖 Incorporating other deep learning models like GRU or Transformer-based approaches.

## 📜 License

This project is open-source and available for educational purposes. 🎓

