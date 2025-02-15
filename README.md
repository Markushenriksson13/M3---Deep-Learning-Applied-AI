# 📌 M3 - Deep Learning & Applied AI  

This repository focuses on **deep learning techniques** and their real-world applications. It includes implementations of neural networks, AI models, and data processing workflows using frameworks like **PyTorch, TensorFlow, and Scikit-learn**.  

## 🔥 Key Topics Covered  

- 🕰 **Time Series Forecasting**  
- 📝 **Natural Language Processing (NLP)**  
- 🖼 **Computer Vision**  
- 🎮 **Reinforcement Learning**  
- ⚡ **Model Optimization & Deployment**  

The goal is to apply deep learning methodologies to **practical scenarios**, providing hands-on implementations, code samples, and structured workflows.  

---

## 📂 Repository Structure  

📁 **Assignments & Projects** – Hands-on deep learning tasks with real-world applications.  
📁 **Model Implementations** – Various neural network architectures and frameworks.  
📁 **Datasets & Preprocessing** – Scripts for data collection, feature engineering, and preprocessing.  
📁 **Optimization & Fine-Tuning** – Techniques to improve AI model performance.  

---

# 📕 Assignments  

## 🟢 Assignment 1: LSTM Stock Price Prediction  

### 🔍 Overview  
This project implements a **Long Short-Term Memory (LSTM) neural network** to predict stock prices using historical data. The model is trained on **Siemens Energy (ENR.DE) stock prices** retrieved from **Yahoo Finance**.  

### ✨ Key Features  

✅ **Data collection** using `yfinance`  
✅ **Feature engineering** with lag and rolling statistical features  
✅ **Data normalization** using `MinMaxScaler`  
✅ **LSTM model implementation** with PyTorch  
✅ **Training, validation, and visualization** of results  
✅ **Model saving and reloading** for future use  

### 🚀 Usage  

Run the following script to train and evaluate the model:  

```bash
python Assignment_1_M3_LSTM_Stock_prediction.ipynb
```

## 📈 Results & Future Improvements  

- 🔧 **Hyperparameter tuning** for better accuracy  
- 📊 **Adding external financial indicators** to enhance predictions  
- 🤖 **Exploring other architectures** like GRU or Transformer-based models  

---

# 🟡 Assignment 2: Sentiment Classifier using SBERT  

## 🔍 Overview  
This project implements a **financial sentiment classifier** using **Sentence-BERT (SBERT)**. The model is trained on **financial news headlines** to predict whether the sentiment is **positive, negative, or neutral**.  

## ✨ Key Features  

✅ **Data collection & preprocessing** of financial text  
✅ **Sentence embedding** using **SBERT**  
✅ **Fine-tuning with PyTorch** for sentiment classification  
✅ **Evaluation on financial sentiment datasets**  
✅ **Real-time sentiment analysis on news headlines**  

---

## 🏗 Project Workflow  

### 1️⃣ Data Collection & Preprocessing  
- Gather financial news headlines and labeled sentiment data.  
- Tokenize and preprocess text for model training.  

### 2️⃣ Embedding with SBERT  
- Convert sentences into numerical representations using `sentence-transformers`.  

### 3️⃣ Model Training & Evaluation  
- Fine-tune a **neural network classifier** on SBERT embeddings.  
- Evaluate performance using **accuracy, F1-score, and confusion matrix**.  

### 4️⃣ Real-time Sentiment Analysis  
- Apply the trained model on **new financial news** to classify sentiment.  

---

## 🚀 Usage  

Run the script to train and test the model:  

```bash
python Assignment_2_M3_Financial_Sentiment_Classifier.ipynb
```
## 📈 Results & Future Improvements  

- 📊 **Higher accuracy** with domain-specific embeddings  
- 🔄 **Integrating real-time financial news APIs**  
- 🤖 **Exploring transformer-based architectures like FinBERT**  

---

## 📜 License  

This project is **open-source** and available for educational purposes. 🎓  

---

## 🔖 Hashtags  

`#DeepLearning` `#ArtificialIntelligence` `#LSTM` `#TimeSeriesForecasting` `#StockPrediction` `#SentimentAnalysis` `#NLP` `#SBERT` `#FinancialAI` `#PyTorch` `#MachineLearning`  
