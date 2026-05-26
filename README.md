# Stock Price Prediction using LSTM

Predicting TCS (NSE) stock prices using a stacked LSTM neural network trained on 10 years of historical data.

Built as part of the **Executive Programme in Algorithmic Trading (EPAT)** at QuantInsti.

---

## Overview

| Item | Detail |
|------|--------|
| Stock | TCS.NS (Tata Consultancy Services) |
| Data | 10 years (~2,500 trading days) via yfinance |
| Model | Stacked LSTM (2 layers × 100 units) |
| Window | 60-day sliding window |
| Split | 80% train / 20% test |
| Metric | RMSE on unscaled prices |

---

## Model Architecture---

## Key Concepts Applied

- **60-timestep sliding window** for sequential pattern learning
- **MinMaxScaler** fitted only on training data (no data leakage)
- **Adam optimizer** with MSE loss
- **inverse_transform** to convert predictions back to actual prices
- **RMSE** for model evaluation

---

## How to Run

```bash
git clone https://github.com/jadhavpramod1136-psj/lstm-stock-price-prediction
cd lstm-stock-price-prediction
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
jupyter notebook
```

---

## Tech Stack

Python · TensorFlow/Keras · pandas · NumPy · scikit-learn · yfinance · Matplotlib

---

## Author

**Pramod Jadhav**  
EPAT Algorithmic Trading | 8 Years Market Experience  
[GitHub](https://github.com/jadhavpramod1136-psj)
