# Bitcoin Price Prediction using LSTM Neural Networks




A deep learning project that predicts Bitcoin prices using Long Short-Term Memory (LSTM) neural networks. Two versions are provided: a base model using price data and an enhanced model incorporating technical indicators and sentiment analysis.

## 📊 Project Overview

This project uses historical Bitcoin price data from CoinMarketCap (2010-2026) to train LSTM models that predict the next day's closing price. The models leverage TensorFlow/Keras and incorporate various machine learning techniques for time series forecasting.

## 🎯 Features

### Base Model (Version 1)
- **5-layer LSTM architecture** with dropout regularization
- Uses 4 fundamental features: `close`, `volume`, `high`, `low`
- 60-day lookback window for sequence prediction
- Early stopping and learning rate reduction callbacks
- **Performance**: RMSE: $7,086.30, MAE: $4,746.43, MAPE: 5.55%

### Enhanced Model (Version 2)
- **Bidirectional LSTM** for improved pattern recognition
- **23 comprehensive features** including:
  - Price data (open, high, low, close, volume)
  - Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
  - Sentiment features (price sentiment, fear index, volume surge)
  - Momentum and volatility metrics
- Huber loss function for robustness to outliers
- **Performance**: RMSE: $12,955.62, MAE: $9,024.28, MAPE: 10.37%

## 📁 Repository Structure

```
bitcoin-price-prediction/
│
├── bitcoin_base_model.py          # Base LSTM model
├── bitcoin_enhanced_model.py      # Enhanced model with technical indicators
├── data/
│   └── Bitcoin_historical_data_coinmarketcap.csv
├── models/
│   ├── bitcoin_lstm_model.h5
│   └── bitcoin_enhanced_lstm_model.h5
├── visualizations/
│   ├── base_model_results.png
│   └── enhanced_model_results.png
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Pandas
Matplotlib
Scikit-learn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Place your Bitcoin CSV data in the project directory:
```
Bitcoin_5_13_2010-7_12_2010_historical_data_coinmarketcap.csv
```

### Running the Models

**Base Model:**
```bash
python bitcoin_base_model.py
```

**Enhanced Model:**
```bash
python bitcoin_enhanced_model.py
```

## 📈 Model Architecture

### Base Model (Version 1)
```
Input Layer (60, 4)
    ↓
LSTM(512) → Dropout(0.2)
    ↓
LSTM(248) → Dropout(0.2)
    ↓
LSTM(248) → Dropout(0.2)
    ↓
LSTM(128) → Dropout(0.2)
    ↓
LSTM(64) → Dropout(0.2)
    ↓
Dense(1) - Output
```

**Total Parameters**: 2,549,249

### Enhanced Model (Version 2)
```
Input Layer (60, 23)
    ↓
Bidirectional LSTM(100) → Dropout(0.3)
    ↓
LSTM(100) → Dropout(0.3)
    ↓
LSTM(50) → Dropout(0.2)
    ↓
Dense(25, relu) → Dropout(0.2)
    ↓
Dense(1) - Output
```

**Total Parameters**: 251,101

## 📊 Results

### Base Model Performance
| Metric | Value |
|--------|-------|
| RMSE | $7,086.30 |
| MAE | $4,746.43 |
| MAPE | 5.55% |
| Training Samples | 4,477 |
| Testing Samples | 1,120 |

**Sample Prediction:**
- Current Price: $93,729.03
- Predicted Price: $85,650.09
- Expected Change: -8.62%

### Enhanced Model Performance
| Metric | Value |
|--------|-------|
| RMSE | $12,955.62 |
| MAE | $9,024.28 |
| MAPE | 10.37% |
| Training Samples | 4,265 |
| Testing Samples | 1,067 |

**Sample Prediction:**
- Current Price: $93,729.03
- Predicted Price: $78,044.63
- Expected Change: -16.73%
- RSI: 79.75 (Overbought)
- Sentiment: Bearish 📉

## 🔍 Technical Indicators Explained

### Moving Averages
- **SMA (7, 14, 30)**: Simple moving averages for trend identification
- **EMA (12, 26)**: Exponential moving averages for MACD calculation

### Momentum Indicators
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (overbought/oversold levels)

### Volatility Indicators
- **Bollinger Bands**: Price volatility and deviation metrics
- **Volatility**: Rolling standard deviation of prices

### Sentiment Features (Simulated)
- **Price Sentiment**: Bullish (+1), Neutral (0), Bearish (-1)
- **Fear Index**: Market fear level (0-100)
- **Volume Surge**: Abnormal trading activity detection

## 🛠️ Customization

### Adjusting Sequence Length
```python
SEQUENCE_LENGTH = 60  # Change to desired lookback period
```

### Modifying Model Architecture
```python
model = Sequential([
    LSTM(units=YOUR_UNITS, return_sequences=True),
    # Add/remove layers as needed
])
```

### Adding Real Sentiment Data

To use actual sentiment data instead of simulated features:

```python
# Install required packages
pip install tweepy newsapi-python requests

# Example: Twitter Sentiment
import tweepy
from textblob import TextBlob

def get_twitter_sentiment():
    # Your implementation here
    pass

# Example: Fear & Greed Index
import requests

def get_fear_greed_index():
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    return response.json()['data'][0]['value']
```

## 📉 Visualizations

The models generate comprehensive visualizations:

1. **Training History**: Loss and MAE curves
2. **Predictions vs Actual**: Model performance on test data
3. **Error Distribution**: Histogram of prediction errors
4. **Technical Indicators**: RSI, Moving Averages, Volume Analysis
5. **Sentiment Metrics**: Fear Index and Volume Surge Detection


<img width="1489" height="490" alt="image" src="https://github.com/user-attachments/assets/c0343535-817d-4730-99f0-74d973401b71" />
<img width="1393" height="868" alt="image" src="https://github.com/user-attachments/assets/101d13f1-3b32-4263-aa54-88565df9c7e7" />


## ⚠️ Important Disclaimer

**This project is for educational and research purposes only.**

- Cryptocurrency markets are highly volatile and unpredictable
- Past performance does not guarantee future results
- Do NOT use this model for actual trading without:
  - Proper risk management
  - Additional validation
  - Professional financial advice
- The model cannot account for:
  - Regulatory changes
  - Major market events
  - Black swan events
  - Market manipulation

## 🔮 Future Enhancements

- [ ] Integrate real-time Twitter sentiment analysis
- [ ] Add News API for financial news sentiment
- [ ] Implement GRU (Gated Recurrent Unit) comparison
- [ ] Add Transformer-based architecture
- [ ] Include on-chain metrics (active addresses, hash rate)
- [ ] Implement ensemble methods
- [ ] Add real-time prediction API
- [ ] Create web dashboard for live predictions
- [ ] Support for multiple cryptocurrencies
- [ ] Hyperparameter optimization with Optuna

## 📚 Technical Stack

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Preprocessing**: Scikit-learn (MinMaxScaler)
- **Development**: Python 3.8+, Google Colab

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 🙏 Acknowledgments

- CoinMarketCap for historical Bitcoin data
- TensorFlow team for the deep learning framework
- The cryptocurrency community for inspiration
- Research papers on LSTM for time series prediction

## 📖 References

1. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
2. Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
3. McNally, S., et al. (2018). Predicting the price of Bitcoin using Machine Learning.

---

**⭐ If you find this project helpful, please consider giving it a star!**

**💡 Remember**: This is an educational project. Always do your own research and never invest more than you can afford to lose.
