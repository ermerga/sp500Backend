import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://ermerga.github.io"}})


# Function to load or fetch S&P 500 data
def load_sp500_data(ticker):
    sp500 = yf.Ticker(ticker)
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")
    sp500.index = pd.to_datetime(sp500.index)
    return sp500

# Function to fetch today's S&P 500 data
def fetch_todays_data(ticker):
    sp500_ticker = yf.Ticker(ticker)
    today_data = sp500_ticker.history(period="1d")
    if not today_data.empty:
        return {
            "Open": today_data["Open"].iloc[-1],
            "High": today_data["High"].iloc[-1],
            "Low": today_data["Low"].iloc[-1],
            "Close": today_data["Close"].iloc[-1],
            "Volume": today_data["Volume"].iloc[-1]
        }
    else:
        raise ValueError("Failed to fetch today's data. Market may be closed or data unavailable.")

# Function to preprocess data
def preprocess_data(sp500):
    del sp500["Dividends"]
    del sp500["Stock Splits"]
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":].copy()
    
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        ratio_column = f"Close_Ratio_{horizon}"
        sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
        trend_column = f"Trend_{horizon}"
        sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
        new_predictors += [ratio_column, trend_column]
    
    sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])
    return sp500, new_predictors

# Prediction function
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= 0.3] = 1
    preds[preds < 0.3] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Function to predict for new data
def predict_new_data(model, sp500, new_data, predictors):
    new_df = pd.DataFrame([new_data], index=[pd.to_datetime("today")])
    horizons = [2, 5, 60, 250, 1000]
    for horizon in horizons:
        rolling_averages = sp500.rolling(horizon).mean()
        if not rolling_averages.empty:
            last_rolling_avg = rolling_averages["Close"].iloc[-1]
            ratio_column = f"Close_Ratio_{horizon}"
            new_df[ratio_column] = new_data["Close"] / last_rolling_avg
            trend_column = f"Trend_{horizon}"
            trend_value = sp500["Target"].tail(horizon).sum()
            new_df[trend_column] = trend_value
    new_df = new_df[predictors]
    pred_proba = model.predict_proba(new_df)[:, 1]
    return 1 if pred_proba >= 0.5 else 0


@app.route('/predict')
def predict_browser():
    ticker = request.args.get('ticker')
    if not ticker:
        return jsonify({"error": "Ticker Required"}), 400
    

    sp500 = load_sp500_data(ticker)
    sp500, new_predictors = preprocess_data(sp500)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    
    # Train model on all data
    model.fit(sp500[new_predictors], sp500["Target"])
    
    # Fetch today's data
    try:
        new_data = fetch_todays_data(ticker)
        close_price = f"${new_data['Close']:,.2f}"
        prediction = predict_new_data(model, sp500, new_data, new_predictors)
    except ValueError as e:
        print(e)
    
    # # Optional: Run backtest to evaluate model
    # predictions = backtest(sp500, model, new_predictors)
    # print("How Accurate is This Prediction:", precision_score(predictions["Target"], predictions["Predictions"]))
    os.remove("sp500.csv")

    return jsonify({
    "ticker": ticker,
    "close": close_price,
    "prediction": "up" if prediction == 1 else "down"
})



# Main execution
if __name__ == "__main__":
    app.run(port=5000)
