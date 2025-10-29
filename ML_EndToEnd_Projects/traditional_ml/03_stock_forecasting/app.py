"""
Time Series Forecasting - Complete Implementation
Stock price prediction using ARIMA, Prophet, LSTM, and ensemble methods.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time series libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# FastAPI for serving
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import joblib
import json
from pathlib import Path

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockForecastingEngine:
    """Complete stock forecasting system with multiple models."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.data = {}
        self.predictions = {}
        
    def fetch_stock_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        logger.info(f"Fetching data for {symbol} for period {period}")
        
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Add technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
            data['Bollinger_Upper'], data['Bollinger_Lower'] = self._calculate_bollinger_bands(data['Close'])
            data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
            
            # Price changes
            data['Price_Change'] = data['Close'].pct_change()
            data['High_Low_Ratio'] = data['High'] / data['Low']
            data['Open_Close_Ratio'] = data['Open'] / data['Close']
            
            self.data[symbol] = data
            logger.info(f"Fetched {len(data)} records for {symbol}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator."""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band
    
    def prepare_data_for_lstm(self, data: pd.DataFrame, lookback: int = 60) -> tuple:
        """Prepare data for LSTM model."""
        # Use multiple features
        features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD']
        df = data[features].dropna()
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        return np.array(X), np.array(y), scaler
    
    def train_arima_model(self, symbol: str, target_col: str = 'Close') -> dict:
        """Train ARIMA model."""
        logger.info(f"Training ARIMA model for {symbol}")
        
        data = self.data[symbol][target_col].dropna()
        
        # Check stationarity
        adf_result = adfuller(data)
        is_stationary = adf_result[1] < 0.05
        
        if not is_stationary:
            # Difference the series
            data_diff = data.diff().dropna()
            adf_result = adfuller(data_diff)
            d = 1
        else:
            data_diff = data
            d = 0
        
        # Find best ARIMA parameters (simplified)
        best_aic = float('inf')
        best_params = None
        
        for p in range(0, 3):
            for q in range(0, 3):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                except:
                    continue
        
        # Train final model
        final_model = ARIMA(data, order=best_params)
        fitted_model = final_model.fit()
        
        self.models[f'{symbol}_arima'] = fitted_model
        
        # Forecast
        forecast_steps = 30
        forecast = fitted_model.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=forecast_steps, freq='D')
        
        result = {
            'model': fitted_model,
            'best_params': best_params,
            'aic': best_aic,
            'forecast': pd.Series(forecast, index=forecast_index),
            'last_actual': data.iloc[-1]
        }
        
        logger.info(f"ARIMA model trained. Best params: {best_params}, AIC: {best_aic:.2f}")
        return result
    
    def train_prophet_model(self, symbol: str, target_col: str = 'Close') -> dict:
        """Train Prophet model."""
        logger.info(f"Training Prophet model for {symbol}")
        
        data = self.data[symbol].reset_index()
        prophet_data = pd.DataFrame({
            'ds': data['Date'],
            'y': data[target_col]
        })
        
        # Add additional regressors
        prophet_data['volume'] = data['Volume']
        prophet_data['rsi'] = data['RSI']
        prophet_data = prophet_data.dropna()
        
        # Initialize and train Prophet
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        model.add_regressor('volume')
        model.add_regressor('rsi')
        model.fit(prophet_data)
        
        # Make forecast
        future = model.make_future_dataframe(periods=30)
        
        # Add regressor values for future (using last known values)
        last_volume = prophet_data['volume'].iloc[-1]
        last_rsi = prophet_data['rsi'].iloc[-1]
        
        future['volume'] = future['volume'].fillna(last_volume)
        future['rsi'] = future['rsi'].fillna(last_rsi)
        
        forecast = model.predict(future)
        
        self.models[f'{symbol}_prophet'] = model
        
        result = {
            'model': model,
            'forecast': forecast,
            'last_actual': prophet_data['y'].iloc[-1]
        }
        
        logger.info(f"Prophet model trained successfully")
        return result
    
    def train_lstm_model(self, symbol: str, epochs: int = 50) -> dict:
        """Train LSTM model."""
        logger.info(f"Training LSTM model for {symbol}")
        
        # Prepare data
        X, y, scaler = self.prepare_data_for_lstm(self.data[symbol])
        self.scalers[f'{symbol}_lstm'] = scaler
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        self.models[f'{symbol}_lstm'] = model
        
        # Make predictions
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        
        # Inverse transform predictions
        train_predictions_inv = scaler.inverse_transform(
            np.concatenate([train_predictions, np.zeros((train_predictions.shape[0], 4))], axis=1)
        )[:, 0]
        
        test_predictions_inv = scaler.inverse_transform(
            np.concatenate([test_predictions, np.zeros((test_predictions.shape[0], 4))], axis=1)
        )[:, 0]
        
        # Future predictions
        last_sequence = X[-1:]
        future_predictions = []
        
        for _ in range(30):  # Predict 30 days
            pred = model.predict(last_sequence, verbose=0)
            future_predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            new_row = np.concatenate([pred, last_sequence[0, -1, 1:].reshape(1, -1)], axis=1)
            last_sequence = np.concatenate([last_sequence[:, 1:, :], new_row.reshape(1, 1, -1)], axis=1)
        
        future_predictions_inv = scaler.inverse_transform(
            np.concatenate([np.array(future_predictions).reshape(-1, 1), np.zeros((30, 4))], axis=1)
        )[:, 0]
        
        result = {
            'model': model,
            'history': history.history,
            'train_predictions': train_predictions_inv,
            'test_predictions': test_predictions_inv,
            'future_predictions': future_predictions_inv,
            'train_mae': mean_absolute_error(y_train, train_predictions.flatten()),
            'test_mae': mean_absolute_error(y_test, test_predictions.flatten())
        }
        
        logger.info(f"LSTM model trained. Test MAE: {result['test_mae']:.4f}")
        return result
    
    def create_ensemble_forecast(self, symbol: str, arima_result: dict, 
                                prophet_result: dict, lstm_result: dict) -> dict:
        """Create ensemble forecast combining all models."""
        logger.info(f"Creating ensemble forecast for {symbol}")
        
        # Get forecasts from all models
        arima_forecast = arima_result['forecast'].values
        prophet_forecast = prophet_result['forecast']['yhat'].tail(30).values
        lstm_forecast = lstm_result['future_predictions']
        
        # Simple average ensemble
        ensemble_forecast = (arima_forecast + prophet_forecast + lstm_forecast) / 3
        
        # Weighted ensemble based on historical performance
        # (In practice, you'd calculate these weights from validation data)
        weights = {'arima': 0.3, 'prophet': 0.4, 'lstm': 0.3}
        weighted_ensemble = (
            arima_forecast * weights['arima'] +
            prophet_forecast * weights['prophet'] +
            lstm_forecast * weights['lstm']
        )
        
        # Create date index for forecasts
        last_date = self.data[symbol].index[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        result = {
            'simple_ensemble': pd.Series(ensemble_forecast, index=forecast_dates),
            'weighted_ensemble': pd.Series(weighted_ensemble, index=forecast_dates),
            'individual_forecasts': {
                'arima': pd.Series(arima_forecast, index=forecast_dates),
                'prophet': pd.Series(prophet_forecast, index=forecast_dates),
                'lstm': pd.Series(lstm_forecast, index=forecast_dates)
            },
            'weights': weights
        }
        
        return result
    
    def train_all_models(self, symbol: str) -> dict:
        """Train all models for a given symbol."""
        logger.info(f"Training all models for {symbol}")
        
        # Fetch data
        self.fetch_stock_data(symbol)
        
        # Train individual models
        arima_result = self.train_arima_model(symbol)
        prophet_result = self.train_prophet_model(symbol)
        lstm_result = self.train_lstm_model(symbol)
        
        # Create ensemble
        ensemble_result = self.create_ensemble_forecast(symbol, arima_result, prophet_result, lstm_result)
        
        # Store all results
        all_results = {
            'symbol': symbol,
            'arima': arima_result,
            'prophet': prophet_result,
            'lstm': lstm_result,
            'ensemble': ensemble_result,
            'data_shape': self.data[symbol].shape,
            'training_date': datetime.now().isoformat()
        }
        
        self.predictions[symbol] = all_results
        
        return all_results
    
    def save_models(self, symbol: str, output_dir: str = "models"):
        """Save trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save ARIMA model
        if f'{symbol}_arima' in self.models:
            arima_path = output_path / f"{symbol}_arima_model.pkl"
            joblib.dump(self.models[f'{symbol}_arima'], arima_path)
        
        # Save Prophet model
        if f'{symbol}_prophet' in self.models:
            prophet_path = output_path / f"{symbol}_prophet_model.pkl"
            joblib.dump(self.models[f'{symbol}_prophet'], prophet_path)
        
        # Save LSTM model
        if f'{symbol}_lstm' in self.models:
            lstm_path = output_path / f"{symbol}_lstm_model.h5"
            self.models[f'{symbol}_lstm'].save(lstm_path)
        
        # Save scaler
        if f'{symbol}_lstm' in self.scalers:
            scaler_path = output_path / f"{symbol}_scaler.pkl"
            joblib.dump(self.scalers[f'{symbol}_lstm'], scaler_path)
        
        # Save predictions
        if symbol in self.predictions:
            # Convert to JSON-serializable format
            pred_data = self.predictions[symbol].copy()
            for key in ['arima', 'prophet', 'lstm', 'ensemble']:
                if key in pred_data:
                    # Convert pandas Series to dict
                    for subkey, value in pred_data[key].items():
                        if isinstance(value, pd.Series):
                            pred_data[key][subkey] = {
                                'values': value.values.tolist(),
                                'index': value.index.strftime('%Y-%m-%d').tolist()
                            }
            
            pred_path = output_path / f"{symbol}_predictions.json"
            with open(pred_path, 'w') as f:
                json.dump(pred_data, f, indent=2, default=str)
        
        logger.info(f"Models saved for {symbol}")


# FastAPI Application
app = FastAPI(
    title="Stock Forecasting API",
    description="AI-powered stock price forecasting using ensemble methods",
    version="1.0.0"
)

# Global forecasting engine
forecasting_engine = StockForecastingEngine()

# Request/Response models
class ForecastRequest(BaseModel):
    symbol: str
    days: int = 30

class ForecastResponse(BaseModel):
    symbol: str
    current_price: float
    forecasts: Dict[str, List[float]]
    dates: List[str]
    confidence_intervals: Optional[Dict[str, Dict[str, List[float]]]] = None
    technical_indicators: Dict[str, float]

@app.post("/forecast", response_model=ForecastResponse)
async def forecast_stock(request: ForecastRequest):
    """Generate stock price forecast."""
    try:
        symbol = request.symbol.upper()
        
        # Train models and get forecasts
        results = forecasting_engine.train_all_models(symbol)
        
        # Get current price
        current_data = forecasting_engine.data[symbol]
        current_price = float(current_data['Close'].iloc[-1])
        
        # Prepare response
        ensemble_forecast = results['ensemble']['weighted_ensemble']
        forecast_dates = ensemble_forecast.index.strftime('%Y-%m-%d').tolist()
        
        forecasts = {
            'ensemble': ensemble_forecast.values.tolist(),
            'arima': results['ensemble']['individual_forecasts']['arima'].values.tolist(),
            'prophet': results['ensemble']['individual_forecasts']['prophet'].values.tolist(),
            'lstm': results['ensemble']['individual_forecasts']['lstm'].values.tolist()
        }
        
        # Technical indicators
        tech_indicators = {
            'rsi': float(current_data['RSI'].iloc[-1]),
            'sma_20': float(current_data['SMA_20'].iloc[-1]),
            'sma_50': float(current_data['SMA_50'].iloc[-1]),
            'macd': float(current_data['MACD'].iloc[-1])
        }
        
        return ForecastResponse(
            symbol=symbol,
            current_price=current_price,
            forecasts=forecasts,
            dates=forecast_dates,
            technical_indicators=tech_indicators
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Stock Forecasting API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    # Example usage
    engine = StockForecastingEngine()
    
    # Train models for a stock
    symbol = "AAPL"
    results = engine.train_all_models(symbol)
    
    print(f"\n=== Stock Forecasting Results for {symbol} ===")
    print(f"Current Price: ${results['ensemble']['individual_forecasts']['lstm'][0]:.2f}")
    print(f"30-day Ensemble Forecast: ${results['ensemble']['weighted_ensemble'].iloc[-1]:.2f}")
    
    # Save models
    engine.save_models(symbol)
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()