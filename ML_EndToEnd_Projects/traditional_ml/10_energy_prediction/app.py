"""
Energy Consumption Prediction System - Complete Implementation
Advanced energy forecasting using IoT data, weather patterns, and time series analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Time series
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Tuple
import uvicorn

# Utilities
import joblib
import pickle
import json
from pathlib import Path
import logging
import asyncio
import random
from scipy import signal

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyDataGenerator:
    """Generate realistic energy consumption and IoT sensor data."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_weather_data(self, start_date: str, n_days: int) -> pd.DataFrame:
        """Generate synthetic weather data."""
        dates = pd.date_range(start=start_date, periods=n_days, freq='H')
        
        weather_data = []
        
        for i, timestamp in enumerate(dates):
            # Base temperature with seasonal variation
            day_of_year = timestamp.dayofyear
            hour_of_day = timestamp.hour
            
            # Seasonal temperature pattern
            base_temp = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
            
            # Daily temperature variation
            daily_variation = 8 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
            
            temperature = base_temp + daily_variation + np.random.normal(0, 2)
            
            # Humidity (inversely related to temperature)
            humidity = 80 - temperature * 0.8 + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 95)
            
            # Wind speed
            wind_speed = np.abs(np.random.normal(10, 5))
            
            # Solar irradiance (only during day hours)
            if 6 <= hour_of_day <= 18:
                solar_peak = 1000 * np.sin(np.pi * (hour_of_day - 6) / 12)
                cloud_factor = np.random.uniform(0.3, 1.0)
                solar_irradiance = solar_peak * cloud_factor
            else:
                solar_irradiance = 0
            
            # Precipitation
            precipitation = np.random.exponential(0.1) if np.random.random() < 0.1 else 0
            
            weather_data.append({
                'timestamp': timestamp,
                'temperature': round(temperature, 2),
                'humidity': round(humidity, 2),
                'wind_speed': round(wind_speed, 2),
                'solar_irradiance': round(solar_irradiance, 2),
                'precipitation': round(precipitation, 2),
                'pressure': round(1013 + np.random.normal(0, 5), 2)
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_building_data(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Generate building characteristics and occupancy data."""
        
        building_data = []
        
        for _, weather_row in weather_df.iterrows():
            timestamp = weather_row['timestamp']
            hour = timestamp.hour
            day_of_week = timestamp.dayofweek
            
            # Occupancy patterns
            if day_of_week < 5:  # Weekdays
                if 7 <= hour <= 9:  # Morning rush
                    occupancy = np.random.uniform(0.8, 1.0)
                elif 10 <= hour <= 16:  # Work hours
                    occupancy = np.random.uniform(0.6, 0.9)
                elif 17 <= hour <= 19:  # Evening
                    occupancy = np.random.uniform(0.5, 0.8)
                else:  # Night/early morning
                    occupancy = np.random.uniform(0.1, 0.3)
            else:  # Weekends
                if 8 <= hour <= 22:
                    occupancy = np.random.uniform(0.3, 0.7)
                else:
                    occupancy = np.random.uniform(0.1, 0.4)
            
            # Building characteristics
            building_area = 10000  # sq ft
            building_age = 15  # years
            insulation_rating = 0.8  # 0-1 scale
            
            building_data.append({
                'timestamp': timestamp,
                'occupancy_rate': round(occupancy, 3),
                'building_area': building_area,
                'building_age': building_age,
                'insulation_rating': insulation_rating,
                'hvac_setpoint': 22.0,  # Celsius
                'lighting_usage': round(occupancy * np.random.uniform(0.7, 1.0), 3)
            })
        
        return pd.DataFrame(building_data)
    
    def generate_iot_sensor_data(self, weather_df: pd.DataFrame, building_df: pd.DataFrame) -> pd.DataFrame:
        """Generate IoT sensor readings."""
        
        iot_data = []
        
        for i, (weather_row, building_row) in enumerate(zip(weather_df.itertuples(), building_df.itertuples())):
            timestamp = weather_row.timestamp
            
            # Indoor temperature (influenced by outdoor temp, HVAC, occupancy)
            outdoor_temp = weather_row.temperature
            occupancy = building_row.occupancy_rate
            hvac_setpoint = building_row.hvac_setpoint
            
            # HVAC influence
            hvac_cooling = max(0, outdoor_temp - hvac_setpoint) * 0.5
            hvac_heating = max(0, hvac_setpoint - outdoor_temp) * 0.3
            
            # Occupancy heat generation
            occupancy_heat = occupancy * 2.0
            
            indoor_temp = hvac_setpoint + (outdoor_temp - hvac_setpoint) * 0.2 + occupancy_heat - hvac_cooling + hvac_heating
            indoor_temp += np.random.normal(0, 0.5)
            
            # CO2 levels (influenced by occupancy)
            base_co2 = 400  # ppm
            occupancy_co2 = occupancy * 600
            co2_level = base_co2 + occupancy_co2 + np.random.normal(0, 50)
            
            # Air quality index
            outdoor_pollution = np.random.normal(50, 20)  # Base AQI
            ventilation_effect = occupancy * 10
            air_quality = outdoor_pollution + ventilation_effect + np.random.normal(0, 10)
            air_quality = max(0, air_quality)
            
            # Equipment status
            hvac_status = 1 if abs(indoor_temp - hvac_setpoint) > 1.0 else 0
            lighting_power = building_row.lighting_usage * 5000  # Watts
            
            iot_data.append({
                'timestamp': timestamp,
                'indoor_temperature': round(indoor_temp, 2),
                'indoor_humidity': round(weather_row.humidity * 0.9, 2),
                'co2_level': round(co2_level, 1),
                'air_quality_index': round(air_quality, 1),
                'hvac_status': hvac_status,
                'lighting_power': round(lighting_power, 1),
                'occupancy_count': round(occupancy * 100),  # Estimated people count
                'window_status': np.random.choice([0, 1], p=[0.8, 0.2]),  # 0=closed, 1=open
                'equipment_load': round(occupancy * np.random.uniform(1000, 3000), 1)  # Watts
            })
        
        return pd.DataFrame(iot_data)
    
    def generate_energy_consumption(self, weather_df: pd.DataFrame, building_df: pd.DataFrame, 
                                   iot_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic energy consumption data."""
        
        energy_data = []
        
        for weather_row, building_row, iot_row in zip(weather_df.itertuples(), building_df.itertuples(), iot_df.itertuples()):
            timestamp = weather_row.timestamp
            
            # Base load (always present)
            base_load = 2000  # Watts
            
            # HVAC consumption (major component)
            outdoor_temp = weather_row.temperature
            indoor_temp = iot_row.indoor_temperature
            hvac_setpoint = building_row.hvac_setpoint
            
            # Cooling load
            if outdoor_temp > hvac_setpoint:
                cooling_load = (outdoor_temp - hvac_setpoint) * 500 * building_row.occupancy_rate
                cooling_load *= (1 - building_row.insulation_rating * 0.3)  # Insulation effect
            else:
                cooling_load = 0
            
            # Heating load
            if outdoor_temp < hvac_setpoint:
                heating_load = (hvac_setpoint - outdoor_temp) * 400 * building_row.occupancy_rate
                heating_load *= (1 - building_row.insulation_rating * 0.3)
            else:
                heating_load = 0
            
            # Lighting consumption
            lighting_consumption = iot_row.lighting_power
            
            # Equipment consumption
            equipment_consumption = iot_row.equipment_load
            
            # Solar generation (if applicable)
            solar_generation = weather_row.solar_irradiance * 0.2 * np.random.uniform(0.8, 1.2)  # Watts
            
            # Total consumption
            total_consumption = (base_load + cooling_load + heating_load + 
                               lighting_consumption + equipment_consumption - solar_generation)
            
            # Add some realistic noise
            total_consumption += np.random.normal(0, total_consumption * 0.05)
            total_consumption = max(0, total_consumption)
            
            # Calculate costs (variable rate structure)
            hour = timestamp.hour
            if 9 <= hour <= 21:  # Peak hours
                rate_per_kwh = 0.15
            else:  # Off-peak hours
                rate_per_kwh = 0.08
            
            cost = total_consumption * rate_per_kwh / 1000  # Convert W to kW
            
            energy_data.append({
                'timestamp': timestamp,
                'total_consumption_watts': round(total_consumption, 2),
                'hvac_consumption': round(cooling_load + heating_load, 2),
                'lighting_consumption': round(lighting_consumption, 2),
                'equipment_consumption': round(equipment_consumption, 2),
                'solar_generation': round(solar_generation, 2),
                'net_consumption': round(total_consumption, 2),
                'cost_usd': round(cost, 4),
                'rate_per_kwh': rate_per_kwh,
                'peak_hour': 1 if 9 <= hour <= 21 else 0
            })
        
        return pd.DataFrame(energy_data)
    
    def generate_complete_dataset(self, start_date: str = '2023-01-01', n_days: int = 365) -> Dict[str, pd.DataFrame]:
        """Generate complete energy dataset with all components."""
        logger.info(f"Generating complete energy dataset for {n_days} days")
        
        # Generate all data components
        weather_df = self.generate_weather_data(start_date, n_days)
        building_df = self.generate_building_data(weather_df)
        iot_df = self.generate_iot_sensor_data(weather_df, building_df)
        energy_df = self.generate_energy_consumption(weather_df, building_df, iot_df)
        
        # Combine all data
        combined_df = weather_df.merge(building_df, on='timestamp')\
                               .merge(iot_df, on='timestamp')\
                               .merge(energy_df, on='timestamp')
        
        logger.info(f"Generated {len(combined_df)} hourly records")
        
        return {
            'weather': weather_df,
            'building': building_df,
            'iot': iot_df,
            'energy': energy_df,
            'combined': combined_df
        }


class EnergyFeatureEngineer:
    """Advanced feature engineering for energy prediction."""
    
    def __init__(self):
        self.scalers = {}
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive time-based features."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['year'] = df['timestamp'].dt.year
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary time features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        df['is_peak_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 21)).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Season classification
        df['season'] = df['month'] % 12 // 3 + 1  # 1=Winter, 2=Spring, 3=Summer, 4=Fall
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'total_consumption_watts') -> pd.DataFrame:
        """Create lag and rolling window features."""
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Lag features
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # Up to 1 week
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12, 24, 48, 168]:
            df[f'{target_col}_mean_{window}h'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_std_{window}h'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_min_{window}h'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_max_{window}h'] = df[target_col].rolling(window=window).max()
        
        # Difference features
        df[f'{target_col}_diff_1h'] = df[target_col].diff(1)
        df[f'{target_col}_diff_24h'] = df[target_col].diff(24)
        df[f'{target_col}_diff_168h'] = df[target_col].diff(168)
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create weather-derived features."""
        df = df.copy()
        
        # Comfort indices
        df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] - 10)
        df['wind_chill'] = df['temperature'] - 0.7 * df['wind_speed']
        
        # Temperature deviation from comfort zone
        comfort_temp = 22  # Celsius
        df['temp_deviation'] = abs(df['temperature'] - comfort_temp)
        df['temp_above_comfort'] = np.maximum(0, df['temperature'] - comfort_temp)
        df['temp_below_comfort'] = np.maximum(0, comfort_temp - df['temperature'])
        
        # Cooling/Heating degree hours
        df['cooling_degree_hours'] = np.maximum(0, df['temperature'] - 18)
        df['heating_degree_hours'] = np.maximum(0, 18 - df['temperature'])
        
        # Weather interactions
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
        df['solar_temp_interaction'] = df['solar_irradiance'] * df['temperature'] / 100
        
        # Weather categories
        df['temp_category'] = pd.cut(df['temperature'], 
                                   bins=[-np.inf, 10, 20, 30, np.inf],
                                   labels=['Cold', 'Cool', 'Warm', 'Hot'])
        
        df['humidity_category'] = pd.cut(df['humidity'],
                                       bins=[0, 30, 60, 80, 100],
                                       labels=['Low', 'Medium', 'High', 'Very_High'])
        
        return df
    
    def create_building_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create building and occupancy features."""
        df = df.copy()
        
        # Occupancy patterns
        df['occupancy_density'] = df['occupancy_rate'] / df['building_area'] * 10000
        df['occupancy_lag_1h'] = df['occupancy_rate'].shift(1)
        df['occupancy_change'] = df['occupancy_rate'].diff()
        
        # HVAC efficiency features
        df['indoor_outdoor_temp_diff'] = abs(df['indoor_temperature'] - df['temperature'])
        df['hvac_load_efficiency'] = df['hvac_consumption'] / (df['indoor_outdoor_temp_diff'] + 1)
        
        # Air quality interactions
        df['co2_occupancy_ratio'] = df['co2_level'] / (df['occupancy_rate'] + 0.1)
        df['air_quality_category'] = pd.cut(df['air_quality_index'],
                                          bins=[0, 50, 100, 150, np.inf],
                                          labels=['Good', 'Moderate', 'Poor', 'Unhealthy'])
        
        return df
    
    def create_energy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create energy-specific features."""
        df = df.copy()
        
        # Energy efficiency metrics
        df['energy_per_occupant'] = df['total_consumption_watts'] / (df['occupancy_count'] + 1)
        df['lighting_efficiency'] = df['lighting_consumption'] / (df['occupancy_rate'] + 0.1)
        df['hvac_efficiency'] = df['hvac_consumption'] / (df['temp_deviation'] + 1)
        
        # Solar utilization
        df['solar_utilization_rate'] = df['solar_generation'] / (df['solar_irradiance'] + 1)
        df['net_solar_benefit'] = df['solar_generation'] - df['total_consumption_watts'] * 0.1
        
        # Cost features
        df['cost_per_kwh_actual'] = df['cost_usd'] / (df['total_consumption_watts'] / 1000 + 0.001)
        df['daily_cost_estimate'] = df['cost_usd'] * 24
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        # Select numerical columns for scaling
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude target and ID columns
        exclude_cols = ['timestamp', 'total_consumption_watts']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if fit:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                df[numerical_cols] = self.scalers['numerical'].transform(df[numerical_cols])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering")
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create lag features
        df = self.create_lag_features(df)
        
        # Create weather features
        df = self.create_weather_features(df)
        
        # Create building features
        df = self.create_building_features(df)
        
        # Create energy features
        df = self.create_energy_features(df)
        
        # Handle categorical features
        categorical_columns = df.select_dtypes(include=['category', 'object']).columns
        for col in categorical_columns:
            if col != 'timestamp':
                df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
        
        # Scale features
        df = self.scale_features(df, fit=fit)
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df


class EnergyModelTrainer:
    """Train and evaluate energy prediction models."""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.evaluation_results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for model training."""
        # Remove rows with NaN values (from lag features)
        df_clean = df.dropna()
        
        # Select features for training
        exclude_columns = ['timestamp', 'total_consumption_watts'] + \
                         [col for col in df_clean.columns if col.endswith('_category') and not col.endswith('_encoded')]
        
        feature_columns = [col for col in df_clean.columns if col not in exclude_columns]
        
        X = df_clean[feature_columns]
        y = df_clean['total_consumption_watts'] if 'total_consumption_watts' in df_clean.columns else None
        
        return X, y, feature_columns
    
    def train_time_series_models(self, df: pd.DataFrame) -> dict:
        """Train time series specific models."""
        logger.info("Training time series models")
        
        # Prepare data
        df_sorted = df.sort_values('timestamp')
        energy_series = df_sorted['total_consumption_watts'].dropna()
        
        results = {}
        
        # ARIMA Model
        try:
            # Simple ARIMA parameter selection
            model = ARIMA(energy_series, order=(2, 1, 2))
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=24)
            
            results['arima'] = {
                'model': fitted_model,
                'forecast': forecast,
                'aic': fitted_model.aic
            }
            
            self.models['arima'] = fitted_model
            
        except Exception as e:
            logger.warning(f"ARIMA training failed: {e}")
        
        # Prophet Model
        try:
            # Prepare Prophet data
            prophet_data = pd.DataFrame({
                'ds': df_sorted['timestamp'],
                'y': df_sorted['total_consumption_watts']
            }).dropna()
            
            # Add regressors
            prophet_data['temperature'] = df_sorted['temperature'].iloc[:len(prophet_data)]
            prophet_data['occupancy_rate'] = df_sorted['occupancy_rate'].iloc[:len(prophet_data)]
            
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            
            model.add_regressor('temperature')
            model.add_regressor('occupancy_rate')
            model.fit(prophet_data)
            
            # Make forecast
            future = model.make_future_dataframe(periods=24, freq='H')
            future['temperature'] = prophet_data['temperature'].iloc[-1]  # Use last known values
            future['occupancy_rate'] = prophet_data['occupancy_rate'].iloc[-1]
            
            forecast = model.predict(future)
            
            results['prophet'] = {
                'model': model,
                'forecast': forecast
            }
            
            self.models['prophet'] = model
            
        except Exception as e:
            logger.warning(f"Prophet training failed: {e}")
        
        return results
    
    def train_ml_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train machine learning models."""
        logger.info("Training ML models")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split data
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        models = {}
        results = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['random_forest'] = rf
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # Support Vector Regression
        svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        svr_model.fit(X_train, y_train)
        models['svr'] = svr_model
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        models['gradient_boosting'] = gb_model
        
        # Evaluate models
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            self.models[name] = model
        
        return results
    
    def train_deep_learning_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train deep learning models."""
        logger.info("Training deep learning models")
        
        # Prepare data for neural networks
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Scale targets
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        results = {}
        
        # LSTM Model
        try:
            # Prepare sequences for LSTM
            def create_sequences(data, target, seq_length):
                X_seq, y_seq = [], []
                for i in range(len(data) - seq_length):
                    X_seq.append(data[i:(i + seq_length)])
                    y_seq.append(target[i + seq_length])
                return np.array(X_seq), np.array(y_seq)
            
            seq_length = 24  # 24 hours
            X_train_seq, y_train_seq = create_sequences(X_train.values, y_train_scaled, seq_length)
            X_test_seq, y_test_seq = create_sequences(X_test.values, y_test_scaled, seq_length)
            
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[1])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
            
            # Train
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_data=(X_test_seq, y_test_seq),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            y_pred_scaled = lstm_model.predict(X_test_seq, verbose=0)
            y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()
            y_test_actual = target_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
            
            results['lstm'] = {
                'model': lstm_model,
                'target_scaler': target_scaler,
                'mae': mean_absolute_error(y_test_actual, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred)),
                'r2': r2_score(y_test_actual, y_pred),
                'history': history.history
            }
            
            self.models['lstm'] = lstm_model
            
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")
        
        # Dense Neural Network
        try:
            dense_model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            dense_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train
            history = dense_model.fit(
                X_train, y_train_scaled,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test_scaled),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            y_pred_scaled = dense_model.predict(X_test, verbose=0).flatten()
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            
            results['dense_nn'] = {
                'model': dense_model,
                'target_scaler': target_scaler,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'history': history.history
            }
            
            self.models['dense_nn'] = dense_model
            
        except Exception as e:
            logger.warning(f"Dense NN training failed: {e}")
        
        return results


class EnergyPredictionEngine:
    """Complete energy prediction system."""
    
    def __init__(self):
        self.data_generator = EnergyDataGenerator()
        self.feature_engineer = EnergyFeatureEngineer()
        self.model_trainer = EnergyModelTrainer()
        
        self.data = None
        self.processed_data = None
        self.is_trained = False
        
    def initialize_system(self, start_date: str = '2023-01-01', n_days: int = 365) -> dict:
        """Initialize the energy prediction system."""
        logger.info("Initializing energy prediction system")
        
        # Generate comprehensive dataset
        self.data = self.data_generator.generate_complete_dataset(start_date, n_days)
        
        # Feature engineering
        self.processed_data = self.feature_engineer.engineer_features(
            self.data['combined'], fit=True
        )
        
        return {
            'total_records': len(self.processed_data),
            'date_range': f"{start_date} to {pd.to_datetime(start_date) + timedelta(days=n_days)}",
            'features_created': self.processed_data.shape[1],
            'avg_consumption': self.processed_data['total_consumption_watts'].mean()
        }
    
    def train_all_models(self) -> dict:
        """Train all energy prediction models."""
        if self.processed_data is None:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        logger.info("Training all energy prediction models")
        
        # Prepare features
        X, y, feature_names = self.model_trainer.prepare_features(self.processed_data)
        
        results = {
            'data_shape': self.processed_data.shape,
            'feature_count': len(feature_names),
            'models': {}
        }
        
        # Train time series models
        ts_results = self.model_trainer.train_time_series_models(self.processed_data)
        results['models'].update(ts_results)
        
        # Train ML models
        ml_results = self.model_trainer.train_ml_models(X, y)
        results['models'].update(ml_results)
        
        # Train deep learning models
        dl_results = self.model_trainer.train_deep_learning_models(X, y)
        results['models'].update(dl_results)
        
        self.is_trained = True
        logger.info("All models trained successfully")
        
        return results
    
    def predict_energy_consumption(self, input_data: Dict, hours_ahead: int = 24) -> Dict:
        """Predict energy consumption."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
        
        # Feature engineering
        processed_input = self.feature_engineer.engineer_features(input_df, fit=False)
        
        # Prepare features
        X, _, _ = self.model_trainer.prepare_features(processed_input)
        
        if X.empty:
            raise ValueError("Could not prepare features from input data")
        
        # Make predictions with available models
        predictions = {}
        
        for model_name, model in self.model_trainer.models.items():
            try:
                if model_name in ['arima', 'prophet']:
                    # Time series models handle their own prediction logic
                    if model_name == 'arima':
                        pred = model.forecast(steps=hours_ahead)
                        predictions[model_name] = pred.tolist() if hasattr(pred, 'tolist') else [float(pred)]
                    elif model_name == 'prophet':
                        future = model.make_future_dataframe(periods=hours_ahead, freq='H')
                        # Add last known regressor values
                        future['temperature'] = input_data.get('temperature', 20)
                        future['occupancy_rate'] = input_data.get('occupancy_rate', 0.5)
                        forecast = model.predict(future)
                        predictions[model_name] = forecast['yhat'].tail(hours_ahead).tolist()
                
                elif model_name in ['lstm', 'dense_nn']:
                    # Neural network models
                    if model_name == 'lstm':
                        # For LSTM, we need sequence data - simplified for demo
                        # In practice, you'd need the last 24 hours of data
                        pred = model.predict(X.values.reshape(1, 1, -1), verbose=0)[0][0]
                    else:  # dense_nn
                        pred = model.predict(X, verbose=0)[0][0]
                    
                    predictions[model_name] = [float(pred)]
                
                else:
                    # Traditional ML models
                    pred = model.predict(X)[0]
                    predictions[model_name] = [float(pred)]
                    
            except Exception as e:
                logger.warning(f"Prediction failed for {model_name}: {e}")
        
        # Ensemble prediction
        if predictions:
            # Simple average of available predictions
            all_preds = []
            for model_preds in predictions.values():
                if isinstance(model_preds, list) and len(model_preds) > 0:
                    all_preds.append(model_preds[0])
            
            ensemble_pred = np.mean(all_preds) if all_preds else 0
        else:
            ensemble_pred = 0
        
        return {
            'timestamp': input_data['timestamp'],
            'predicted_consumption_watts': round(ensemble_pred, 2),
            'individual_predictions': predictions,
            'prediction_confidence': 'high' if len(predictions) >= 3 else 'medium' if len(predictions) >= 2 else 'low'
        }


# FastAPI Application
app = FastAPI(
    title="Energy Consumption Prediction API",
    description="Advanced energy forecasting using IoT data, weather patterns, and time series analysis",
    version="1.0.0"
)

# Global energy prediction engine
energy_engine = EnergyPredictionEngine()

# Request/Response models
class EnergyPredictionRequest(BaseModel):
    timestamp: str
    temperature: float = Field(..., ge=-50, le=60)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(default=5.0, ge=0)
    solar_irradiance: float = Field(default=0, ge=0, le=1200)
    occupancy_rate: float = Field(..., ge=0, le=1)
    hvac_setpoint: float = Field(default=22.0, ge=15, le=30)
    indoor_temperature: float = Field(default=22.0, ge=10, le=35)
    equipment_load: float = Field(default=1000, ge=0)

class EnergyPredictionResponse(BaseModel):
    timestamp: str
    predicted_consumption_watts: float
    individual_predictions: Dict
    prediction_confidence: str

@app.post("/initialize")
async def initialize_system(start_date: str = "2023-01-01", n_days: int = 365):
    """Initialize the energy prediction system."""
    try:
        result = energy_engine.initialize_system(start_date, n_days)
        return {
            "message": "Energy prediction system initialized",
            "details": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(background_tasks: BackgroundTasks = None):
    """Train energy prediction models."""
    try:
        if energy_engine.processed_data is None:
            raise HTTPException(status_code=400, detail="System not initialized. Call /initialize first.")
        
        if background_tasks:
            background_tasks.add_task(energy_engine.train_all_models)
            return {
                "message": "Training started",
                "status": "processing"
            }
        else:
            results = energy_engine.train_all_models()
            return {
                "message": "Training completed",
                "models_trained": list(results['models'].keys()),
                "feature_count": results['feature_count']
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=EnergyPredictionResponse)
async def predict_energy(request: EnergyPredictionRequest):
    """Predict energy consumption."""
    try:
        if not energy_engine.is_trained:
            raise HTTPException(status_code=400, detail="Models not trained. Call /train first.")
        
        # Convert request to dict
        input_data = request.dict()
        
        # Make prediction
        result = energy_engine.predict_energy_consumption(input_data)
        
        return EnergyPredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_system_status():
    """Get system status."""
    try:
        if energy_engine.processed_data is None:
            return {"status": "not_initialized", "message": "System not initialized"}
        
        if not energy_engine.is_trained:
            return {
                "status": "initialized", 
                "message": "System initialized but models not trained",
                "data_records": len(energy_engine.processed_data)
            }
        
        return {
            "status": "ready",
            "message": "System ready for predictions",
            "data_records": len(energy_engine.processed_data),
            "models_trained": list(energy_engine.model_trainer.models.keys()),
            "feature_count": energy_engine.processed_data.shape[1]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Energy Consumption Prediction API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting Energy Consumption Prediction System")
    
    # Example usage
    engine = EnergyPredictionEngine()
    
    print("\n=== Energy Consumption Prediction System ===")
    print("Initializing system...")
    
    # Initialize system
    init_result = engine.initialize_system(n_days=30)  # Smaller dataset for demo
    print(f"Initialized with {init_result['total_records']} records")
    print(f"Features created: {init_result['features_created']}")
    
    print("\nTraining models...")
    results = engine.train_all_models()
    
    print("\nModel Performance:")
    for model_name, model_result in results['models'].items():
        if 'mae' in model_result:
            print(f"{model_name}: MAE = {model_result['mae']:.2f}")
    
    # Test prediction
    sample_input = {
        'timestamp': '2023-01-31 14:00:00',
        'temperature': 25.0,
        'humidity': 60.0,
        'wind_speed': 8.0,
        'solar_irradiance': 800.0,
        'occupancy_rate': 0.8,
        'hvac_setpoint': 22.0,
        'indoor_temperature': 23.0,
        'equipment_load': 2000.0
    }
    
    prediction = engine.predict_energy_consumption(sample_input)
    print(f"\nSample prediction: {prediction['predicted_consumption_watts']:.2f} watts")
    print(f"Confidence: {prediction['prediction_confidence']}")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()