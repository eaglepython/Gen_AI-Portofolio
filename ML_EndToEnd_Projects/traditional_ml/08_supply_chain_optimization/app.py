"""
Supply Chain Optimization System - Complete Implementation
Advanced supply chain management with demand forecasting, inventory optimization, and logistics planning.
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
from sklearn.linear_model import LinearRegression, Ridge
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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Optimization
from scipy.optimize import minimize, linprog
import cvxpy as cp

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
from itertools import product

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SupplyChainDataGenerator:
    """Generate realistic supply chain data."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_demand_data(self, n_products: int = 50, n_days: int = 730) -> pd.DataFrame:
        """Generate synthetic demand data."""
        logger.info(f"Generating demand data for {n_products} products over {n_days} days")
        
        # Create date range
        dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
        
        # Product categories and base demand
        categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Books']
        products = []
        
        for i in range(n_products):
            category = np.random.choice(categories)
            base_demand = np.random.uniform(50, 500)
            seasonality_strength = np.random.uniform(0.1, 0.4)
            trend_strength = np.random.uniform(-0.001, 0.002)
            
            products.append({
                'product_id': f'PROD_{i:03d}',
                'category': category,
                'base_demand': base_demand,
                'seasonality_strength': seasonality_strength,
                'trend_strength': trend_strength,
                'price': np.random.uniform(10, 1000)
            })
        
        # Generate demand data
        demand_data = []
        
        for date_idx, date in enumerate(dates):
            for product in products:
                # Base demand
                demand = product['base_demand']
                
                # Trend
                demand += product['trend_strength'] * date_idx
                
                # Seasonality (weekly and yearly)
                day_of_week = date.dayofweek
                day_of_year = date.dayofyear
                
                # Weekly seasonality (higher on weekends for some categories)
                if product['category'] in ['Electronics', 'Clothing']:
                    weekly_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_week / 7)
                else:
                    weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)
                
                # Yearly seasonality
                yearly_factor = 1 + product['seasonality_strength'] * np.sin(2 * np.pi * day_of_year / 365)
                
                demand *= weekly_factor * yearly_factor
                
                # Special events and promotions
                if np.random.random() < 0.05:  # 5% chance of promotion
                    demand *= np.random.uniform(1.5, 3.0)
                
                # Add noise
                demand += np.random.normal(0, demand * 0.1)
                demand = max(0, demand)
                
                # External factors
                is_holiday = date.month == 12 and date.day in [24, 25, 31]
                is_weekend = date.dayofweek >= 5
                
                demand_data.append({
                    'date': date,
                    'product_id': product['product_id'],
                    'category': product['category'],
                    'demand': round(demand, 2),
                    'price': product['price'],
                    'is_holiday': is_holiday,
                    'is_weekend': is_weekend,
                    'day_of_week': day_of_week,
                    'month': date.month,
                    'quarter': date.quarter
                })
        
        df = pd.DataFrame(demand_data)
        logger.info(f"Generated {len(df)} demand records")
        return df
    
    def generate_inventory_data(self, demand_df: pd.DataFrame) -> pd.DataFrame:
        """Generate inventory and supply data."""
        logger.info("Generating inventory and supply data")
        
        products = demand_df['product_id'].unique()
        inventory_data = []
        
        for product_id in products:
            product_demand = demand_df[demand_df['product_id'] == product_id]
            
            # Supplier characteristics
            lead_time = np.random.randint(1, 14)  # days
            order_cost = np.random.uniform(50, 500)
            holding_cost_rate = np.random.uniform(0.1, 0.3)  # per unit per year
            stockout_cost = np.random.uniform(10, 100)  # per unit
            
            # Initial inventory
            avg_demand = product_demand['demand'].mean()
            safety_stock = avg_demand * lead_time * 1.5
            max_inventory = avg_demand * 30
            
            inventory_data.append({
                'product_id': product_id,
                'category': product_demand['category'].iloc[0],
                'lead_time_days': lead_time,
                'order_cost': order_cost,
                'holding_cost_rate': holding_cost_rate,
                'stockout_cost': stockout_cost,
                'safety_stock': round(safety_stock, 2),
                'max_inventory': round(max_inventory, 2),
                'current_inventory': round(np.random.uniform(safety_stock, max_inventory), 2),
                'reorder_point': round(avg_demand * lead_time + safety_stock, 2),
                'economic_order_qty': round(np.sqrt(2 * avg_demand * 365 * order_cost / holding_cost_rate), 2)
            })
        
        return pd.DataFrame(inventory_data)
    
    def generate_supplier_data(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """Generate supplier information."""
        logger.info("Generating supplier data")
        
        supplier_data = []
        supplier_names = ['Alpha Corp', 'Beta Industries', 'Gamma Ltd', 'Delta Supply', 'Epsilon Co']
        
        for _, product in inventory_df.iterrows():
            # Each product can have multiple suppliers
            n_suppliers = np.random.randint(1, 4)
            
            for i in range(n_suppliers):
                supplier_data.append({
                    'product_id': product['product_id'],
                    'supplier_id': f"SUP_{len(supplier_data):03d}",
                    'supplier_name': np.random.choice(supplier_names),
                    'cost_per_unit': np.random.uniform(5, 50),
                    'lead_time_days': np.random.randint(1, 21),
                    'min_order_qty': np.random.randint(10, 100),
                    'max_capacity': np.random.randint(1000, 10000),
                    'reliability_score': np.random.uniform(0.7, 1.0),
                    'quality_score': np.random.uniform(0.8, 1.0),
                    'location': np.random.choice(['Local', 'Domestic', 'International']),
                    'sustainability_score': np.random.uniform(0.5, 1.0)
                })
        
        return pd.DataFrame(supplier_data)


class DemandForecaster:
    """Advanced demand forecasting engine."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.forecast_results = {}
        
    def prepare_time_series_features(self, df: pd.DataFrame, product_id: str) -> pd.DataFrame:
        """Prepare time series features for a specific product."""
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date').reset_index(drop=True)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            product_data[f'demand_lag_{lag}'] = product_data['demand'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            product_data[f'demand_mean_{window}'] = product_data['demand'].rolling(window=window).mean()
            product_data[f'demand_std_{window}'] = product_data['demand'].rolling(window=window).std()
        
        # Trend features
        product_data['demand_trend'] = product_data['demand'].diff()
        product_data['demand_trend_7d'] = product_data['demand'] - product_data['demand'].shift(7)
        
        # Cyclical features
        product_data['sin_day_of_year'] = np.sin(2 * np.pi * product_data['date'].dt.dayofyear / 365)
        product_data['cos_day_of_year'] = np.cos(2 * np.pi * product_data['date'].dt.dayofyear / 365)
        product_data['sin_day_of_week'] = np.sin(2 * np.pi * product_data['day_of_week'] / 7)
        product_data['cos_day_of_week'] = np.cos(2 * np.pi * product_data['day_of_week'] / 7)
        
        return product_data
    
    def train_arima_model(self, df: pd.DataFrame, product_id: str) -> dict:
        """Train ARIMA model for specific product."""
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        
        demand_series = product_data['demand'].values
        
        # Simple ARIMA parameter selection
        best_aic = float('inf')
        best_params = None
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(demand_series, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_params = (p, d, q)
                    except:
                        continue
        
        # Train final model
        if best_params:
            final_model = ARIMA(demand_series, order=best_params)
            fitted_model = final_model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=30)
            
            self.models[f'{product_id}_arima'] = fitted_model
            
            return {
                'model': fitted_model,
                'params': best_params,
                'aic': best_aic,
                'forecast': forecast
            }
        
        return {}
    
    def train_prophet_model(self, df: pd.DataFrame, product_id: str) -> dict:
        """Train Prophet model for specific product."""
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        
        # Prepare Prophet data
        prophet_data = pd.DataFrame({
            'ds': product_data['date'],
            'y': product_data['demand']
        })
        
        # Add external regressors
        prophet_data['is_weekend'] = product_data['is_weekend'].astype(int)
        prophet_data['is_holiday'] = product_data['is_holiday'].astype(int)
        prophet_data['price'] = product_data['price']
        
        # Initialize Prophet
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        model.add_regressor('is_weekend')
        model.add_regressor('is_holiday')
        model.add_regressor('price')
        
        model.fit(prophet_data)
        
        # Make forecast
        future = model.make_future_dataframe(periods=30)
        future['is_weekend'] = future['ds'].dt.dayofweek >= 5
        future['is_holiday'] = ((future['ds'].dt.month == 12) & 
                               (future['ds'].dt.day.isin([24, 25, 31]))).astype(int)
        future['price'] = prophet_data['price'].iloc[-1]  # Use last known price
        
        forecast = model.predict(future)
        
        self.models[f'{product_id}_prophet'] = model
        
        return {
            'model': model,
            'forecast': forecast
        }
    
    def train_ml_models(self, df: pd.DataFrame, product_id: str) -> dict:
        """Train ML models for demand forecasting."""
        # Prepare features
        product_data = self.prepare_time_series_features(df, product_id)
        product_data = product_data.dropna()
        
        if len(product_data) < 100:
            return {}
        
        # Select features
        feature_columns = [
            'demand_lag_1', 'demand_lag_7', 'demand_lag_14', 'demand_lag_30',
            'demand_mean_7', 'demand_mean_14', 'demand_mean_30',
            'demand_std_7', 'demand_std_14', 'demand_std_30',
            'demand_trend', 'demand_trend_7d',
            'sin_day_of_year', 'cos_day_of_year', 'sin_day_of_week', 'cos_day_of_week',
            'is_weekend', 'is_holiday', 'price', 'day_of_week', 'month', 'quarter'
        ]
        
        X = product_data[feature_columns]
        y = product_data['demand']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {}
        results = {}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        models['random_forest'] = rf
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X, y)
        models['xgboost'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
        lgb_model.fit(X, y)
        models['lightgbm'] = lgb_model
        
        # Evaluate models
        for name, model in models.items():
            y_pred = model.predict(X)
            
            results[name] = {
                'model': model,
                'mae': mean_absolute_error(y, y_pred),
                'rmse': np.sqrt(mean_squared_error(y, y_pred)),
                'r2': r2_score(y, y_pred)
            }
            
            self.models[f'{product_id}_{name}'] = model
        
        return results
    
    def forecast_demand(self, df: pd.DataFrame, product_id: str, days_ahead: int = 30) -> dict:
        """Generate ensemble forecast for a product."""
        forecasts = {}
        
        # ARIMA forecast
        arima_result = self.train_arima_model(df, product_id)
        if arima_result:
            forecasts['arima'] = arima_result['forecast']
        
        # Prophet forecast
        prophet_result = self.train_prophet_model(df, product_id)
        if prophet_result:
            forecasts['prophet'] = prophet_result['forecast']['yhat'].tail(days_ahead).values
        
        # ML models forecast
        ml_results = self.train_ml_models(df, product_id)
        
        # Create ensemble forecast
        if forecasts:
            # Simple average of available forecasts
            forecast_arrays = [np.array(f) for f in forecasts.values()]
            min_length = min(len(f) for f in forecast_arrays)
            
            ensemble_forecast = np.mean([f[:min_length] for f in forecast_arrays], axis=0)
            
            # Generate future dates
            last_date = df[df['product_id'] == product_id]['date'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead)
            
            result = {
                'product_id': product_id,
                'forecast_dates': future_dates.tolist(),
                'ensemble_forecast': ensemble_forecast.tolist(),
                'individual_forecasts': {k: v.tolist() if hasattr(v, 'tolist') else list(v) 
                                       for k, v in forecasts.items()},
                'forecast_period': days_ahead
            }
            
            self.forecast_results[product_id] = result
            return result
        
        return {}


class InventoryOptimizer:
    """Inventory optimization engine."""
    
    def __init__(self):
        self.optimization_results = {}
        
    def calculate_eoq(self, annual_demand: float, order_cost: float, holding_cost_rate: float) -> float:
        """Calculate Economic Order Quantity."""
        return np.sqrt(2 * annual_demand * order_cost / holding_cost_rate)
    
    def calculate_reorder_point(self, daily_demand: float, lead_time: int, safety_factor: float = 1.65) -> float:
        """Calculate reorder point with safety stock."""
        return daily_demand * lead_time * safety_factor
    
    def optimize_inventory_levels(self, inventory_df: pd.DataFrame, demand_forecasts: dict) -> dict:
        """Optimize inventory levels for all products."""
        logger.info("Optimizing inventory levels")
        
        optimization_results = {}
        
        for _, product in inventory_df.iterrows():
            product_id = product['product_id']
            
            # Get demand forecast
            if product_id in demand_forecasts:
                forecast_data = demand_forecasts[product_id]
                avg_daily_demand = np.mean(forecast_data['ensemble_forecast'])
                annual_demand = avg_daily_demand * 365
            else:
                # Fallback to current estimates
                annual_demand = 1000  # Default estimate
                avg_daily_demand = annual_demand / 365
            
            # Calculate optimal parameters
            eoq = self.calculate_eoq(
                annual_demand, 
                product['order_cost'], 
                product['holding_cost_rate']
            )
            
            reorder_point = self.calculate_reorder_point(
                avg_daily_demand, 
                product['lead_time_days']
            )
            
            # Safety stock calculation
            demand_std = avg_daily_demand * 0.2  # Assume 20% coefficient of variation
            safety_stock = 1.65 * demand_std * np.sqrt(product['lead_time_days'])
            
            # Service level optimization
            target_service_level = 0.95
            stockout_probability = 1 - target_service_level
            
            optimization_results[product_id] = {
                'current_inventory': product['current_inventory'],
                'optimal_eoq': round(eoq, 2),
                'optimal_reorder_point': round(reorder_point, 2),
                'recommended_safety_stock': round(safety_stock, 2),
                'avg_daily_demand': round(avg_daily_demand, 2),
                'annual_demand': round(annual_demand, 2),
                'target_service_level': target_service_level,
                'order_recommendation': 'ORDER' if product['current_inventory'] <= reorder_point else 'HOLD',
                'order_quantity': round(eoq, 2) if product['current_inventory'] <= reorder_point else 0
            }
        
        self.optimization_results = optimization_results
        return optimization_results
    
    def optimize_multi_echelon_inventory(self, network_config: dict) -> dict:
        """Optimize multi-echelon inventory network."""
        # Simplified multi-echelon optimization
        # In practice, this would involve complex mathematical programming
        
        locations = network_config.get('locations', ['warehouse', 'dc1', 'dc2', 'retail'])
        products = network_config.get('products', list(self.optimization_results.keys())[:10])
        
        # Create optimization variables
        inventory_allocation = {}
        
        for location in locations:
            for product in products:
                if product in self.optimization_results:
                    base_qty = self.optimization_results[product]['optimal_eoq']
                    
                    # Simple allocation based on location type
                    if location == 'warehouse':
                        allocation = base_qty * 0.5
                    elif location.startswith('dc'):
                        allocation = base_qty * 0.3
                    else:  # retail
                        allocation = base_qty * 0.2
                    
                    inventory_allocation[f'{location}_{product}'] = round(allocation, 2)
        
        return {
            'allocation': inventory_allocation,
            'total_locations': len(locations),
            'total_products': len(products),
            'optimization_objective': 'minimize_total_cost'
        }


class SupplyChainOptimizer:
    """Complete supply chain optimization system."""
    
    def __init__(self):
        self.data_generator = SupplyChainDataGenerator()
        self.demand_forecaster = DemandForecaster()
        self.inventory_optimizer = InventoryOptimizer()
        
        self.demand_data = None
        self.inventory_data = None
        self.supplier_data = None
        self.is_initialized = False
        
    def initialize_system(self, n_products: int = 50, n_days: int = 730) -> dict:
        """Initialize the supply chain system with data."""
        logger.info("Initializing supply chain system")
        
        # Generate data
        self.demand_data = self.data_generator.generate_demand_data(n_products, n_days)
        self.inventory_data = self.data_generator.generate_inventory_data(self.demand_data)
        self.supplier_data = self.data_generator.generate_supplier_data(self.inventory_data)
        
        self.is_initialized = True
        
        return {
            'demand_records': len(self.demand_data),
            'products': n_products,
            'suppliers': len(self.supplier_data),
            'date_range': f"{self.demand_data['date'].min()} to {self.demand_data['date'].max()}"
        }
    
    def forecast_all_products(self, days_ahead: int = 30) -> dict:
        """Generate demand forecasts for all products."""
        if not self.is_initialized:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        logger.info(f"Forecasting demand for all products, {days_ahead} days ahead")
        
        products = self.demand_data['product_id'].unique()
        all_forecasts = {}
        
        for product_id in products[:10]:  # Limit for demo
            try:
                forecast = self.demand_forecaster.forecast_demand(
                    self.demand_data, product_id, days_ahead
                )
                if forecast:
                    all_forecasts[product_id] = forecast
            except Exception as e:
                logger.warning(f"Failed to forecast for {product_id}: {e}")
        
        return all_forecasts
    
    def optimize_entire_supply_chain(self, days_ahead: int = 30) -> dict:
        """Perform end-to-end supply chain optimization."""
        logger.info("Performing complete supply chain optimization")
        
        if not self.is_initialized:
            self.initialize_system()
        
        # Step 1: Demand Forecasting
        demand_forecasts = self.forecast_all_products(days_ahead)
        
        # Step 2: Inventory Optimization
        inventory_optimization = self.inventory_optimizer.optimize_inventory_levels(
            self.inventory_data, demand_forecasts
        )
        
        # Step 3: Multi-echelon Optimization
        network_config = {
            'locations': ['central_warehouse', 'dc_east', 'dc_west', 'retail_stores'],
            'products': list(demand_forecasts.keys())
        }
        
        multi_echelon_optimization = self.inventory_optimizer.optimize_multi_echelon_inventory(
            network_config
        )
        
        # Step 4: Supplier Selection and Order Planning
        order_recommendations = self.generate_order_recommendations(inventory_optimization)
        
        # Step 5: Cost Analysis
        cost_analysis = self.calculate_total_costs(inventory_optimization, demand_forecasts)
        
        return {
            'demand_forecasts': demand_forecasts,
            'inventory_optimization': inventory_optimization,
            'multi_echelon_optimization': multi_echelon_optimization,
            'order_recommendations': order_recommendations,
            'cost_analysis': cost_analysis,
            'optimization_date': datetime.now().isoformat()
        }
    
    def generate_order_recommendations(self, inventory_optimization: dict) -> dict:
        """Generate purchase order recommendations."""
        order_recommendations = []
        
        for product_id, optimization in inventory_optimization.items():
            if optimization['order_recommendation'] == 'ORDER':
                # Find best supplier
                product_suppliers = self.supplier_data[
                    self.supplier_data['product_id'] == product_id
                ]
                
                if not product_suppliers.empty:
                    # Simple supplier selection (lowest cost with good reliability)
                    best_supplier = product_suppliers.loc[
                        (product_suppliers['reliability_score'] > 0.8) &
                        (product_suppliers['cost_per_unit'] == product_suppliers['cost_per_unit'].min())
                    ].iloc[0] if len(product_suppliers) > 0 else product_suppliers.iloc[0]
                    
                    order_recommendations.append({
                        'product_id': product_id,
                        'supplier_id': best_supplier['supplier_id'],
                        'supplier_name': best_supplier['supplier_name'],
                        'order_quantity': optimization['order_quantity'],
                        'unit_cost': best_supplier['cost_per_unit'],
                        'total_cost': optimization['order_quantity'] * best_supplier['cost_per_unit'],
                        'lead_time_days': best_supplier['lead_time_days'],
                        'priority': 'HIGH' if optimization['current_inventory'] < optimization['recommended_safety_stock'] else 'MEDIUM'
                    })
        
        return {
            'orders': order_recommendations,
            'total_orders': len(order_recommendations),
            'total_value': sum(order['total_cost'] for order in order_recommendations)
        }
    
    def calculate_total_costs(self, inventory_optimization: dict, demand_forecasts: dict) -> dict:
        """Calculate total supply chain costs."""
        total_holding_cost = 0
        total_order_cost = 0
        total_stockout_cost = 0
        
        for product_id, optimization in inventory_optimization.items():
            # Holding cost
            avg_inventory = optimization['optimal_eoq'] / 2 + optimization['recommended_safety_stock']
            product_info = self.inventory_data[self.inventory_data['product_id'] == product_id].iloc[0]
            holding_cost = avg_inventory * product_info['holding_cost_rate']
            total_holding_cost += holding_cost
            
            # Order cost
            if optimization['order_recommendation'] == 'ORDER':
                total_order_cost += product_info['order_cost']
            
            # Estimate stockout cost
            if optimization['current_inventory'] < optimization['recommended_safety_stock']:
                shortfall = optimization['recommended_safety_stock'] - optimization['current_inventory']
                stockout_cost = shortfall * product_info['stockout_cost']
                total_stockout_cost += stockout_cost
        
        return {
            'total_holding_cost': round(total_holding_cost, 2),
            'total_order_cost': round(total_order_cost, 2),
            'total_stockout_cost': round(total_stockout_cost, 2),
            'total_cost': round(total_holding_cost + total_order_cost + total_stockout_cost, 2),
            'cost_breakdown': {
                'holding_cost_pct': round(total_holding_cost / (total_holding_cost + total_order_cost + total_stockout_cost + 1e-8) * 100, 2),
                'order_cost_pct': round(total_order_cost / (total_holding_cost + total_order_cost + total_stockout_cost + 1e-8) * 100, 2),
                'stockout_cost_pct': round(total_stockout_cost / (total_holding_cost + total_order_cost + total_stockout_cost + 1e-8) * 100, 2)
            }
        }


# FastAPI Application
app = FastAPI(
    title="Supply Chain Optimization API",
    description="Advanced supply chain management with demand forecasting and inventory optimization",
    version="1.0.0"
)

# Global supply chain optimizer
supply_chain_optimizer = SupplyChainOptimizer()

# Request/Response models
class OptimizationRequest(BaseModel):
    n_products: int = Field(default=50, ge=10, le=100)
    n_days: int = Field(default=730, ge=365, le=1095)
    forecast_days: int = Field(default=30, ge=7, le=90)

class ProductForecastRequest(BaseModel):
    product_id: str
    days_ahead: int = Field(default=30, ge=7, le=90)

class InventoryRequest(BaseModel):
    product_id: str
    current_inventory: float = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=1, le=30)
    order_cost: float = Field(..., gt=0)
    holding_cost_rate: float = Field(..., gt=0, le=1)

@app.post("/initialize")
async def initialize_system(request: OptimizationRequest):
    """Initialize the supply chain system."""
    try:
        result = supply_chain_optimizer.initialize_system(
            request.n_products, 
            request.n_days
        )
        return {
            "message": "Supply chain system initialized",
            "details": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize")
async def optimize_supply_chain(forecast_days: int = 30, background_tasks: BackgroundTasks = None):
    """Perform complete supply chain optimization."""
    try:
        if not supply_chain_optimizer.is_initialized:
            raise HTTPException(status_code=400, detail="System not initialized. Call /initialize first.")
        
        if background_tasks:
            background_tasks.add_task(
                supply_chain_optimizer.optimize_entire_supply_chain, 
                forecast_days
            )
            return {
                "message": "Optimization started",
                "status": "processing",
                "forecast_days": forecast_days
            }
        else:
            results = supply_chain_optimizer.optimize_entire_supply_chain(forecast_days)
            return {
                "message": "Optimization completed",
                "results": results
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def forecast_demand(request: ProductForecastRequest):
    """Forecast demand for a specific product."""
    try:
        if not supply_chain_optimizer.is_initialized:
            raise HTTPException(status_code=400, detail="System not initialized.")
        
        forecast = supply_chain_optimizer.demand_forecaster.forecast_demand(
            supply_chain_optimizer.demand_data,
            request.product_id,
            request.days_ahead
        )
        
        if not forecast:
            raise HTTPException(status_code=404, detail=f"Could not generate forecast for {request.product_id}")
        
        return forecast
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inventory/optimize")
async def optimize_inventory(request: InventoryRequest):
    """Optimize inventory for a specific product."""
    try:
        # Calculate EOQ and reorder point
        eoq = supply_chain_optimizer.inventory_optimizer.calculate_eoq(
            annual_demand=10000,  # Default estimate
            order_cost=request.order_cost,
            holding_cost_rate=request.holding_cost_rate
        )
        
        reorder_point = supply_chain_optimizer.inventory_optimizer.calculate_reorder_point(
            daily_demand=27.4,  # Default estimate  
            lead_time=request.lead_time_days
        )
        
        order_needed = request.current_inventory <= reorder_point
        
        return {
            "product_id": request.product_id,
            "current_inventory": request.current_inventory,
            "optimal_eoq": round(eoq, 2),
            "reorder_point": round(reorder_point, 2),
            "order_needed": order_needed,
            "order_quantity": round(eoq, 2) if order_needed else 0,
            "safety_stock": round(reorder_point * 0.3, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_system_status():
    """Get system status and current data."""
    try:
        if not supply_chain_optimizer.is_initialized:
            return {"status": "not_initialized", "message": "System not initialized"}
        
        status = {
            "status": "initialized",
            "products": len(supply_chain_optimizer.inventory_data),
            "suppliers": len(supply_chain_optimizer.supplier_data),
            "demand_records": len(supply_chain_optimizer.demand_data),
            "forecasting_models": len(supply_chain_optimizer.demand_forecaster.models),
            "optimization_results": len(supply_chain_optimizer.inventory_optimizer.optimization_results)
        }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products")
async def get_products():
    """Get list of all products."""
    try:
        if not supply_chain_optimizer.is_initialized:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        products = supply_chain_optimizer.inventory_data[['product_id', 'category', 'current_inventory']].to_dict('records')
        return {"products": products, "total_count": len(products)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/suppliers")
async def get_suppliers():
    """Get list of all suppliers."""
    try:
        if not supply_chain_optimizer.is_initialized:
            raise HTTPException(status_code=400, detail="System not initialized")
        
        suppliers = supply_chain_optimizer.supplier_data.to_dict('records')
        return {"suppliers": suppliers, "total_count": len(suppliers)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Supply Chain Optimization API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting Supply Chain Optimization System")
    
    # Example usage
    optimizer = SupplyChainOptimizer()
    
    print("\n=== Supply Chain Optimization System ===")
    print("Initializing system...")
    
    # Initialize system
    init_result = optimizer.initialize_system(n_products=20, n_days=365)
    print(f"Initialized with {init_result['products']} products")
    
    print("\nRunning optimization...")
    results = optimizer.optimize_entire_supply_chain(days_ahead=30)
    
    print("\nOptimization Results:")
    print(f"Products forecasted: {len(results['demand_forecasts'])}")
    print(f"Products optimized: {len(results['inventory_optimization'])}")
    print(f"Total orders recommended: {results['order_recommendations']['total_orders']}")
    print(f"Total cost: ${results['cost_analysis']['total_cost']:,.2f}")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()