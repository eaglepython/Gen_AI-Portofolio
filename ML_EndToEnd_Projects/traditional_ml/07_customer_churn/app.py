"""
Customer Churn Prediction System - Complete Implementation
Advanced churn prediction using behavioral analytics and ensemble ML methods.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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
from typing import List, Dict, Optional, Union
import uvicorn

# Utilities
import joblib
import pickle
import json
from pathlib import Path
import logging
import asyncio
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnDataGenerator:
    """Generate realistic customer churn data."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_customer_data(self, n_customers: int = 10000, churn_rate: float = 0.2) -> pd.DataFrame:
        """Generate synthetic customer data with realistic churn patterns."""
        logger.info(f"Generating {n_customers} customers with {churn_rate:.1%} churn rate")
        
        # Basic demographics
        data = {
            'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
            'age': np.random.normal(40, 15, n_customers).astype(int),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),
            'partner': np.random.choice([0, 1], n_customers, p=[0.52, 0.48]),
            'dependents': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
        }
        
        # Subscription details
        data['tenure_months'] = np.random.exponential(scale=24, size=n_customers).astype(int)
        data['tenure_months'] = np.clip(data['tenure_months'], 1, 72)
        
        data['contract'] = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, 
                                          p=[0.55, 0.21, 0.24])
        data['paperless_billing'] = np.random.choice([0, 1], n_customers, p=[0.41, 0.59])
        data['payment_method'] = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], 
                                                n_customers, p=[0.34, 0.23, 0.22, 0.21])
        
        # Services
        data['phone_service'] = np.random.choice([0, 1], n_customers, p=[0.1, 0.9])
        data['multiple_lines'] = np.random.choice([0, 1], n_customers, p=[0.48, 0.52])
        data['internet_service'] = np.random.choice(['No', 'DSL', 'Fiber optic'], n_customers, p=[0.21, 0.34, 0.45])
        data['online_security'] = np.random.choice([0, 1], n_customers, p=[0.5, 0.5])
        data['online_backup'] = np.random.choice([0, 1], n_customers, p=[0.56, 0.44])
        data['device_protection'] = np.random.choice([0, 1], n_customers, p=[0.56, 0.44])
        data['tech_support'] = np.random.choice([0, 1], n_customers, p=[0.51, 0.49])
        data['streaming_tv'] = np.random.choice([0, 1], n_customers, p=[0.5, 0.5])
        data['streaming_movies'] = np.random.choice([0, 1], n_customers, p=[0.5, 0.5])
        
        # Financial
        data['monthly_charges'] = np.random.normal(65, 30, n_customers)
        data['monthly_charges'] = np.clip(data['monthly_charges'], 18.8, 118.75)
        
        data['total_charges'] = data['tenure_months'] * np.array(data['monthly_charges']) + \
                              np.random.normal(0, 100, n_customers)
        data['total_charges'] = np.maximum(data['total_charges'], 0)
        
        # Usage patterns
        data['avg_monthly_gb'] = np.random.lognormal(mean=3, sigma=1, size=n_customers)
        data['peak_usage_hours'] = np.random.normal(3, 1.5, n_customers)
        data['peak_usage_hours'] = np.clip(data['peak_usage_hours'], 0, 10)
        
        data['support_tickets'] = np.random.poisson(lam=1.5, size=n_customers)
        data['late_payments'] = np.random.poisson(lam=0.8, size=n_customers)
        data['service_outages'] = np.random.poisson(lam=2, size=n_customers)
        
        # Engagement metrics
        data['login_frequency'] = np.random.exponential(scale=15, size=n_customers)
        data['feature_usage_score'] = np.random.beta(2, 5, size=n_customers) * 100
        data['satisfaction_score'] = np.random.normal(7, 2, n_customers)
        data['satisfaction_score'] = np.clip(data['satisfaction_score'], 1, 10)
        
        df = pd.DataFrame(data)
        
        # Clip age
        df['age'] = np.clip(df['age'], 18, 80)
        
        # Generate churn labels with realistic patterns
        churn_probability = np.zeros(n_customers)
        
        # Factors that increase churn probability
        churn_probability += (df['contract'] == 'Month-to-month') * 0.3
        churn_probability += (df['tenure_months'] < 6) * 0.25
        churn_probability += (df['monthly_charges'] > df['monthly_charges'].quantile(0.8)) * 0.2
        churn_probability += (df['support_tickets'] > 3) * 0.15
        churn_probability += (df['late_payments'] > 2) * 0.15
        churn_probability += (df['satisfaction_score'] < 5) * 0.2
        churn_probability += (df['senior_citizen'] == 1) * 0.1
        churn_probability += (df['payment_method'] == 'Electronic check') * 0.1
        churn_probability += (df['internet_service'] == 'Fiber optic') * 0.05
        
        # Factors that decrease churn probability  
        churn_probability -= (df['contract'] == 'Two year') * 0.2
        churn_probability -= (df['tenure_months'] > 24) * 0.15
        churn_probability -= (df['partner'] == 1) * 0.05
        churn_probability -= (df['dependents'] == 1) * 0.05
        churn_probability -= (df['online_security'] == 1) * 0.05
        churn_probability -= (df['tech_support'] == 1) * 0.05
        
        # Normalize to [0, 1]
        churn_probability = np.clip(churn_probability, 0, 1)
        
        # Generate binary churn labels
        df['churn'] = np.random.binomial(1, churn_probability, n_customers)
        
        # Adjust to target churn rate
        current_churn_rate = df['churn'].mean()
        if current_churn_rate != churn_rate:
            # Simple adjustment
            n_churn_needed = int(n_customers * churn_rate)
            df['churn'] = 0
            churn_indices = np.random.choice(n_customers, size=n_churn_needed, replace=False, p=churn_probability/churn_probability.sum())
            df.loc[churn_indices, 'churn'] = 1
        
        logger.info(f"Generated data with actual churn rate: {df['churn'].mean():.1%}")
        return df


class ChurnFeatureEngineer:
    """Advanced feature engineering for churn prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral and engagement features."""
        df = df.copy()
        
        # Engagement metrics
        df['charges_per_month'] = df['total_charges'] / (df['tenure_months'] + 1)
        df['gb_per_dollar'] = df['avg_monthly_gb'] / (df['monthly_charges'] + 1)
        df['usage_efficiency'] = df['feature_usage_score'] / (df['monthly_charges'] + 1)
        
        # Service adoption
        services = ['phone_service', 'multiple_lines', 'online_security', 'online_backup', 
                   'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies']
        df['total_services'] = df[services].sum(axis=1)
        df['service_adoption_rate'] = df['total_services'] / len(services)
        
        # Risk indicators
        df['high_charges'] = (df['monthly_charges'] > df['monthly_charges'].quantile(0.8)).astype(int)
        df['low_satisfaction'] = (df['satisfaction_score'] < 5).astype(int)
        df['high_support_usage'] = (df['support_tickets'] > df['support_tickets'].quantile(0.8)).astype(int)
        df['payment_issues'] = (df['late_payments'] > 1).astype(int)
        df['service_issues'] = (df['service_outages'] > df['service_outages'].quantile(0.8)).astype(int)
        
        # Tenure categories
        df['tenure_category'] = pd.cut(df['tenure_months'], 
                                     bins=[0, 6, 12, 24, 48, 100], 
                                     labels=['New', 'Short', 'Medium', 'Long', 'Veteran'])
        
        # Value segments
        df['value_segment'] = pd.cut(df['monthly_charges'],
                                   bins=[0, 35, 65, 95, 200],
                                   labels=['Low', 'Medium', 'High', 'Premium'])
        
        # Age groups
        df['age_group'] = pd.cut(df['age'],
                               bins=[0, 30, 45, 60, 100],
                               labels=['Young', 'Middle', 'Mature', 'Senior'])
        
        return df
    
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create financial and pricing features."""
        df = df.copy()
        
        # Price sensitivity indicators
        df['price_per_service'] = df['monthly_charges'] / (df['total_services'] + 1)
        df['total_value'] = df['total_charges'] / (df['monthly_charges'] * df['tenure_months'] + 1)
        
        # Payment behavior
        df['payment_reliability'] = 1 / (1 + df['late_payments'])
        df['electronic_payment'] = (df['payment_method'] == 'Electronic check').astype(int)
        
        # Contract value
        contract_multiplier = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
        df['contract_value'] = df['contract'].map(contract_multiplier) * df['monthly_charges']
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        df = df.copy()
        
        # Age and tenure interactions
        df['age_tenure_interaction'] = df['age'] * df['tenure_months']
        df['senior_month_to_month'] = df['senior_citizen'] * (df['contract'] == 'Month-to-month').astype(int)
        
        # Service and satisfaction interactions
        df['service_satisfaction'] = df['total_services'] * df['satisfaction_score']
        df['support_satisfaction'] = df['support_tickets'] * (10 - df['satisfaction_score'])
        
        # Financial interactions
        df['charges_tenure'] = df['monthly_charges'] * df['tenure_months']
        df['charges_satisfaction'] = df['monthly_charges'] * (10 - df['satisfaction_score'])
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        # Binary encoding
        binary_features = ['gender', 'contract', 'payment_method', 'internet_service', 
                          'tenure_category', 'value_segment', 'age_group']
        
        for feature in binary_features:
            if feature in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                    self.encoders[feature] = le
                else:
                    if feature in self.encoders:
                        le = self.encoders[feature]
                        df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df[f'{feature}_encoded'] = 0
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        numerical_features = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges', 'avg_monthly_gb',
            'peak_usage_hours', 'support_tickets', 'late_payments', 'service_outages',
            'login_frequency', 'feature_usage_score', 'satisfaction_score',
            'charges_per_month', 'gb_per_dollar', 'usage_efficiency', 'total_services',
            'price_per_service', 'payment_reliability', 'contract_value'
        ]
        
        # Only scale features that exist
        numerical_features = [f for f in numerical_features if f in df.columns]
        
        if fit:
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                df[numerical_features] = self.scalers['numerical'].transform(df[numerical_features])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering")
        
        # Create behavioral features
        df = self.create_behavioral_features(df)
        
        # Create financial features
        df = self.create_financial_features(df)
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_features(df, fit=fit)
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df


class ChurnModelTrainer:
    """Train and evaluate churn prediction models."""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.thresholds = {}
        self.evaluation_results = {}
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for model training."""
        # Select features for training
        feature_columns = [
            'age', 'senior_citizen', 'partner', 'dependents', 'tenure_months',
            'monthly_charges', 'total_charges', 'paperless_billing',
            'phone_service', 'multiple_lines', 'online_security', 'online_backup',
            'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies',
            'avg_monthly_gb', 'peak_usage_hours', 'support_tickets', 'late_payments',
            'service_outages', 'login_frequency', 'feature_usage_score', 'satisfaction_score',
            'total_services', 'service_adoption_rate', 'high_charges', 'low_satisfaction',
            'high_support_usage', 'payment_issues', 'service_issues',
            'charges_per_month', 'gb_per_dollar', 'usage_efficiency',
            'price_per_service', 'payment_reliability', 'contract_value',
            'gender_encoded', 'contract_encoded', 'payment_method_encoded',
            'internet_service_encoded', 'tenure_category_encoded',
            'value_segment_encoded', 'age_group_encoded'
        ]
        
        # Only use features that exist
        available_features = [f for f in feature_columns if f in df.columns]
        
        X = df[available_features]
        y = df['churn'] if 'churn' in df.columns else None
        
        return X, y, available_features
    
    def train_classical_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train classical ML models."""
        logger.info("Training classical models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        models = {}
        results = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_balanced, y_train_balanced)
        models['logistic_regression'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced'
        )
        rf.fit(X_train_balanced, y_train_balanced)
        models['random_forest'] = rf
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train_balanced, y_train_balanced)
        models['gradient_boosting'] = gb
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])
        )
        xgb_model.fit(X_train_balanced, y_train_balanced)
        models['xgboost'] = xgb_model
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        lgb_model.fit(X_train_balanced, y_train_balanced)
        models['lightgbm'] = lgb_model
        
        # Evaluate models
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Find optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            results[name] = {
                'model': model,
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'optimal_threshold': optimal_threshold,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'predictions_proba': y_pred_proba,
                'y_test': y_test
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            self.thresholds[name] = optimal_threshold
        
        self.models.update(models)
        return results
    
    def train_neural_network(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train neural network model."""
        logger.info("Training neural network")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Class weights
        class_weights = {
            0: 1.0,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        
        # Find optimal threshold
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        self.models['neural_network'] = model
        self.thresholds['neural_network'] = optimal_threshold
        
        result = {
            'model': model,
            'history': history.history,
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'optimal_threshold': optimal_threshold,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return result
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series = None) -> dict:
        """Create ensemble model."""
        logger.info("Creating ensemble model")
        
        if not self.models:
            raise ValueError("No trained models available for ensemble")
        
        ensemble_predictions = []
        model_weights = {
            'logistic_regression': 0.15,
            'random_forest': 0.20,
            'gradient_boosting': 0.20,
            'xgboost': 0.25,
            'lightgbm': 0.15,
            'neural_network': 0.05
        }
        
        # Get predictions
        for model_name, weight in model_weights.items():
            if model_name in self.models:
                model = self.models[model_name]
                
                if model_name == 'neural_network':
                    proba = model.predict(X, verbose=0).flatten()
                else:
                    proba = model.predict_proba(X)[:, 1]
                
                ensemble_predictions.append(proba * weight)
        
        if ensemble_predictions:
            final_predictions = np.sum(ensemble_predictions, axis=0)
            
            if y is not None:
                # Find optimal threshold
                precision, recall, thresholds = precision_recall_curve(y, final_predictions)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                optimal_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[optimal_idx]
                
                y_pred = (final_predictions >= optimal_threshold).astype(int)
                
                result = {
                    'predictions_proba': final_predictions,
                    'predictions': y_pred,
                    'optimal_threshold': optimal_threshold,
                    'weights': model_weights,
                    'roc_auc': roc_auc_score(y, final_predictions),
                    'classification_report': classification_report(y, y_pred, output_dict=True)
                }
            else:
                result = {
                    'predictions_proba': final_predictions,
                    'weights': model_weights
                }
            
            self.thresholds['ensemble'] = result.get('optimal_threshold', 0.5)
            return result
        
        return {}


class ChurnPredictionEngine:
    """Main churn prediction engine."""
    
    def __init__(self):
        self.data_generator = ChurnDataGenerator()
        self.feature_engineer = ChurnFeatureEngineer()
        self.model_trainer = ChurnModelTrainer()
        self.training_data = None
        self.is_trained = False
        
    def generate_and_prepare_data(self, n_customers: int = 10000) -> pd.DataFrame:
        """Generate and prepare training data."""
        logger.info("Generating and preparing customer data")
        
        # Generate data
        raw_data = self.data_generator.generate_customer_data(n_customers)
        
        # Engineer features
        processed_data = self.feature_engineer.engineer_features(raw_data, fit=True)
        
        self.training_data = processed_data
        return processed_data
    
    def train_all_models(self, data: pd.DataFrame = None) -> dict:
        """Train all churn prediction models."""
        if data is None:
            data = self.training_data
        
        if data is None:
            raise ValueError("No training data available")
        
        logger.info("Training all churn prediction models")
        
        # Prepare features
        X, y, feature_names = self.model_trainer.prepare_features(data)
        
        results = {
            'data_shape': data.shape,
            'feature_count': len(feature_names),
            'churn_rate': y.mean(),
            'models': {}
        }
        
        # Train classical models
        classical_results = self.model_trainer.train_classical_models(X, y)
        results['models'].update(classical_results)
        
        # Train neural network
        nn_result = self.model_trainer.train_neural_network(X, y)
        results['models']['neural_network'] = nn_result
        
        # Create ensemble
        ensemble_result = self.model_trainer.create_ensemble_model(X, y)
        results['models']['ensemble'] = ensemble_result
        
        self.is_trained = True
        logger.info("All models trained successfully")
        
        return results
    
    def predict_churn(self, customer_data: Union[Dict, pd.DataFrame], 
                     model_name: str = 'ensemble') -> Dict:
        """Predict churn for new customer(s)."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")
        
        # Convert to DataFrame if needed
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        # Engineer features
        df_processed = self.feature_engineer.engineer_features(df, fit=False)
        
        # Prepare features
        X, _, _ = self.model_trainer.prepare_features(df_processed)
        
        # Get model
        if model_name not in self.model_trainer.models and model_name != 'ensemble':
            raise ValueError(f"Model {model_name} not found")
        
        # Make prediction
        if model_name == 'ensemble':
            ensemble_result = self.model_trainer.create_ensemble_model(X)
            churn_probability = ensemble_result['predictions_proba'][0] if len(ensemble_result['predictions_proba']) > 0 else 0.5
            threshold = self.model_trainer.thresholds.get('ensemble', 0.5)
        else:
            model = self.model_trainer.models[model_name]
            threshold = self.model_trainer.thresholds.get(model_name, 0.5)
            
            if model_name == 'neural_network':
                churn_probability = model.predict(X, verbose=0)[0][0]
            else:
                churn_probability = model.predict_proba(X)[0, 1]
        
        will_churn = churn_probability >= threshold
        
        # Risk level
        if churn_probability >= 0.8:
            risk_level = 'VERY_HIGH'
        elif churn_probability >= 0.6:
            risk_level = 'HIGH'
        elif churn_probability >= 0.4:
            risk_level = 'MEDIUM'
        elif churn_probability >= 0.2:
            risk_level = 'LOW'
        else:
            risk_level = 'VERY_LOW'
        
        result = {
            'customer_id': df.iloc[0].get('customer_id', 'unknown'),
            'churn_probability': float(churn_probability),
            'will_churn': bool(will_churn),
            'risk_level': risk_level,
            'threshold': float(threshold),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return result


# FastAPI Application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Advanced churn prediction using behavioral analytics and ensemble ML",
    version="1.0.0"
)

# Global churn engine
churn_engine = ChurnPredictionEngine()

# Request/Response models
class CustomerData(BaseModel):
    customer_id: Optional[str] = None
    age: int = Field(..., ge=18, le=100)
    gender: str = Field(..., regex="^[MF]$")
    senior_citizen: int = Field(..., ge=0, le=1)
    partner: int = Field(..., ge=0, le=1)
    dependents: int = Field(..., ge=0, le=1)
    tenure_months: int = Field(..., ge=1)
    contract: str
    paperless_billing: int = Field(..., ge=0, le=1)
    payment_method: str
    phone_service: int = Field(..., ge=0, le=1)
    multiple_lines: int = Field(..., ge=0, le=1)
    internet_service: str
    online_security: int = Field(..., ge=0, le=1)
    online_backup: int = Field(..., ge=0, le=1)
    device_protection: int = Field(..., ge=0, le=1)
    tech_support: int = Field(..., ge=0, le=1)
    streaming_tv: int = Field(..., ge=0, le=1)
    streaming_movies: int = Field(..., ge=0, le=1)
    monthly_charges: float = Field(..., gt=0)
    total_charges: float = Field(..., ge=0)
    avg_monthly_gb: float = Field(default=10.0, gt=0)
    peak_usage_hours: float = Field(default=2.0, ge=0)
    support_tickets: int = Field(default=1, ge=0)
    late_payments: int = Field(default=0, ge=0)
    service_outages: int = Field(default=1, ge=0)
    login_frequency: float = Field(default=15.0, gt=0)
    feature_usage_score: float = Field(default=50.0, ge=0, le=100)
    satisfaction_score: float = Field(default=7.0, ge=1, le=10)

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    will_churn: bool
    risk_level: str
    threshold: float
    model_used: str
    timestamp: str

@app.post("/predict", response_model=ChurnPredictionResponse)
async def predict_churn(customer: CustomerData, model_name: str = "ensemble"):
    """Predict churn for a customer."""
    try:
        if not churn_engine.is_trained:
            raise HTTPException(status_code=400, detail="Models not trained. Call /train first.")
        
        # Convert to dict
        customer_dict = customer.dict()
        if not customer_dict.get('customer_id'):
            customer_dict['customer_id'] = f"CUST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Predict
        result = churn_engine.predict_churn(customer_dict, model_name)
        
        return ChurnPredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(n_customers: int = 10000, background_tasks: BackgroundTasks = None):
    """Train churn prediction models."""
    try:
        # Generate and prepare data
        data = churn_engine.generate_and_prepare_data(n_customers)
        
        # Train models
        if background_tasks:
            background_tasks.add_task(churn_engine.train_all_models, data)
            return {
                "message": "Training started",
                "data_shape": data.shape,
                "churn_rate": data['churn'].mean(),
                "status": "training_in_progress"
            }
        else:
            results = churn_engine.train_all_models(data)
            return {
                "message": "Training completed",
                "results": {
                    "data_shape": results['data_shape'],
                    "churn_rate": results['churn_rate'],
                    "models_trained": list(results['models'].keys())
                }
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
async def get_model_status():
    """Get model training status and performance."""
    try:
        if not churn_engine.is_trained:
            return {"status": "not_trained", "message": "Models not trained yet"}
        
        # Get performance metrics
        performance = {}
        for name, threshold in churn_engine.model_trainer.thresholds.items():
            performance[name] = {
                "threshold": threshold,
                "feature_importance": churn_engine.model_trainer.feature_importance.get(name, {})
            }
        
        return {
            "status": "trained",
            "models_available": list(churn_engine.model_trainer.models.keys()),
            "performance": performance,
            "data_shape": churn_engine.training_data.shape if churn_engine.training_data is not None else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(customers: List[CustomerData], model_name: str = "ensemble"):
    """Predict churn for multiple customers."""
    try:
        if not churn_engine.is_trained:
            raise HTTPException(status_code=400, detail="Models not trained. Call /train first.")
        
        if len(customers) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 customers per batch")
        
        results = []
        for i, customer in enumerate(customers):
            customer_dict = customer.dict()
            if not customer_dict.get('customer_id'):
                customer_dict['customer_id'] = f"BATCH_{i:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            result = churn_engine.predict_churn(customer_dict, model_name)
            results.append(result)
        
        return {
            "predictions": results,
            "total_customers": len(results),
            "churn_predicted": sum(1 for r in results if r['will_churn']),
            "model_used": model_name
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
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting Customer Churn Prediction System")
    
    # Example usage
    engine = ChurnPredictionEngine()
    
    print("\n=== Customer Churn Prediction System ===")
    print("Generating training data...")
    
    # Generate data and train models
    data = engine.generate_and_prepare_data(5000)
    print(f"Generated {len(data)} customers")
    print(f"Churn rate: {data['churn'].mean():.2%}")
    
    print("\nTraining models...")
    results = engine.train_all_models(data)
    
    print("\nModel Performance:")
    for model_name, model_result in results['models'].items():
        if 'roc_auc' in model_result:
            print(f"{model_name}: {model_result['roc_auc']:.4f} ROC-AUC")
    
    # Test prediction
    sample_customer = {
        'age': 45,
        'gender': 'F',
        'senior_citizen': 0,
        'partner': 1,
        'dependents': 1,
        'tenure_months': 3,
        'contract': 'Month-to-month',
        'paperless_billing': 1,
        'payment_method': 'Electronic check',
        'phone_service': 1,
        'multiple_lines': 0,
        'internet_service': 'Fiber optic',
        'online_security': 0,
        'online_backup': 0,
        'device_protection': 0,
        'tech_support': 0,
        'streaming_tv': 1,
        'streaming_movies': 1,
        'monthly_charges': 85.0,
        'total_charges': 255.0,
        'avg_monthly_gb': 25.0,
        'peak_usage_hours': 4.0,
        'support_tickets': 3,
        'late_payments': 1,
        'service_outages': 2,
        'login_frequency': 8.0,
        'feature_usage_score': 30.0,
        'satisfaction_score': 4.0
    }
    
    prediction = engine.predict_churn(sample_customer)
    print(f"\nSample prediction: {prediction['churn_probability']:.3f} ({prediction['risk_level']})")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()