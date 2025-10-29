"""
Fraud Detection System - Complete Implementation
Real-time fraud detection using ensemble ML methods and anomaly detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn

# Utilities
import joblib
import pickle
import json
from pathlib import Path
import logging
from contextlib import asynccontextmanager
import asyncio
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDataGenerator:
    """Generate realistic synthetic fraud detection data."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_transaction_data(self, n_samples: int = 10000, fraud_rate: float = 0.02) -> pd.DataFrame:
        """Generate synthetic transaction data."""
        logger.info(f"Generating {n_samples} transaction samples with {fraud_rate:.1%} fraud rate")
        
        # Generate base features
        data = {
            'transaction_id': [f'TXN_{i:06d}' for i in range(n_samples)],
            'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='5min'),
            'amount': np.random.lognormal(mean=3, sigma=1.5, size=n_samples),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online', 'atm'], size=n_samples),
            'card_type': np.random.choice(['credit', 'debit'], size=n_samples, p=[0.7, 0.3]),
            'is_weekend': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
            'hour_of_day': np.random.randint(0, 24, size=n_samples),
        }
        
        # Customer features
        data['customer_age'] = np.random.normal(40, 15, size=n_samples).astype(int)
        data['customer_age'] = np.clip(data['customer_age'], 18, 80)
        
        data['customer_income'] = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples)
        data['account_age_days'] = np.random.exponential(scale=800, size=n_samples).astype(int)
        
        # Location features
        data['merchant_country'] = np.random.choice(['US', 'CA', 'UK', 'FR', 'DE', 'JP'], size=n_samples, 
                                                   p=[0.7, 0.1, 0.08, 0.06, 0.04, 0.02])
        data['customer_country'] = np.random.choice(['US', 'CA', 'UK', 'FR', 'DE', 'JP'], size=n_samples,
                                                   p=[0.75, 0.1, 0.06, 0.05, 0.03, 0.01])
        
        # Payment method
        data['payment_method'] = np.random.choice(['chip', 'swipe', 'contactless', 'online'], size=n_samples,
                                                 p=[0.4, 0.2, 0.25, 0.15])
        
        df = pd.DataFrame(data)
        
        # Generate fraud labels
        n_fraud = int(n_samples * fraud_rate)
        fraud_indices = np.random.choice(n_samples, size=n_fraud, replace=False)
        df['is_fraud'] = 0
        df.loc[fraud_indices, 'is_fraud'] = 1
        
        # Modify fraud transactions to be more suspicious
        fraud_mask = df['is_fraud'] == 1
        
        # Fraudulent transactions tend to be larger amounts
        df.loc[fraud_mask, 'amount'] *= np.random.uniform(2, 10, size=n_fraud)
        
        # More likely at unusual hours
        df.loc[fraud_mask, 'hour_of_day'] = np.random.choice([1, 2, 3, 22, 23], size=n_fraud)
        
        # More likely to be online
        df.loc[fraud_mask, 'payment_method'] = np.random.choice(['online', 'contactless'], size=n_fraud, p=[0.8, 0.2])
        
        # Country mismatches
        fraud_country_mismatch = np.random.choice([True, False], size=n_fraud, p=[0.3, 0.7])
        df.loc[fraud_mask & fraud_country_mismatch, 'merchant_country'] = np.random.choice(['XX', 'YY', 'ZZ'], size=np.sum(fraud_country_mismatch))
        
        return df


class FraudFeatureEngineer:
    """Advanced feature engineering for fraud detection."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_month_end'] = (df['timestamp'].dt.day > 25).astype(int)
        df['is_month_start'] = (df['timestamp'].dt.day <= 5).astype(int)
        
        # Hour categories
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)
        
        return df
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer behavior features."""
        df = df.copy()
        df_sorted = df.sort_values(['timestamp'])
        
        # Transaction frequency features
        df['transactions_last_hour'] = 0
        df['transactions_last_day'] = 0
        df['total_amount_last_hour'] = 0.0
        df['total_amount_last_day'] = 0.0
        
        # For demonstration, create simplified versions
        # In real implementation, you'd calculate these properly
        df['avg_transaction_amount'] = df.groupby('customer_age')['amount'].transform('mean')
        df['std_transaction_amount'] = df.groupby('customer_age')['amount'].transform('std')
        
        # Amount deviation from personal average
        df['amount_deviation'] = (df['amount'] - df['avg_transaction_amount']) / (df['std_transaction_amount'] + 1e-8)
        
        # Income ratio
        df['amount_to_income_ratio'] = df['amount'] / (df['customer_income'] / 12)  # Monthly income
        
        return df
    
    def create_merchant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-related features."""
        df = df.copy()
        
        # Merchant statistics
        merchant_stats = df.groupby('merchant_category').agg({
            'amount': ['mean', 'std', 'count'],
            'is_fraud': 'mean'
        }).round(4)
        
        merchant_stats.columns = ['merchant_avg_amount', 'merchant_std_amount', 'merchant_transaction_count', 'merchant_fraud_rate']
        merchant_stats = merchant_stats.reset_index()
        
        # Merge back
        df = df.merge(merchant_stats, on='merchant_category', how='left')
        
        # Amount deviation from merchant average
        df['amount_vs_merchant_avg'] = (df['amount'] - df['merchant_avg_amount']) / (df['merchant_std_amount'] + 1e-8)
        
        # Country mismatch
        df['country_mismatch'] = (df['customer_country'] != df['merchant_country']).astype(int)
        
        return df
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create risk-based features."""
        df = df.copy()
        
        # High risk indicators
        df['high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        df['very_high_amount'] = (df['amount'] > df['amount'].quantile(0.99)).astype(int)
        
        # Unusual timing
        df['unusual_hour'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
        
        # Payment method risk
        payment_risk = {'online': 3, 'contactless': 2, 'swipe': 1, 'chip': 0}
        df['payment_risk_score'] = df['payment_method'].map(payment_risk)
        
        # Combine risk factors
        df['risk_score'] = (
            df['high_amount'] * 2 +
            df['country_mismatch'] * 3 +
            df['unusual_hour'] * 1 +
            df['payment_risk_score'] +
            df['is_weekend']
        )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        df = df.copy()
        
        categorical_features = ['merchant_category', 'card_type', 'merchant_country', 'customer_country', 'payment_method']
        
        for feature in categorical_features:
            if feature in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[f'{feature}_encoded'] = le.fit_transform(df[feature].astype(str))
                    self.encoders[feature] = le
                else:
                    if feature in self.encoders:
                        # Handle unseen categories
                        le = self.encoders[feature]
                        df[f'{feature}_encoded'] = df[feature].astype(str).apply(
                            lambda x: le.transform([x])[0] if x in le.classes_ else -1
                        )
                    else:
                        df[f'{feature}_encoded'] = 0
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        df = df.copy()
        
        numerical_features = [
            'amount', 'customer_age', 'customer_income', 'account_age_days',
            'amount_deviation', 'amount_to_income_ratio', 'amount_vs_merchant_avg', 'risk_score'
        ]
        
        # Only scale features that exist
        numerical_features = [f for f in numerical_features if f in df.columns]
        
        if fit:
            scaler = RobustScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                df[numerical_features] = self.scalers['numerical'].transform(df[numerical_features])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Complete feature engineering pipeline."""
        logger.info("Starting feature engineering")
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create customer features
        df = self.create_customer_features(df)
        
        # Create merchant features
        df = self.create_merchant_features(df)
        
        # Create risk features
        df = self.create_risk_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_numerical_features(df, fit=fit)
        
        logger.info(f"Feature engineering completed. Shape: {df.shape}")
        return df


class FraudModelTrainer:
    """Train and evaluate fraud detection models."""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.thresholds = {}
        
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for training."""
        # Select features for training
        feature_columns = [
            'amount', 'customer_age', 'customer_income', 'account_age_days',
            'hour_of_day', 'day_of_week', 'month', 'is_weekend', 'is_night',
            'is_business_hours', 'merchant_category_encoded', 'card_type_encoded',
            'payment_method_encoded', 'country_mismatch', 'risk_score',
            'amount_deviation', 'amount_to_income_ratio', 'high_amount'
        ]
        
        # Only use features that exist
        available_features = [f for f in feature_columns if f in df.columns]
        
        X = df[available_features]
        y = df['is_fraud'] if 'is_fraud' in df.columns else None
        
        return X, y, available_features
    
    def train_isolation_forest(self, X: pd.DataFrame, contamination: float = 0.02) -> dict:
        """Train Isolation Forest for anomaly detection."""
        logger.info("Training Isolation Forest")
        
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto'
        )
        
        model.fit(X)
        
        # Get anomaly scores
        anomaly_scores = model.decision_function(X)
        predictions = model.predict(X)
        
        # Convert predictions (-1 for outliers, 1 for inliers)
        fraud_predictions = (predictions == -1).astype(int)
        
        self.models['isolation_forest'] = model
        
        result = {
            'model': model,
            'anomaly_scores': anomaly_scores,
            'predictions': fraud_predictions,
            'contamination': contamination
        }
        
        return result
    
    def train_supervised_models(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train supervised ML models."""
        logger.info("Training supervised models")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        models = {}
        results = {}
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train_balanced, y_train_balanced)
        models['logistic_regression'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight='balanced'
        )
        rf.fit(X_train_balanced, y_train_balanced)
        models['random_forest'] = rf
        
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
            y_pred = model.predict(X_test)
            
            # Find optimal threshold
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Predictions with optimal threshold
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'optimal_threshold': optimal_threshold,
                'classification_report': classification_report(y_test, y_pred_optimal, output_dict=True),
                'predictions_proba': y_pred_proba,
                'predictions': y_pred_optimal
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
            
            self.thresholds[name] = optimal_threshold
        
        self.models.update(models)
        return results
    
    def train_deep_learning_model(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train deep learning model."""
        logger.info("Training deep learning model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build neural network
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
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
        
        # Calculate class weights
        class_weights = {
            0: 1.0,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
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
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'optimal_threshold': optimal_threshold,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return result
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series = None) -> dict:
        """Create ensemble model combining multiple approaches."""
        logger.info("Creating ensemble model")
        
        if not self.models:
            raise ValueError("No trained models available for ensemble")
        
        ensemble_predictions = []
        model_weights = {
            'isolation_forest': 0.2,
            'logistic_regression': 0.2,
            'random_forest': 0.2,
            'xgboost': 0.2,
            'lightgbm': 0.15,
            'neural_network': 0.05
        }
        
        # Get predictions from each model
        for model_name, weight in model_weights.items():
            if model_name in self.models:
                model = self.models[model_name]
                
                if model_name == 'isolation_forest':
                    # Anomaly score (convert to probability-like score)
                    scores = model.decision_function(X)
                    proba = (scores - scores.min()) / (scores.max() - scores.min())
                    proba = 1 - proba  # Invert so higher = more likely fraud
                elif model_name == 'neural_network':
                    proba = model.predict(X, verbose=0).flatten()
                else:
                    proba = model.predict_proba(X)[:, 1]
                
                ensemble_predictions.append(proba * weight)
        
        # Combine predictions
        if ensemble_predictions:
            final_predictions = np.sum(ensemble_predictions, axis=0)
            
            # Find optimal threshold if labels provided
            if y is not None:
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
                    'accuracy': accuracy_score(y, y_pred),
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


class FraudDetectionEngine:
    """Main fraud detection engine."""
    
    def __init__(self):
        self.data_generator = FraudDataGenerator()
        self.feature_engineer = FraudFeatureEngineer()
        self.model_trainer = FraudModelTrainer()
        self.training_data = None
        self.is_trained = False
        
    def generate_and_prepare_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate and prepare training data."""
        logger.info("Generating and preparing data")
        
        # Generate data
        raw_data = self.data_generator.generate_transaction_data(n_samples)
        
        # Engineer features
        processed_data = self.feature_engineer.engineer_features(raw_data, fit=True)
        
        self.training_data = processed_data
        return processed_data
    
    def train_all_models(self, data: pd.DataFrame = None) -> dict:
        """Train all fraud detection models."""
        if data is None:
            data = self.training_data
        
        if data is None:
            raise ValueError("No training data available")
        
        logger.info("Training all fraud detection models")
        
        # Prepare features
        X, y, feature_names = self.model_trainer.prepare_features(data)
        
        results = {
            'data_shape': data.shape,
            'feature_count': len(feature_names),
            'fraud_rate': y.mean() if y is not None else None,
            'models': {}
        }
        
        # Train isolation forest (unsupervised)
        if_result = self.model_trainer.train_isolation_forest(X)
        results['models']['isolation_forest'] = if_result
        
        # Train supervised models
        if y is not None:
            supervised_results = self.model_trainer.train_supervised_models(X, y)
            results['models'].update(supervised_results)
            
            # Train deep learning model
            dl_result = self.model_trainer.train_deep_learning_model(X, y)
            results['models']['neural_network'] = dl_result
            
            # Create ensemble
            ensemble_result = self.model_trainer.create_ensemble_model(X, y)
            results['models']['ensemble'] = ensemble_result
        
        self.is_trained = True
        logger.info("All models trained successfully")
        
        return results
    
    def predict_fraud(self, transaction_data: Union[Dict, pd.DataFrame], 
                     model_name: str = 'ensemble') -> Dict:
        """Predict fraud for new transaction(s)."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_all_models() first.")
        
        # Convert to DataFrame if needed
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
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
            fraud_probability = ensemble_result['predictions_proba'][0] if len(ensemble_result['predictions_proba']) > 0 else 0.5
            threshold = self.model_trainer.thresholds.get('ensemble', 0.5)
        else:
            model = self.model_trainer.models[model_name]
            threshold = self.model_trainer.thresholds.get(model_name, 0.5)
            
            if model_name == 'isolation_forest':
                score = model.decision_function(X)[0]
                fraud_probability = 1 / (1 + np.exp(-score))  # Convert to probability-like
            elif model_name == 'neural_network':
                fraud_probability = model.predict(X, verbose=0)[0][0]
            else:
                fraud_probability = model.predict_proba(X)[0, 1]
        
        is_fraud = fraud_probability >= threshold
        
        # Risk level
        if fraud_probability >= 0.8:
            risk_level = 'HIGH'
        elif fraud_probability >= 0.5:
            risk_level = 'MEDIUM'
        elif fraud_probability >= 0.3:
            risk_level = 'LOW'
        else:
            risk_level = 'VERY_LOW'
        
        result = {
            'transaction_id': df.iloc[0].get('transaction_id', 'unknown'),
            'fraud_probability': float(fraud_probability),
            'is_fraud': bool(is_fraud),
            'risk_level': risk_level,
            'threshold': float(threshold),
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def save_models(self, output_dir: str = "fraud_models"):
        """Save trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature engineer
        joblib.dump(self.feature_engineer, output_path / "feature_engineer.pkl")
        
        # Save models
        for name, model in self.model_trainer.models.items():
            if name == 'neural_network':
                model.save(output_path / f"{name}_model.h5")
            else:
                joblib.dump(model, output_path / f"{name}_model.pkl")
        
        # Save thresholds and metadata
        metadata = {
            'thresholds': self.model_trainer.thresholds,
            'feature_importance': self.model_trainer.feature_importance,
            'is_trained': self.is_trained
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Models saved to {output_dir}")


# FastAPI Application
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system using ensemble ML methods",
    version="1.0.0"
)

# Global fraud detection engine
fraud_engine = FraudDetectionEngine()

# Request/Response models
class TransactionData(BaseModel):
    transaction_id: Optional[str] = None
    amount: float = Field(..., gt=0)
    merchant_category: str
    card_type: str
    customer_age: int = Field(..., ge=18, le=100)
    customer_income: float = Field(..., gt=0)
    account_age_days: int = Field(..., ge=0)
    is_weekend: int = Field(..., ge=0, le=1)
    hour_of_day: int = Field(..., ge=0, le=23)
    merchant_country: str = "US"
    customer_country: str = "US"
    payment_method: str

class FraudPredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float
    model_used: str
    timestamp: str

class TrainingRequest(BaseModel):
    n_samples: int = Field(default=10000, gt=100)
    fraud_rate: float = Field(default=0.02, gt=0, lt=1)

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: TransactionData, model_name: str = "ensemble"):
    """Predict fraud for a single transaction."""
    try:
        if not fraud_engine.is_trained:
            raise HTTPException(status_code=400, detail="Models not trained. Call /train first.")
        
        # Convert to dict
        transaction_dict = transaction.dict()
        if not transaction_dict.get('transaction_id'):
            transaction_dict['transaction_id'] = f"TXN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Add timestamp
        transaction_dict['timestamp'] = datetime.now()
        
        # Predict
        result = fraud_engine.predict_fraud(transaction_dict, model_name)
        
        return FraudPredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train fraud detection models."""
    try:
        # Generate and prepare data
        data = fraud_engine.generate_and_prepare_data(request.n_samples)
        
        # Train models in background
        background_tasks.add_task(fraud_engine.train_all_models, data)
        
        return {
            "message": "Training started",
            "data_shape": data.shape,
            "fraud_rate": data['is_fraud'].mean(),
            "status": "training_in_progress"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_status")
async def get_model_status():
    """Get model training status and performance."""
    try:
        if not fraud_engine.is_trained:
            return {"status": "not_trained", "message": "Models not trained yet"}
        
        # Get model performance metrics
        performance = {}
        for name, threshold in fraud_engine.model_trainer.thresholds.items():
            performance[name] = {
                "threshold": threshold,
                "feature_importance": fraud_engine.model_trainer.feature_importance.get(name, {})
            }
        
        return {
            "status": "trained",
            "models_available": list(fraud_engine.model_trainer.models.keys()),
            "performance": performance,
            "data_shape": fraud_engine.training_data.shape if fraud_engine.training_data is not None else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(transactions: List[TransactionData], model_name: str = "ensemble"):
    """Predict fraud for multiple transactions."""
    try:
        if not fraud_engine.is_trained:
            raise HTTPException(status_code=400, detail="Models not trained. Call /train first.")
        
        if len(transactions) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 transactions per batch")
        
        results = []
        for i, transaction in enumerate(transactions):
            transaction_dict = transaction.dict()
            if not transaction_dict.get('transaction_id'):
                transaction_dict['transaction_id'] = f"BATCH_{i:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            transaction_dict['timestamp'] = datetime.now()
            result = fraud_engine.predict_fraud(transaction_dict, model_name)
            results.append(result)
        
        return {
            "predictions": results,
            "total_transactions": len(results),
            "fraud_detected": sum(1 for r in results if r['is_fraud']),
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
    return {"message": "Fraud Detection API", "docs": "/docs"}

def main():
    """Main function to run the application."""
    logger.info("Starting Fraud Detection System")
    
    # Example usage
    engine = FraudDetectionEngine()
    
    print("\n=== Fraud Detection System ===")
    print("Generating training data...")
    
    # Generate data and train models
    data = engine.generate_and_prepare_data(5000)
    print(f"Generated {len(data)} transactions")
    print(f"Fraud rate: {data['is_fraud'].mean():.2%}")
    
    print("\nTraining models...")
    results = engine.train_all_models(data)
    
    print("\nModel Performance:")
    for model_name, model_result in results['models'].items():
        if 'accuracy' in model_result:
            print(f"{model_name}: {model_result['accuracy']:.4f} accuracy")
    
    # Test prediction
    sample_transaction = {
        'amount': 1500.0,
        'merchant_category': 'online',
        'card_type': 'credit',
        'customer_age': 35,
        'customer_income': 50000,
        'account_age_days': 365,
        'is_weekend': 0,
        'hour_of_day': 2,
        'merchant_country': 'XX',
        'customer_country': 'US',
        'payment_method': 'online'
    }
    
    prediction = engine.predict_fraud(sample_transaction)
    print(f"\nSample prediction: {prediction['fraud_probability']:.3f} ({prediction['risk_level']})")
    
    print("\nStarting API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()