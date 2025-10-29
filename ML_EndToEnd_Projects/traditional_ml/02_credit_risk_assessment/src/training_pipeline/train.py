"""
Credit Risk Assessment - Model Training Module
Trains and evaluates multiple ML models for credit risk prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb

# Model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV

# Model interpretation
from sklearn.inspection import permutation_importance
import shap

logger = logging.getLogger(__name__)


class CreditRiskModelTrainer:
    """Train and evaluate credit risk prediction models."""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.models = {}
        self.model_performances = {}
        self.best_model = None
        self.feature_importance = {}
        
    def _get_default_config(self):
        """Get default training configuration."""
        return {
            'cv_folds': 5,
            'random_state': 42,
            'test_size': 0.2,
            'hyperparameter_tuning': True,
            'model_dir': 'models',
            'use_class_weights': True,
            'early_stopping': True
        }
    
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize ML models with default hyperparameters."""
        logger.info("Initializing ML models...")
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.config['random_state'],
                max_iter=1000,
                class_weight='balanced' if self.config['use_class_weights'] else None
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config['random_state'],
                class_weight='balanced' if self.config['use_class_weights'] else None,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config['random_state']
            ),
            
            'xgboost': xgb.XGBClassifier(
                random_state=self.config['random_state'],
                eval_metric='logloss',
                n_jobs=-1
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                random_state=self.config['random_state'],
                verbose=-1,
                n_jobs=-1
            ),
            
            'decision_tree': DecisionTreeClassifier(
                random_state=self.config['random_state'],
                class_weight='balanced' if self.config['use_class_weights'] else None
            ),
            
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            
            'naive_bayes': GaussianNB(),
            
            'svm': SVC(
                random_state=self.config['random_state'],
                probability=True,
                class_weight='balanced' if self.config['use_class_weights'] else None
            )
        }
        
        self.models = models
        logger.info(f"Initialized {len(models)} models")
        
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """Train all models."""
        logger.info("Training models...")
        
        if not self.models:
            self.initialize_models()
        
        trained_models = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Special handling for XGBoost and LightGBM with validation set
                if model_name in ['xgboost', 'lightgbm'] and X_val is not None:
                    if model_name == 'xgboost':
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=10,
                            verbose=False
                        )
                    else:  # lightgbm
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=10,
                            verbose=False
                        )
                else:
                    model.fit(X_train, y_train)
                
                trained_models[model_name] = model
                logger.info(f"Successfully trained {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            try:
                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} - AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
        
        self.model_performances = evaluation_results
        
        # Find best model
        self._find_best_model()
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': self._calculate_specificity(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['auc'] = 0.0
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
        
        # Business metrics
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _calculate_specificity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _find_best_model(self):
        """Find the best performing model based on AUC score."""
        if not self.model_performances:
            return
        
        best_auc = 0
        best_model_name = None
        
        for model_name, metrics in self.model_performances.items():
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = {
                'name': best_model_name,
                'model': self.models[best_model_name],
                'metrics': self.model_performances[best_model_name]
            }
            logger.info(f"Best model: {best_model_name} (AUC: {best_auc:.4f})")
    
    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """Perform cross-validation for all models."""
        logger.info("Performing cross-validation...")
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=self.config['cv_folds'], shuffle=True, 
                            random_state=self.config['random_state'])
        
        for model_name, model in self.models.items():
            logger.info(f"Cross-validating {model_name}...")
            
            try:
                # Calculate CV scores for multiple metrics
                scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                cv_scores = {}
                
                for metric in scoring_metrics:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                    cv_scores[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                
                cv_results[model_name] = cv_scores
                logger.info(f"{model_name} CV AUC: {cv_scores['roc_auc']['mean']:.4f} (+/- {cv_scores['roc_auc']['std']:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to cross-validate {model_name}: {e}")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                             model_names: List[str] = None) -> Dict[str, Any]:
        """Perform hyperparameter tuning for specified models."""
        if not self.config['hyperparameter_tuning']:
            logger.info("Hyperparameter tuning disabled")
            return {}
        
        logger.info("Starting hyperparameter tuning...")
        
        if model_names is None:
            # Tune only the best performing models to save time
            model_names = ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']
        
        tuned_models = {}
        param_grids = self._get_hyperparameter_grids()
        
        for model_name in model_names:
            if model_name not in self.models or model_name not in param_grids:
                continue
                
            logger.info(f"Tuning hyperparameters for {model_name}...")
            
            try:
                model = self.models[model_name]
                param_grid = param_grids[model_name]
                
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=20,  # Number of parameter settings to try
                    cv=3,  # Reduced CV folds for speed
                    scoring='roc_auc',
                    n_jobs=-1,
                    random_state=self.config['random_state']
                )
                
                search.fit(X_train, y_train)
                
                tuned_models[model_name] = {
                    'model': search.best_estimator_,
                    'best_params': search.best_params_,
                    'best_score': search.best_score_
                }
                
                # Update the main model
                self.models[model_name] = search.best_estimator_
                
                logger.info(f"{model_name} - Best CV AUC: {search.best_score_:.4f}")
                logger.info(f"{model_name} - Best params: {search.best_params_}")
                
            except Exception as e:
                logger.error(f"Failed to tune {model_name}: {e}")
        
        return tuned_models
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for tuning."""
        return {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
    
    def calculate_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, pd.Series]:
        """Calculate feature importance for all models."""
        logger.info("Calculating feature importance...")
        
        importance_results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Calculating importance for {model_name}...")
            
            try:
                if hasattr(model, 'feature_importances_'):
                    # Tree-based models
                    importance = pd.Series(model.feature_importances_, index=X.columns)
                    importance_results[model_name] = importance.sort_values(ascending=False)
                
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
                    importance_results[model_name] = importance.sort_values(ascending=False)
                
                else:
                    # Use permutation importance for other models
                    perm_importance = permutation_importance(
                        model, X, y, n_repeats=5, random_state=self.config['random_state'], n_jobs=-1
                    )
                    importance = pd.Series(perm_importance.importances_mean, index=X.columns)
                    importance_results[model_name] = importance.sort_values(ascending=False)
                
            except Exception as e:
                logger.error(f"Failed to calculate importance for {model_name}: {e}")
        
        self.feature_importance = importance_results
        return importance_results
    
    def explain_predictions(self, X_sample: pd.DataFrame, model_name: str = None) -> Dict:
        """Generate SHAP explanations for model predictions."""
        if model_name is None:
            model_name = self.best_model['name'] if self.best_model else list(self.models.keys())[0]
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        logger.info(f"Generating SHAP explanations for {model_name}...")
        
        model = self.models[model_name]
        
        try:
            # Initialize SHAP explainer
            if model_name in ['random_forest', 'xgboost', 'lightgbm', 'decision_tree']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:100])  # Use sample for speed
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class
            
            return {
                'explainer': explainer,
                'shap_values': shap_values,
                'expected_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                'feature_names': X_sample.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate SHAP explanations: {e}")
            return {}
    
    def save_models(self, output_dir: str = None):
        """Save trained models and results."""
        if output_dir is None:
            output_dir = self.config['model_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_path}")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_file = output_path / f"{model_name}_model.pkl"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} model")
        
        # Save best model separately
        if self.best_model:
            best_model_file = output_path / "best_model.pkl"
            joblib.dump(self.best_model['model'], best_model_file)
            
            # Save best model metadata
            best_model_info = {
                'name': self.best_model['name'],
                'metrics': self.best_model['metrics'],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output_path / "best_model_info.json", 'w') as f:
                import json
                json.dump(best_model_info, f, indent=2)
        
        # Save model performances
        if self.model_performances:
            with open(output_path / "model_performances.json", 'w') as f:
                import json
                json.dump(self.model_performances, f, indent=2)
        
        # Save feature importance
        if self.feature_importance:
            for model_name, importance in self.feature_importance.items():
                importance_file = output_path / f"{model_name}_feature_importance.csv"
                importance.to_csv(importance_file)
        
        logger.info("Model saving completed")
    
    def load_models(self, model_dir: str):
        """Load previously trained models."""
        model_path = Path(model_dir)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        logger.info(f"Loading models from {model_path}")
        
        # Load models
        self.models = {}
        for model_file in model_path.glob("*_model.pkl"):
            model_name = model_file.stem.replace('_model', '')
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} model")
        
        # Load performances if available
        perf_file = model_path / "model_performances.json"
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                import json
                self.model_performances = json.load(f)
        
        # Load best model info if available
        best_info_file = model_path / "best_model_info.json"
        if best_info_file.exists():
            with open(best_info_file, 'r') as f:
                import json
                best_info = json.load(f)
                
            if best_info['name'] in self.models:
                self.best_model = {
                    'name': best_info['name'],
                    'model': self.models[best_info['name']],
                    'metrics': best_info['metrics']
                }
    
    def train_and_evaluate_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                                   X_val: pd.DataFrame, y_val: pd.Series,
                                   X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Complete training and evaluation pipeline."""
        logger.info("Starting complete training and evaluation pipeline...")
        
        # Initialize models
        self.initialize_models()
        
        # Hyperparameter tuning (on train+validation)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        
        if self.config['hyperparameter_tuning']:
            self.hyperparameter_tuning(X_train_val, y_train_val)
        
        # Train models
        self.train_models(X_train_val, y_train_val)
        
        # Evaluate on test set
        test_results = self.evaluate_models(X_test, y_test)
        
        # Cross-validation
        cv_results = self.cross_validate_models(X_train_val, y_train_val)
        
        # Feature importance
        feature_importance = self.calculate_feature_importance(X_train_val, y_train_val)
        
        # Save models
        self.save_models()
        
        results = {
            'test_results': test_results,
            'cv_results': cv_results,
            'feature_importance': feature_importance,
            'best_model': self.best_model
        }
        
        logger.info("Training and evaluation pipeline completed")
        
        return results


def main():
    """Test model training."""
    from ..data_processing.preprocess import CreditDataProcessor
    from ..feature_engineering.features import CreditFeatureEngineer
    
    # Load and process data
    processor = CreditDataProcessor()
    data_splits = processor.process_all()
    
    # Feature engineering
    engineer = CreditFeatureEngineer()
    
    # Get training data
    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['validation']
    X_test, y_test = data_splits['test']
    
    # Initialize trainer
    trainer = CreditRiskModelTrainer()
    
    # Run complete pipeline
    results = trainer.train_and_evaluate_pipeline(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    print("\n=== Model Training Results ===")
    print(f"Best model: {results['best_model']['name']}")
    print(f"Best AUC: {results['best_model']['metrics']['auc']:.4f}")
    
    print("\nModel Comparison:")
    for model_name, metrics in results['test_results'].items():
        print(f"  {model_name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")


if __name__ == "__main__":
    main()