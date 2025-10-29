"""
Training pipeline for collaborative filtering, content-based, and hybrid recommendation models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import cross_validate
from surprise import accuracy
import joblib

from ..utils.config import get_config
from ..utils.logging import setup_logging, log_execution_time

logger = setup_logging(__name__)


class RecommenderTrainer:
    """Main trainer for recommendation models."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.metrics = {}
        self.data_splits = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment_name)
    
    @log_execution_time("data_loading")
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load training data splits."""
        data_dir = Path(self.config.data_dir) / "processed"
        
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            file_path = data_dir / f"{split_name}.csv"
            if file_path.exists():
                splits[split_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {split_name}: {len(splits[split_name])} interactions")
            else:
                logger.warning(f"Split file not found: {file_path}")
        
        # Load additional data
        for data_name in ['users', 'items']:
            file_path = data_dir / f"{data_name}.csv"
            if file_path.exists():
                splits[data_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {data_name}: {len(splits[data_name])} records")
        
        self.data_splits = splits
        return splits
    
    @log_execution_time("collaborative_training")
    def train_collaborative_filtering(self, algorithm: str = "svd") -> Dict:
        """Train collaborative filtering model using Surprise library."""
        if 'train' not in self.data_splits:
            raise ValueError("Training data not loaded")
        
        train_df = self.data_splits['train']
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 5))
        
        # Filter only interactions with explicit ratings
        rating_interactions = train_df[train_df['rating'] > 0].copy()
        
        if len(rating_interactions) == 0:
            logger.warning("No explicit ratings found, using interaction_type as implicit rating")
            # Convert interaction types to ratings
            interaction_to_rating = {
                'view': 1,
                'cart': 3,
                'purchase': 5,
                'rating': train_df['rating'].mean()
            }
            train_df['rating'] = train_df['interaction_type'].map(interaction_to_rating).fillna(3)
            rating_interactions = train_df.copy()
        
        data = Dataset.load_from_df(
            rating_interactions[['user_id', 'item_id', 'rating']], 
            reader
        )
        
        # Train model
        if algorithm.lower() == "svd":
            model = SVD(
                n_factors=self.config.n_factors,
                n_epochs=self.config.n_epochs,
                lr_all=self.config.learning_rate,
                reg_all=self.config.reg_all
            )
        elif algorithm.lower() == "nmf":
            model = NMF(
                n_factors=self.config.n_factors,
                n_epochs=self.config.n_epochs,
                reg_pu=self.config.reg_all,
                reg_qi=self.config.reg_all
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Cross-validation
        cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)
        
        # Train on full dataset
        trainset = data.build_full_trainset()
        model.fit(trainset)
        
        # Evaluate on validation set if available
        val_metrics = {}
        if 'validation' in self.data_splits:
            val_df = self.data_splits['validation']
            val_df = val_df[val_df['rating'] > 0]
            
            if len(val_df) > 0:
                predictions = []
                for _, row in val_df.iterrows():
                    pred = model.predict(row['user_id'], row['item_id'])
                    predictions.append(pred.est)
                
                val_metrics = {
                    'val_rmse': np.sqrt(mean_squared_error(val_df['rating'], predictions)),
                    'val_mae': mean_absolute_error(val_df['rating'], predictions)
                }
        
        # Store model and metrics
        self.models['collaborative'] = model
        self.metrics['collaborative'] = {
            'cv_rmse_mean': cv_results['test_rmse'].mean(),
            'cv_rmse_std': cv_results['test_rmse'].std(),
            'cv_mae_mean': cv_results['test_mae'].mean(),
            'cv_mae_std': cv_results['test_mae'].std(),
            **val_metrics
        }
        
        # Save model
        model_dir = Path(self.config.model_dir) / "collaborative"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / f"{algorithm}_model.pkl", 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"Collaborative filtering ({algorithm}) training completed")
        logger.info(f"CV RMSE: {self.metrics['collaborative']['cv_rmse_mean']:.4f} ± {self.metrics['collaborative']['cv_rmse_std']:.4f}")
        
        return self.metrics['collaborative']
    
    @log_execution_time("content_based_training")
    def train_content_based(self) -> Dict:
        """Train content-based recommendation model."""
        if 'items' not in self.data_splits:
            logger.warning("Item data not available, creating synthetic content features")
            self._create_synthetic_item_features()
        
        items_df = self.data_splits['items']
        
        # Create content features
        if 'category' in items_df.columns and 'brand' in items_df.columns:
            # Combine category and brand for content representation
            items_df['content'] = items_df['category'].astype(str) + " " + items_df['brand'].astype(str)
        else:
            # Create synthetic content
            items_df['content'] = "electronics gadget"
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        content_matrix = tfidf.fit_transform(items_df['content'])
        
        # Calculate item similarity matrix
        item_similarity = cosine_similarity(content_matrix)
        
        # Create item-to-index mapping
        item_to_idx = {item_id: idx for idx, item_id in enumerate(items_df['item_id'])}
        idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}
        
        content_model = {
            'tfidf_vectorizer': tfidf,
            'content_matrix': content_matrix,
            'item_similarity': item_similarity,
            'item_to_idx': item_to_idx,
            'idx_to_item': idx_to_item,
            'items_df': items_df
        }
        
        # Evaluate content-based model
        metrics = self._evaluate_content_based(content_model)
        
        self.models['content'] = content_model
        self.metrics['content'] = metrics
        
        # Save model
        model_dir = Path(self.config.model_dir) / "content_based"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        with open(model_dir / "tfidf_model.pkl", 'wb') as f:
            pickle.dump(content_model, f)
        
        logger.info("Content-based model training completed")
        logger.info(f"Content similarity coverage: {metrics['coverage']:.2%}")
        
        return metrics
    
    def _create_synthetic_item_features(self):
        """Create synthetic item features for demonstration."""
        if 'train' not in self.data_splits:
            return
        
        unique_items = self.data_splits['train']['item_id'].unique()
        
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
        
        np.random.seed(42)
        synthetic_items = pd.DataFrame({
            'item_id': unique_items,
            'category': np.random.choice(categories, len(unique_items)),
            'brand': np.random.choice(brands, len(unique_items)),
            'price': np.random.lognormal(3, 1, len(unique_items))
        })
        
        self.data_splits['items'] = synthetic_items
        logger.info(f"Created synthetic features for {len(synthetic_items)} items")
    
    def _evaluate_content_based(self, model: Dict) -> Dict:
        """Evaluate content-based model."""
        item_similarity = model['item_similarity']
        
        # Calculate metrics
        avg_similarity = item_similarity.mean()
        coverage = (item_similarity > 0.1).sum() / (item_similarity.shape[0] * item_similarity.shape[1])
        
        return {
            'avg_similarity': avg_similarity,
            'coverage': coverage,
            'total_items': item_similarity.shape[0]
        }
    
    @log_execution_time("hybrid_training")
    def train_hybrid_model(self) -> Dict:
        """Train hybrid recommendation model."""
        if 'collaborative' not in self.models or 'content' not in self.models:
            raise ValueError("Both collaborative and content models must be trained first")
        
        # Hybrid model combines predictions from both models
        # For demonstration, we'll use weighted combination
        weights = self.config.hybrid_weights
        
        # Evaluate hybrid performance on validation set
        metrics = {}
        if 'validation' in self.data_splits:
            val_df = self.data_splits['validation']
            val_df = val_df[val_df['rating'] > 0]
            
            if len(val_df) > 0:
                hybrid_predictions = []
                
                for _, row in val_df.iterrows():
                    # Get collaborative prediction
                    try:
                        cf_pred = self.models['collaborative'].predict(
                            row['user_id'], row['item_id']
                        ).est
                    except:
                        cf_pred = 3.0  # Default rating
                    
                    # Get content-based prediction (simplified)
                    cb_pred = self._get_content_prediction(row['item_id'], row['user_id'])
                    
                    # Get popularity prediction
                    pop_pred = 3.5  # Average popularity score
                    
                    # Combine predictions
                    hybrid_pred = (
                        cf_pred * weights['collaborative'] +
                        cb_pred * weights['content'] +
                        pop_pred * weights['popularity']
                    )
                    
                    hybrid_predictions.append(hybrid_pred)
                
                metrics = {
                    'val_rmse': np.sqrt(mean_squared_error(val_df['rating'], hybrid_predictions)),
                    'val_mae': mean_absolute_error(val_df['rating'], hybrid_predictions),
                    'weights': weights
                }
        
        self.metrics['hybrid'] = metrics
        
        # Save hybrid configuration
        model_dir = Path(self.config.model_dir) / "hybrid"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        hybrid_config = {
            'weights': weights,
            'metrics': metrics,
            'timestamp': datetime.utcnow()
        }
        
        with open(model_dir / "hybrid_config.pkl", 'wb') as f:
            pickle.dump(hybrid_config, f)
        
        logger.info("Hybrid model training completed")
        if metrics:
            logger.info(f"Hybrid RMSE: {metrics['val_rmse']:.4f}")
        
        return metrics
    
    def _get_content_prediction(self, item_id: int, user_id: int) -> float:
        """Get content-based prediction for an item."""
        content_model = self.models['content']
        
        if item_id not in content_model['item_to_idx']:
            return 3.0  # Default rating
        
        item_idx = content_model['item_to_idx'][item_id]
        
        # Get similar items
        similarities = content_model['item_similarity'][item_idx]
        
        # Return weighted average (simplified approach)
        return 3.0 + (similarities.mean() - 0.5) * 2
    
    @log_execution_time("model_training")
    def train_all_models(self, models: List[str] = None) -> Dict:
        """Train all specified models."""
        if models is None:
            models = ['collaborative', 'content', 'hybrid']
        
        # Load data first
        self.load_data()
        
        all_metrics = {}
        
        with mlflow.start_run(run_name=f"recommender_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log configuration
            mlflow.log_params(self.config.to_dict())
            
            if 'collaborative' in models:
                cf_metrics = self.train_collaborative_filtering(self.config.collaborative_model)
                all_metrics['collaborative'] = cf_metrics
                mlflow.log_metrics({f"cf_{k}": v for k, v in cf_metrics.items() if isinstance(v, (int, float))})
            
            if 'content' in models:
                cb_metrics = self.train_content_based()
                all_metrics['content'] = cb_metrics
                mlflow.log_metrics({f"cb_{k}": v for k, v in cb_metrics.items() if isinstance(v, (int, float))})
            
            if 'hybrid' in models and 'collaborative' in models and 'content' in models:
                hybrid_metrics = self.train_hybrid_model()
                all_metrics['hybrid'] = hybrid_metrics
                mlflow.log_metrics({f"hybrid_{k}": v for k, v in hybrid_metrics.items() if isinstance(v, (int, float))})
            
            # Log artifacts
            mlflow.log_artifacts(self.config.model_dir)
        
        self.metrics.update(all_metrics)
        
        # Save training summary
        summary_path = Path(self.config.model_dir) / "training_summary.pkl"
        with open(summary_path, 'wb') as f:
            pickle.dump({
                'metrics': all_metrics,
                'config': self.config.to_dict(),
                'timestamp': datetime.utcnow()
            }, f)
        
        logger.info("All model training completed")
        return all_metrics
    
    def evaluate_models(self) -> Dict:
        """Comprehensive evaluation of all trained models."""
        if 'test' not in self.data_splits:
            logger.warning("Test set not available for evaluation")
            return {}
        
        test_df = self.data_splits['test']
        test_df = test_df[test_df['rating'] > 0]
        
        if len(test_df) == 0:
            logger.warning("No explicit ratings in test set")
            return {}
        
        evaluation_results = {}
        
        for model_name in self.models.keys():
            if model_name == 'content':
                continue  # Skip content model for rating prediction evaluation
            
            try:
                predictions = []
                
                for _, row in test_df.iterrows():
                    if model_name == 'collaborative':
                        pred = self.models[model_name].predict(
                            row['user_id'], row['item_id']
                        ).est
                    else:  # hybrid
                        pred = self._get_hybrid_prediction(row['user_id'], row['item_id'])
                    
                    predictions.append(pred)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(test_df['rating'], predictions))
                mae = mean_absolute_error(test_df['rating'], predictions)
                
                evaluation_results[model_name] = {
                    'test_rmse': rmse,
                    'test_mae': mae,
                    'test_samples': len(test_df)
                }
                
                logger.info(f"{model_name.title()} Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return evaluation_results
    
    def _get_hybrid_prediction(self, user_id: int, item_id: int) -> float:
        """Get hybrid prediction combining all models."""
        weights = self.config.hybrid_weights
        
        # Collaborative prediction
        try:
            cf_pred = self.models['collaborative'].predict(user_id, item_id).est
        except:
            cf_pred = 3.0
        
        # Content prediction
        cb_pred = self._get_content_prediction(item_id, user_id)
        
        # Popularity prediction
        pop_pred = 3.5
        
        return (
            cf_pred * weights['collaborative'] +
            cb_pred * weights['content'] +
            pop_pred * weights['popularity']
        )


def main():
    """Main training function."""
    config = get_config()
    trainer = RecommenderTrainer(config)
    
    # Train all models
    metrics = trainer.train_all_models()
    
    # Evaluate models
    evaluation = trainer.evaluate_models()
    
    print("\n=== Training Summary ===")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name.title()} Model:")
        for metric_name, value in model_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")
    
    if evaluation:
        print("\n=== Test Evaluation ===")
        for model_name, eval_metrics in evaluation.items():
            print(f"\n{model_name.title()} Model:")
            for metric_name, value in eval_metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()