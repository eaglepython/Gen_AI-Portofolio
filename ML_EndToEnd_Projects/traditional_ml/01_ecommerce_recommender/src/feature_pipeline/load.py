"""
Data loading and splitting module for e-commerce recommender system.
Handles time-aware data splitting and interaction preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import logging
from datetime import datetime, timedelta
import requests
import zipfile
import os

from ..utils.config import Config
from ..utils.logging import setup_logging

logger = setup_logging(__name__)


class DataLoader:
    """Handles data loading and time-aware splitting for recommendation system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_movielens_data(self, dataset_size: str = "25m") -> Path:
        """Download MovieLens dataset for demonstration purposes."""
        dataset_url = {
            "25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
            "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
            "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        }
        
        if dataset_size not in dataset_url:
            raise ValueError(f"Dataset size {dataset_size} not supported")
        
        zip_path = self.raw_dir / f"ml-{dataset_size}.zip"
        extract_path = self.raw_dir / f"ml-{dataset_size}"
        
        if extract_path.exists():
            logger.info(f"Dataset ml-{dataset_size} already exists")
            return extract_path
        
        logger.info(f"Downloading MovieLens {dataset_size} dataset...")
        
        response = requests.get(dataset_url[dataset_size], stream=True)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        # Remove zip file to save space
        zip_path.unlink()
        
        return extract_path
    
    def load_movielens_data(self, dataset_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load MovieLens data into pandas DataFrames."""
        logger.info("Loading MovieLens dataset...")
        
        # Load ratings
        ratings_file = dataset_path / "ratings.csv"
        if not ratings_file.exists():
            # Try alternative file structure
            ratings_file = dataset_path / "ratings.dat"
            if ratings_file.exists():
                ratings = pd.read_csv(
                    ratings_file,
                    sep="::",
                    names=["user_id", "item_id", "rating", "timestamp"],
                    engine="python"
                )
            else:
                raise FileNotFoundError(f"Ratings file not found in {dataset_path}")
        else:
            ratings = pd.read_csv(ratings_file)
            ratings.rename(columns={"userId": "user_id", "movieId": "item_id"}, inplace=True)
        
        # Load movies/items
        movies_file = dataset_path / "movies.csv"
        if not movies_file.exists():
            movies_file = dataset_path / "movies.dat"
            if movies_file.exists():
                movies = pd.read_csv(
                    movies_file,
                    sep="::",
                    names=["item_id", "title", "genres"],
                    engine="python",
                    encoding="latin1"
                )
            else:
                raise FileNotFoundError(f"Movies file not found in {dataset_path}")
        else:
            movies = pd.read_csv(movies_file)
            movies.rename(columns={"movieId": "item_id"}, inplace=True)
        
        # Load tags (if available)
        tags_file = dataset_path / "tags.csv"
        if tags_file.exists():
            tags = pd.read_csv(tags_file)
            tags.rename(columns={"userId": "user_id", "movieId": "item_id"}, inplace=True)
        else:
            tags = pd.DataFrame(columns=["user_id", "item_id", "tag", "timestamp"])
        
        logger.info(f"Loaded {len(ratings)} ratings, {len(movies)} movies, {len(tags)} tags")
        
        return ratings, movies, tags
    
    def generate_synthetic_ecommerce_data(self, 
                                        n_users: int = 100000,
                                        n_items: int = 50000,
                                        n_interactions: int = 5000000) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate synthetic e-commerce data for demonstration."""
        logger.info(f"Generating synthetic e-commerce data: {n_users} users, {n_items} items, {n_interactions} interactions")
        
        np.random.seed(42)
        
        # Generate users
        users = pd.DataFrame({
            'user_id': range(1, n_users + 1),
            'age': np.random.randint(18, 80, n_users),
            'gender': np.random.choice(['M', 'F', 'O'], n_users, p=[0.45, 0.45, 0.1]),
            'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP'], n_users, p=[0.3, 0.15, 0.1, 0.05, 0.1, 0.1, 0.2]),
            'registration_date': pd.date_range(start='2020-01-01', end='2024-12-31', periods=n_users)
        })
        
        # Generate items
        categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty', 'Toys', 'Food']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF', 'BrandG', 'BrandH']
        
        items = pd.DataFrame({
            'item_id': range(1, n_items + 1),
            'category': np.random.choice(categories, n_items),
            'brand': np.random.choice(brands, n_items),
            'price': np.random.lognormal(mean=3, sigma=1, size=n_items).round(2),
            'launch_date': pd.date_range(start='2019-01-01', end='2024-12-31', periods=n_items)
        })
        
        # Generate interactions with temporal patterns
        interaction_types = ['view', 'cart', 'purchase']
        interaction_weights = [0.7, 0.2, 0.1]
        
        # Create power-law distributions for user activity and item popularity
        user_activity = np.random.pareto(1.5, n_users) + 1
        item_popularity = np.random.pareto(1.2, n_items) + 1
        
        interactions = []
        
        for _ in range(n_interactions):
            # Sample user and item based on activity/popularity
            user_id = np.random.choice(users['user_id'], p=user_activity/user_activity.sum())
            item_id = np.random.choice(items['item_id'], p=item_popularity/item_popularity.sum())
            
            # Generate timestamp with realistic patterns
            base_date = datetime(2022, 1, 1)
            days_offset = np.random.exponential(200)  # Recent bias
            timestamp = base_date + timedelta(days=min(days_offset, 1095))  # Cap at 3 years
            
            # Sample interaction type
            interaction_type = np.random.choice(interaction_types, p=interaction_weights)
            
            # Generate rating (for purchases mostly)
            if interaction_type == 'purchase':
                rating = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
            elif interaction_type == 'cart':
                rating = np.random.choice([0, 4, 5], p=[0.7, 0.2, 0.1])
            else:  # view
                rating = 0
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        logger.info("Synthetic data generation completed")
        return interactions_df, users, items
    
    def create_time_splits(self, df: pd.DataFrame, 
                          train_end: str = "2024-01-01",
                          val_end: str = "2024-07-01") -> Dict[str, pd.DataFrame]:
        """Create time-aware train/validation/test splits."""
        logger.info("Creating time-aware data splits...")
        
        # Convert timestamp to datetime if it's not already
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype == 'int64':
                # Unix timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create splits
        train_mask = df['timestamp'] < train_end
        val_mask = (df['timestamp'] >= train_end) & (df['timestamp'] < val_end)
        test_mask = df['timestamp'] >= val_end
        
        splits = {
            'train': df[train_mask].copy(),
            'validation': df[val_mask].copy(),
            'test': df[test_mask].copy()
        }
        
        # Log split statistics
        for split_name, split_df in splits.items():
            logger.info(f"{split_name}: {len(split_df)} interactions, "
                       f"{split_df['user_id'].nunique()} users, "
                       f"{split_df['item_id'].nunique()} items")
            if len(split_df) > 0:
                logger.info(f"  Date range: {split_df['timestamp'].min()} to {split_df['timestamp'].max()}")
        
        return splits
    
    def save_splits(self, splits: Dict[str, pd.DataFrame], 
                   users: Optional[pd.DataFrame] = None,
                   items: Optional[pd.DataFrame] = None):
        """Save data splits to disk."""
        logger.info("Saving data splits...")
        
        for split_name, split_df in splits.items():
            output_path = self.processed_dir / f"{split_name}.csv"
            split_df.to_csv(output_path, index=False)
            logger.info(f"Saved {split_name} to {output_path}")
        
        if users is not None:
            users_path = self.processed_dir / "users.csv"
            users.to_csv(users_path, index=False)
            logger.info(f"Saved users to {users_path}")
        
        if items is not None:
            items_path = self.processed_dir / "items.csv"
            items.to_csv(items_path, index=False)
            logger.info(f"Saved items to {items_path}")
    
    def load_splits(self) -> Dict[str, pd.DataFrame]:
        """Load existing data splits from disk."""
        logger.info("Loading data splits from disk...")
        
        splits = {}
        for split_name in ['train', 'validation', 'test']:
            file_path = self.processed_dir / f"{split_name}.csv"
            if file_path.exists():
                splits[split_name] = pd.read_csv(file_path)
                logger.info(f"Loaded {split_name}: {len(splits[split_name])} rows")
            else:
                logger.warning(f"Split file {file_path} not found")
        
        return splits


def main():
    """Main function to run data loading and splitting."""
    config = Config()
    loader = DataLoader(config)
    
    # Choice of data source
    data_source = config.get('data_source', 'synthetic')
    
    if data_source == 'movielens':
        # Download and load MovieLens data
        dataset_path = loader.download_movielens_data("25m")
        interactions, items, tags = loader.load_movielens_data(dataset_path)
        users = None  # MovieLens doesn't have user demographics
        
        # Convert to e-commerce format
        interactions = interactions.rename(columns={'rating': 'rating'})
        interactions['interaction_type'] = 'rating'
        
    else:  # synthetic
        # Generate synthetic e-commerce data
        interactions, users, items = loader.generate_synthetic_ecommerce_data()
    
    # Create time-based splits
    splits = loader.create_time_splits(interactions)
    
    # Save all data
    loader.save_splits(splits, users, items)
    
    logger.info("Data loading and splitting completed successfully")


if __name__ == "__main__":
    main()