"""
Data preprocessing module for e-commerce recommendation system.
Handles data loading, cleaning, and preparation for modeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import requests
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import get_config
from ..utils.logging import setup_logging, log_execution_time

logger = setup_logging(__name__)


class DataPreprocessor:
    """Main data preprocessing class for recommendation system."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.label_encoders = {}
        self.data_stats = {}
        
    @log_execution_time("data_loading")
    def load_data(self, data_source: str = None) -> Dict[str, pd.DataFrame]:
        """Load data from various sources."""
        source = data_source or self.config.data_source
        
        if source == "synthetic":
            return self._generate_synthetic_data()
        elif source == "movielens":
            return self._load_movielens_data()
        elif source == "custom":
            return self._load_custom_data()
        else:
            raise ValueError(f"Unknown data source: {source}")
    
    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic e-commerce data for demonstration."""
        logger.info("Generating synthetic e-commerce data...")
        
        np.random.seed(42)
        
        # Parameters
        n_users = 5000
        n_items = 1000
        n_interactions = 50000
        
        # Generate users
        users_data = []
        for user_id in range(1, n_users + 1):
            age = np.random.randint(18, 65)
            gender = np.random.choice(['M', 'F'], p=[0.5, 0.5])
            location = np.random.choice(['US', 'UK', 'CA', 'AU', 'DE'], p=[0.4, 0.2, 0.15, 0.15, 0.1])
            registration_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
            
            users_data.append({
                'user_id': user_id,
                'age': age,
                'gender': gender,
                'location': location,
                'registration_date': registration_date
            })
        
        users_df = pd.DataFrame(users_data)
        
        # Generate items
        categories = ['Electronics', 'Clothing', 'Books', 'Home & Garden', 'Sports', 'Beauty', 'Toys', 'Food']
        brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE', 'BrandF', 'BrandG', 'BrandH']
        
        items_data = []
        for item_id in range(1, n_items + 1):
            category = np.random.choice(categories)
            brand = np.random.choice(brands)
            price = np.random.lognormal(3, 1)  # Log-normal distribution for prices
            launch_date = datetime.now() - timedelta(days=np.random.randint(1, 1000))
            
            items_data.append({
                'item_id': item_id,
                'category': category,
                'brand': brand,
                'price': round(price, 2),
                'launch_date': launch_date
            })
        
        items_df = pd.DataFrame(items_data)
        
        # Generate interactions
        interactions_data = []
        interaction_types = ['view', 'cart', 'purchase', 'rating']
        
        for _ in range(n_interactions):
            user_id = np.random.randint(1, n_users + 1)
            item_id = np.random.randint(1, n_items + 1)
            
            # Make interactions realistic - some users and items are more popular
            # Apply power law distribution
            if np.random.random() < 0.3:  # 30% chance for popular users/items
                user_id = np.random.choice(range(1, min(500, n_users)), 1)[0]  # Top 10% users
                item_id = np.random.choice(range(1, min(100, n_items)), 1)[0]  # Top 10% items
            
            interaction_type = np.random.choice(interaction_types, p=[0.6, 0.2, 0.15, 0.05])
            
            # Generate rating for purchase interactions
            rating = None
            if interaction_type == 'rating' or (interaction_type == 'purchase' and np.random.random() < 0.3):
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
            
            timestamp = datetime.now() - timedelta(days=np.random.randint(0, 90))
            
            interactions_data.append({
                'user_id': user_id,
                'item_id': item_id,
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': timestamp
            })
        
        interactions_df = pd.DataFrame(interactions_data)
        
        logger.info(f"Generated synthetic data:")
        logger.info(f"  Users: {len(users_df)}")
        logger.info(f"  Items: {len(items_df)}")
        logger.info(f"  Interactions: {len(interactions_df)}")
        
        return {
            'users': users_df,
            'items': items_df,
            'interactions': interactions_df
        }
    
    def _load_movielens_data(self) -> Dict[str, pd.DataFrame]:
        """Load MovieLens dataset."""
        logger.info("Loading MovieLens dataset...")
        
        dataset_size = self.config.dataset_size
        
        # Download data if not exists
        data_dir = Path(self.config.data_dir) / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_size == "100k":
            url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
            zip_name = "ml-100k.zip"
            folder_name = "ml-100k"
        elif dataset_size == "1m":
            url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
            zip_name = "ml-1m.zip"
            folder_name = "ml-1m"
        elif dataset_size == "25m":
            url = "http://files.grouplens.org/datasets/movielens/ml-25m.zip"
            zip_name = "ml-25m.zip"
            folder_name = "ml-25m"
        else:
            raise ValueError(f"Unsupported dataset size: {dataset_size}")
        
        zip_path = data_dir / zip_name
        extract_path = data_dir / folder_name
        
        # Download if not exists
        if not extract_path.exists():
            if not zip_path.exists():
                logger.info(f"Downloading {zip_name}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract
            logger.info(f"Extracting {zip_name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        
        # Load data based on dataset size
        if dataset_size == "100k":
            # Load ratings
            ratings_df = pd.read_csv(
                extract_path / "u.data",
                sep='\t',
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            
            # Load users
            users_df = pd.read_csv(
                extract_path / "u.user",
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                engine='python'
            )
            
            # Load items
            items_df = pd.read_csv(
                extract_path / "u.item",
                sep='|',
                names=['item_id', 'title', 'release_date', 'video_release_date', 'url'] + 
                      [f'genre_{i}' for i in range(19)],
                encoding='latin1',
                engine='python'
            )
            
        else:  # 1m or 25m
            # Load ratings
            if dataset_size == "1m":
                ratings_df = pd.read_csv(
                    extract_path / "ratings.dat",
                    sep='::',
                    names=['user_id', 'item_id', 'rating', 'timestamp'],
                    engine='python'
                )
                
                # Load users
                users_df = pd.read_csv(
                    extract_path / "users.dat",
                    sep='::',
                    names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                    engine='python'
                )
                
                # Load movies
                items_df = pd.read_csv(
                    extract_path / "movies.dat",
                    sep='::',
                    names=['item_id', 'title', 'genres'],
                    encoding='latin1',
                    engine='python'
                )
            
            else:  # 25m
                ratings_df = pd.read_csv(extract_path / "ratings.csv")
                ratings_df = ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'})
                
                items_df = pd.read_csv(extract_path / "movies.csv")
                items_df = items_df.rename(columns={'movieId': 'item_id'})
                
                # Create synthetic user data for 25m dataset
                unique_users = ratings_df['user_id'].unique()
                users_data = []
                for user_id in unique_users:
                    users_data.append({
                        'user_id': user_id,
                        'age': np.random.randint(18, 65),
                        'gender': np.random.choice(['M', 'F']),
                        'occupation': f'occupation_{np.random.randint(0, 20)}',
                        'zip_code': f'{np.random.randint(10000, 99999)}'
                    })
                users_df = pd.DataFrame(users_data)
        
        # Convert timestamps
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        
        # Add interaction type (all are ratings in MovieLens)
        ratings_df['interaction_type'] = 'rating'
        
        # Rename to match our schema
        interactions_df = ratings_df.rename(columns={'movieId': 'item_id', 'userId': 'user_id'})
        
        logger.info(f"Loaded MovieLens {dataset_size} dataset:")
        logger.info(f"  Users: {len(users_df)}")
        logger.info(f"  Items: {len(items_df)}")
        logger.info(f"  Interactions: {len(interactions_df)}")
        
        return {
            'users': users_df,
            'items': items_df,
            'interactions': interactions_df
        }
    
    def _load_custom_data(self) -> Dict[str, pd.DataFrame]:
        """Load custom data from files."""
        logger.info("Loading custom data...")
        
        data_dir = Path(self.config.data_dir) / "raw"
        
        # Expected files
        files_to_load = {
            'users': 'users.csv',
            'items': 'items.csv',
            'interactions': 'interactions.csv'
        }
        
        data = {}
        for data_type, filename in files_to_load.items():
            file_path = data_dir / filename
            if file_path.exists():
                data[data_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {data_type}: {len(data[data_type])} records")
            else:
                logger.warning(f"File not found: {file_path}")
        
        return data
    
    @log_execution_time("data_cleaning")
    def clean_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean and validate the data."""
        logger.info("Cleaning data...")
        
        cleaned_data = {}
        
        for data_type, df in data.items():
            logger.info(f"Cleaning {data_type} data...")
            
            # Remove duplicates
            initial_size = len(df)
            if data_type == 'interactions':
                # For interactions, consider duplicates as same user, item, and interaction_type
                df = df.drop_duplicates(subset=['user_id', 'item_id', 'interaction_type', 'timestamp'])
            else:
                df = df.drop_duplicates()
            
            duplicate_removed = initial_size - len(df)
            if duplicate_removed > 0:
                logger.info(f"  Removed {duplicate_removed} duplicates from {data_type}")
            
            # Handle missing values
            missing_before = df.isnull().sum().sum()
            
            if data_type == 'users':
                df = self._clean_users_data(df)
            elif data_type == 'items':
                df = self._clean_items_data(df)
            elif data_type == 'interactions':
                df = self._clean_interactions_data(df)
            
            missing_after = df.isnull().sum().sum()
            if missing_before > 0:
                logger.info(f"  Handled {missing_before - missing_after} missing values in {data_type}")
            
            cleaned_data[data_type] = df
        
        # Filter interactions based on user and item activity
        if 'interactions' in cleaned_data:
            cleaned_data['interactions'] = self._filter_interactions(cleaned_data['interactions'])
        
        return cleaned_data
    
    def _clean_users_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean users data."""
        # Ensure user_id is integer
        df['user_id'] = df['user_id'].astype(int)
        
        # Handle age
        if 'age' in df.columns:
            df['age'] = df['age'].fillna(df['age'].median())
            df['age'] = df['age'].clip(18, 100)  # Reasonable age bounds
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'user_id':
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def _clean_items_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean items data."""
        # Ensure item_id is integer
        df['item_id'] = df['item_id'].astype(int)
        
        # Handle price
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['price'] = df['price'].fillna(df['price'].median())
            df['price'] = df['price'].clip(0.01, None)  # Positive prices
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'item_id':
                df[col] = df[col].fillna('unknown')
        
        return df
    
    def _clean_interactions_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean interactions data."""
        # Ensure user_id and item_id are integers
        df['user_id'] = df['user_id'].astype(int)
        df['item_id'] = df['item_id'].astype(int)
        
        # Handle ratings
        if 'rating' in df.columns:
            # Only keep valid ratings (1-5) or null values
            df.loc[~df['rating'].between(1, 5), 'rating'] = np.nan
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Remove interactions with invalid timestamps
            df = df.dropna(subset=['timestamp'])
        
        # Handle interaction types
        if 'interaction_type' in df.columns:
            valid_types = ['view', 'cart', 'purchase', 'rating', 'like', 'share']
            df = df[df['interaction_type'].isin(valid_types)]
        
        return df
    
    def _filter_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter interactions based on minimum activity thresholds."""
        logger.info("Filtering interactions based on activity thresholds...")
        
        initial_size = len(df)
        
        # Filter users with minimum interactions
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= self.config.min_interactions_per_user].index
        df = df[df['user_id'].isin(active_users)]
        
        # Filter items with minimum interactions
        item_counts = df['item_id'].value_counts()
        active_items = item_counts[item_counts >= self.config.min_interactions_per_item].index
        df = df[df['item_id'].isin(active_items)]
        
        final_size = len(df)
        removed = initial_size - final_size
        
        logger.info(f"Filtered {removed} interactions ({removed/initial_size*100:.1f}%)")
        logger.info(f"Remaining: {len(df['user_id'].unique())} users, {len(df['item_id'].unique())} items")
        
        return df
    
    @log_execution_time("data_splitting")
    def split_data(self, interactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split interactions data into train/validation/test sets."""
        logger.info("Splitting data...")
        
        # Sort by timestamp for temporal split
        interactions_df = interactions_df.sort_values('timestamp')
        
        # Calculate split sizes
        total_size = len(interactions_df)
        train_size = int(total_size * self.config.train_ratio)
        val_size = int(total_size * self.config.validation_ratio)
        
        # Split data
        train_df = interactions_df.iloc[:train_size].copy()
        val_df = interactions_df.iloc[train_size:train_size + val_size].copy()
        test_df = interactions_df.iloc[train_size + val_size:].copy()
        
        logger.info(f"Data split:")
        logger.info(f"  Train: {len(train_df)} ({len(train_df)/total_size*100:.1f}%)")
        logger.info(f"  Validation: {len(val_df)} ({len(val_df)/total_size*100:.1f}%)")
        logger.info(f"  Test: {len(test_df)} ({len(test_df)/total_size*100:.1f}%)")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    @log_execution_time("data_saving")
    def save_processed_data(self, data: Dict[str, pd.DataFrame]):
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        processed_dir = Path(self.config.data_dir) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        for data_type, df in data.items():
            file_path = processed_dir / f"{data_type}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {data_type} to {file_path}")
        
        # Save data statistics
        stats_path = processed_dir / "data_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.data_stats, f, indent=2, default=str)
        
        logger.info(f"Saved data statistics to {stats_path}")
    
    def calculate_data_statistics(self, data: Dict[str, pd.DataFrame]):
        """Calculate and store data statistics."""
        logger.info("Calculating data statistics...")
        
        stats = {}
        
        if 'interactions' in data:
            interactions_df = data['interactions']
            
            stats['interactions'] = {
                'total_interactions': len(interactions_df),
                'unique_users': interactions_df['user_id'].nunique(),
                'unique_items': interactions_df['item_id'].nunique(),
                'sparsity': 1 - (len(interactions_df) / (interactions_df['user_id'].nunique() * interactions_df['item_id'].nunique())),
                'avg_interactions_per_user': len(interactions_df) / interactions_df['user_id'].nunique(),
                'avg_interactions_per_item': len(interactions_df) / interactions_df['item_id'].nunique(),
                'date_range': {
                    'start': interactions_df['timestamp'].min(),
                    'end': interactions_df['timestamp'].max()
                }
            }
            
            if 'rating' in interactions_df.columns:
                ratings = interactions_df['rating'].dropna()
                stats['ratings'] = {
                    'total_ratings': len(ratings),
                    'avg_rating': ratings.mean(),
                    'rating_distribution': ratings.value_counts().to_dict()
                }
            
            if 'interaction_type' in interactions_df.columns:
                stats['interaction_types'] = interactions_df['interaction_type'].value_counts().to_dict()
        
        if 'users' in data:
            users_df = data['users']
            stats['users'] = {
                'total_users': len(users_df),
                'columns': list(users_df.columns)
            }
            
            if 'age' in users_df.columns:
                stats['users']['age_stats'] = {
                    'mean': users_df['age'].mean(),
                    'min': users_df['age'].min(),
                    'max': users_df['age'].max()
                }
        
        if 'items' in data:
            items_df = data['items']
            stats['items'] = {
                'total_items': len(items_df),
                'columns': list(items_df.columns)
            }
            
            if 'category' in items_df.columns:
                stats['items']['categories'] = items_df['category'].value_counts().to_dict()
        
        self.data_stats = stats
        
        # Log key statistics
        logger.info("Data Statistics:")
        if 'interactions' in stats:
            logger.info(f"  Total interactions: {stats['interactions']['total_interactions']:,}")
            logger.info(f"  Unique users: {stats['interactions']['unique_users']:,}")
            logger.info(f"  Unique items: {stats['interactions']['unique_items']:,}")
            logger.info(f"  Sparsity: {stats['interactions']['sparsity']:.4f}")
    
    def process_all(self) -> Dict[str, pd.DataFrame]:
        """Run the complete data preprocessing pipeline."""
        logger.info("Starting complete data preprocessing pipeline...")
        
        # Load data
        raw_data = self.load_data()
        
        # Clean data
        cleaned_data = self.clean_data(raw_data)
        
        # Calculate statistics
        self.calculate_data_statistics(cleaned_data)
        
        # Split interactions data
        if 'interactions' in cleaned_data:
            split_data = self.split_data(cleaned_data['interactions'])
            
            # Add other data types
            for data_type, df in cleaned_data.items():
                if data_type != 'interactions':
                    split_data[data_type] = df
            
            cleaned_data = split_data
        
        # Save processed data
        self.save_processed_data(cleaned_data)
        
        logger.info("Data preprocessing pipeline completed successfully")
        
        return cleaned_data


def main():
    """Main function to run data preprocessing."""
    config = get_config()
    preprocessor = DataPreprocessor(config)
    
    # Process all data
    processed_data = preprocessor.process_all()
    
    print("\n=== Data Preprocessing Summary ===")
    for data_type, df in processed_data.items():
        print(f"{data_type}: {len(df)} records")
    
    print(f"\nProcessed data saved to: {config.data_dir}/processed/")


if __name__ == "__main__":
    main()