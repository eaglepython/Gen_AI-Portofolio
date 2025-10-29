"""
Feature engineering for recommendation systems.
Handles user features, item features, and interaction features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import get_config
from ..utils.logging import setup_logging, log_execution_time

logger = setup_logging(__name__)


class FeatureEngineer:
    """Feature engineering for recommendation systems."""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    @log_execution_time("user_features")
    def create_user_features(self, interactions_df: pd.DataFrame, users_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create comprehensive user features."""
        logger.info("Creating user features...")
        
        # Basic user statistics from interactions
        user_stats = interactions_df.groupby('user_id').agg({
            'item_id': 'count',  # Total interactions
            'rating': ['mean', 'std', 'count'],  # Rating statistics
            'timestamp': ['min', 'max']  # Activity period
        }).round(3)
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in user_stats.columns]
        user_stats = user_stats.rename(columns={
            'item_id_count': 'total_interactions',
            'rating_mean': 'avg_rating',
            'rating_std': 'rating_variance',
            'rating_count': 'rating_count',
            'timestamp_min': 'first_interaction',
            'timestamp_max': 'last_interaction'
        })
        
        # Calculate additional features
        user_stats['rating_variance'] = user_stats['rating_variance'].fillna(0)
        
        # Activity recency (days since last interaction)
        if 'last_interaction' in user_stats.columns:
            current_time = datetime.utcnow()
            user_stats['days_since_last_interaction'] = (
                current_time - pd.to_datetime(user_stats['last_interaction'])
            ).dt.days
        else:
            user_stats['days_since_last_interaction'] = 0
        
        # Activity span (days between first and last interaction)
        if 'first_interaction' in user_stats.columns and 'last_interaction' in user_stats.columns:
            user_stats['activity_span_days'] = (
                pd.to_datetime(user_stats['last_interaction']) - 
                pd.to_datetime(user_stats['first_interaction'])
            ).dt.days
        else:
            user_stats['activity_span_days'] = 0
        
        # Interaction frequency
        user_stats['interaction_frequency'] = np.where(
            user_stats['activity_span_days'] > 0,
            user_stats['total_interactions'] / user_stats['activity_span_days'],
            user_stats['total_interactions']
        )
        
        # Interaction types analysis
        if 'interaction_type' in interactions_df.columns:
            interaction_types = interactions_df.groupby(['user_id', 'interaction_type']).size().unstack(fill_value=0)
            interaction_types.columns = [f'interactions_{col}' for col in interaction_types.columns]
            user_stats = user_stats.join(interaction_types, how='left').fillna(0)
            
            # Conversion rates
            if 'interactions_view' in user_stats.columns and 'interactions_purchase' in user_stats.columns:
                user_stats['view_to_purchase_rate'] = np.where(
                    user_stats['interactions_view'] > 0,
                    user_stats['interactions_purchase'] / user_stats['interactions_view'],
                    0
                )
        
        # Category preferences
        if 'category' in interactions_df.columns:
            user_categories = interactions_df.groupby(['user_id', 'category']).size().unstack(fill_value=0)
            user_categories = user_categories.div(user_categories.sum(axis=1), axis=0).fillna(0)
            user_categories.columns = [f'category_pref_{col}' for col in user_categories.columns]
            
            # Top category and diversity
            user_stats['top_category_ratio'] = user_categories.max(axis=1)
            user_stats['category_diversity'] = (user_categories > 0).sum(axis=1)
            
            user_stats = user_stats.join(user_categories, how='left').fillna(0)
        
        # User behavioral segments
        user_stats['user_segment'] = self._create_user_segments(user_stats)
        
        # Merge with provided user data
        if users_df is not None:
            user_stats = user_stats.join(users_df.set_index('user_id'), how='left')
            
            # Encode categorical features
            categorical_cols = users_df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'user_id' and col in user_stats.columns:
                    le = LabelEncoder()
                    user_stats[f'{col}_encoded'] = le.fit_transform(user_stats[col].fillna('unknown'))
                    self.encoders[f'user_{col}'] = le
        
        # Scale numerical features
        numerical_cols = user_stats.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        user_stats_scaled = user_stats.copy()
        user_stats_scaled[numerical_cols] = scaler.fit_transform(user_stats[numerical_cols])
        self.scalers['user_features'] = scaler
        
        # Store feature statistics
        self.feature_stats['user_features'] = {
            'n_users': len(user_stats),
            'n_features': len(user_stats.columns),
            'feature_names': list(user_stats.columns)
        }
        
        user_stats.reset_index(inplace=True)
        logger.info(f"Created {len(user_stats.columns)} user features for {len(user_stats)} users")
        
        return user_stats
    
    def _create_user_segments(self, user_stats: pd.DataFrame) -> pd.Series:
        """Create user behavioral segments."""
        segments = []
        
        for _, user in user_stats.iterrows():
            total_interactions = user.get('total_interactions', 0)
            avg_rating = user.get('avg_rating', 3.0)
            days_since_last = user.get('days_since_last_interaction', 999)
            
            if total_interactions >= 50 and days_since_last <= 7:
                segment = 'power_user'
            elif total_interactions >= 20 and days_since_last <= 30:
                segment = 'regular_user'
            elif total_interactions >= 5 and days_since_last <= 90:
                segment = 'occasional_user'
            elif days_since_last > 90:
                segment = 'dormant_user'
            else:
                segment = 'new_user'
            
            segments.append(segment)
        
        return pd.Series(segments, index=user_stats.index)
    
    @log_execution_time("item_features")
    def create_item_features(self, interactions_df: pd.DataFrame, items_df: pd.DataFrame = None) -> pd.DataFrame:
        """Create comprehensive item features."""
        logger.info("Creating item features...")
        
        # Basic item statistics from interactions
        item_stats = interactions_df.groupby('item_id').agg({
            'user_id': 'count',  # Total interactions
            'rating': ['mean', 'std', 'count'],  # Rating statistics
            'timestamp': ['min', 'max']  # Activity period
        }).round(3)
        
        # Flatten column names
        item_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in item_stats.columns]
        item_stats = item_stats.rename(columns={
            'user_id_count': 'total_interactions',
            'rating_mean': 'avg_rating',
            'rating_std': 'rating_variance',
            'rating_count': 'rating_count',
            'timestamp_min': 'first_interaction',
            'timestamp_max': 'last_interaction'
        })
        
        item_stats['rating_variance'] = item_stats['rating_variance'].fillna(0)
        
        # Popularity metrics
        total_users = interactions_df['user_id'].nunique()
        item_stats['popularity_score'] = item_stats['total_interactions'] / total_users
        item_stats['user_coverage'] = interactions_df.groupby('item_id')['user_id'].nunique()
        
        # Recency features
        current_time = datetime.utcnow()
        if 'last_interaction' in item_stats.columns:
            item_stats['days_since_last_interaction'] = (
                current_time - pd.to_datetime(item_stats['last_interaction'])
            ).dt.days
        
        # Interaction type distribution
        if 'interaction_type' in interactions_df.columns:
            interaction_types = interactions_df.groupby(['item_id', 'interaction_type']).size().unstack(fill_value=0)
            interaction_types.columns = [f'interactions_{col}' for col in interaction_types.columns]
            item_stats = item_stats.join(interaction_types, how='left').fillna(0)
            
            # Conversion metrics
            if 'interactions_view' in item_stats.columns and 'interactions_purchase' in item_stats.columns:
                item_stats['conversion_rate'] = np.where(
                    item_stats['interactions_view'] > 0,
                    item_stats['interactions_purchase'] / item_stats['interactions_view'],
                    0
                )
        
        # Time-based features
        item_stats['item_age_days'] = (current_time - pd.to_datetime(item_stats['first_interaction'])).dt.days
        
        # Merge with provided item data
        if items_df is not None:
            item_stats = item_stats.join(items_df.set_index('item_id'), how='left')
            
            # Create content features
            if 'category' in items_df.columns:
                # Category encoding
                le_category = LabelEncoder()
                item_stats['category_encoded'] = le_category.fit_transform(
                    item_stats['category'].fillna('unknown')
                )
                self.encoders['item_category'] = le_category
            
            if 'brand' in items_df.columns:
                # Brand encoding
                le_brand = LabelEncoder()
                item_stats['brand_encoded'] = le_brand.fit_transform(
                    item_stats['brand'].fillna('unknown')
                )
                self.encoders['item_brand'] = le_brand
            
            # Price features
            if 'price' in items_df.columns:
                item_stats['price_log'] = np.log1p(item_stats['price'])
                
                # Price percentiles within category
                if 'category' in item_stats.columns:
                    item_stats['price_rank_in_category'] = item_stats.groupby('category')['price'].rank(pct=True)
            
            # Text features (if description available)
            if 'description' in items_df.columns:
                tfidf = TfidfVectorizer(max_features=100, stop_words='english')
                descriptions = item_stats['description'].fillna('')
                tfidf_matrix = tfidf.fit_transform(descriptions)
                
                # Add TF-IDF features
                tfidf_features = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    index=item_stats.index,
                    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                )
                item_stats = pd.concat([item_stats, tfidf_features], axis=1)
                
                self.encoders['item_tfidf'] = tfidf
        
        # Item lifecycle stage
        item_stats['lifecycle_stage'] = self._create_item_lifecycle_stages(item_stats)
        
        # Scale numerical features
        numerical_cols = item_stats.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        item_stats_scaled = item_stats.copy()
        item_stats_scaled[numerical_cols] = scaler.fit_transform(item_stats[numerical_cols])
        self.scalers['item_features'] = scaler
        
        # Store feature statistics
        self.feature_stats['item_features'] = {
            'n_items': len(item_stats),
            'n_features': len(item_stats.columns),
            'feature_names': list(item_stats.columns)
        }
        
        item_stats.reset_index(inplace=True)
        logger.info(f"Created {len(item_stats.columns)} item features for {len(item_stats)} items")
        
        return item_stats
    
    def _create_item_lifecycle_stages(self, item_stats: pd.DataFrame) -> pd.Series:
        """Create item lifecycle stages based on interaction patterns."""
        stages = []
        
        for _, item in item_stats.iterrows():
            total_interactions = item.get('total_interactions', 0)
            days_since_last = item.get('days_since_last_interaction', 999)
            item_age = item.get('item_age_days', 0)
            
            if item_age <= 30 and total_interactions >= 10:
                stage = 'trending'
            elif item_age <= 30:
                stage = 'new'
            elif total_interactions >= 100 and days_since_last <= 7:
                stage = 'popular'
            elif total_interactions >= 20 and days_since_last <= 30:
                stage = 'steady'
            elif days_since_last > 90:
                stage = 'declining'
            else:
                stage = 'niche'
            
            stages.append(stage)
        
        return pd.Series(stages, index=item_stats.index)
    
    @log_execution_time("interaction_features")
    def create_interaction_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for each user-item interaction."""
        logger.info("Creating interaction features...")
        
        interactions_features = interactions_df.copy()
        
        # Time-based features
        interactions_features['timestamp'] = pd.to_datetime(interactions_features['timestamp'])
        interactions_features['hour'] = interactions_features['timestamp'].dt.hour
        interactions_features['day_of_week'] = interactions_features['timestamp'].dt.dayofweek
        interactions_features['month'] = interactions_features['timestamp'].dt.month
        interactions_features['is_weekend'] = interactions_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Time since user's first interaction
        user_first_interaction = interactions_features.groupby('user_id')['timestamp'].min()
        interactions_features['days_since_user_start'] = (
            interactions_features['timestamp'] - 
            interactions_features['user_id'].map(user_first_interaction)
        ).dt.days
        
        # Time since item's first interaction
        item_first_interaction = interactions_features.groupby('item_id')['timestamp'].min()
        interactions_features['days_since_item_start'] = (
            interactions_features['timestamp'] - 
            interactions_features['item_id'].map(item_first_interaction)
        ).dt.days
        
        # User interaction sequence features
        interactions_features = interactions_features.sort_values(['user_id', 'timestamp'])
        interactions_features['user_interaction_sequence'] = interactions_features.groupby('user_id').cumcount() + 1
        
        # Time between consecutive interactions for same user
        interactions_features['time_since_last_interaction'] = (
            interactions_features.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
        ).fillna(0)  # in hours
        
        # Rating features
        if 'rating' in interactions_features.columns:
            # User's average rating up to this point
            interactions_features['user_avg_rating_so_far'] = (
                interactions_features.groupby('user_id')['rating']
                .expanding().mean().reset_index(0, drop=True)
            )
            
            # Item's average rating up to this point
            interactions_features['item_avg_rating_so_far'] = (
                interactions_features.groupby('item_id')['rating']
                .expanding().mean().reset_index(0, drop=True)
            )
        
        # Interaction type encoding
        if 'interaction_type' in interactions_features.columns:
            le_interaction = LabelEncoder()
            interactions_features['interaction_type_encoded'] = le_interaction.fit_transform(
                interactions_features['interaction_type']
            )
            self.encoders['interaction_type'] = le_interaction
        
        # Session features (group interactions within 1 hour as sessions)
        interactions_features['session_id'] = self._create_session_ids(interactions_features)
        interactions_features['session_position'] = interactions_features.groupby('session_id').cumcount() + 1
        session_lengths = interactions_features.groupby('session_id').size()
        interactions_features['session_length'] = interactions_features['session_id'].map(session_lengths)
        
        # Contextual features
        interactions_features['is_repeat_item'] = interactions_features.groupby(['user_id', 'item_id']).cumcount() > 0
        
        # Store feature statistics
        self.feature_stats['interaction_features'] = {
            'n_interactions': len(interactions_features),
            'n_features': len(interactions_features.columns),
            'feature_names': list(interactions_features.columns)
        }
        
        logger.info(f"Created {len(interactions_features.columns)} interaction features for {len(interactions_features)} interactions")
        
        return interactions_features
    
    def _create_session_ids(self, interactions_df: pd.DataFrame) -> pd.Series:
        """Create session IDs based on time gaps between interactions."""
        interactions_df = interactions_df.sort_values(['user_id', 'timestamp'])
        
        # Mark session breaks (gaps > 1 hour)
        session_breaks = (
            interactions_df.groupby('user_id')['timestamp'].diff() > timedelta(hours=1)
        ) | (interactions_df['user_id'] != interactions_df['user_id'].shift())
        
        # Create session IDs
        session_ids = session_breaks.cumsum()
        
        return session_ids
    
    @log_execution_time("context_features")
    def create_context_features(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual features for recommendation."""
        logger.info("Creating context features...")
        
        context_features = pd.DataFrame(index=interactions_df.index)
        
        # Temporal context
        timestamps = pd.to_datetime(interactions_df['timestamp'])
        context_features['is_business_hours'] = timestamps.dt.hour.between(9, 17).astype(int)
        context_features['is_evening'] = timestamps.dt.hour.between(18, 22).astype(int)
        context_features['is_night'] = timestamps.dt.hour.between(23, 6).astype(int)
        context_features['is_morning'] = timestamps.dt.hour.between(6, 11).astype(int)
        
        # Seasonal features
        context_features['season'] = timestamps.dt.month.map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })
        
        # Encode season
        le_season = LabelEncoder()
        context_features['season_encoded'] = le_season.fit_transform(context_features['season'])
        self.encoders['season'] = le_season
        
        # User activity context
        user_hourly_activity = interactions_df.groupby(['user_id', timestamps.dt.hour]).size().unstack(fill_value=0)
        user_peak_hour = user_hourly_activity.idxmax(axis=1)
        context_features['is_user_peak_hour'] = (
            timestamps.dt.hour == interactions_df['user_id'].map(user_peak_hour)
        ).astype(int)
        
        # Item popularity context
        item_hourly_activity = interactions_df.groupby(['item_id', timestamps.dt.hour]).size().unstack(fill_value=0)
        item_peak_hour = item_hourly_activity.idxmax(axis=1)
        context_features['is_item_peak_hour'] = (
            timestamps.dt.hour == interactions_df['item_id'].map(item_peak_hour)
        ).astype(int)
        
        logger.info(f"Created {len(context_features.columns)} context features")
        
        return context_features
    
    @log_execution_time("feature_engineering")
    def create_all_features(self, interactions_df: pd.DataFrame, 
                          users_df: pd.DataFrame = None, 
                          items_df: pd.DataFrame = None) -> Dict[str, pd.DataFrame]:
        """Create all feature sets."""
        logger.info("Starting comprehensive feature engineering...")
        
        # Create all feature sets
        user_features = self.create_user_features(interactions_df, users_df)
        item_features = self.create_item_features(interactions_df, items_df)
        interaction_features = self.create_interaction_features(interactions_df)
        context_features = self.create_context_features(interactions_df)
        
        # Combine interaction and context features
        combined_interactions = pd.concat([interaction_features, context_features], axis=1)
        
        feature_sets = {
            'users': user_features,
            'items': item_features,
            'interactions': combined_interactions
        }
        
        # Save feature engineering artifacts
        self._save_feature_artifacts()
        
        logger.info("Feature engineering completed successfully")
        logger.info(f"Feature statistics: {self.feature_stats}")
        
        return feature_sets
    
    def _save_feature_artifacts(self):
        """Save feature engineering artifacts."""
        artifacts_dir = Path(self.config.feature_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = artifacts_dir / f"{name}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                import pickle
                pickle.dump(scaler, f)
        
        # Save encoders
        for name, encoder in self.encoders.items():
            encoder_path = artifacts_dir / f"{name}_encoder.pkl"
            with open(encoder_path, 'wb') as f:
                import pickle
                pickle.dump(encoder, f)
        
        # Save feature statistics
        stats_path = artifacts_dir / "feature_stats.pkl"
        with open(stats_path, 'wb') as f:
            import pickle
            pickle.dump(self.feature_stats, f)
        
        logger.info(f"Saved feature engineering artifacts to {artifacts_dir}")


def main():
    """Main feature engineering function."""
    config = get_config()
    engineer = FeatureEngineer(config)
    
    # Load sample data for demonstration
    data_dir = Path(config.data_dir) / "processed"
    
    # Load interactions
    train_path = data_dir / "train.csv"
    if train_path.exists():
        interactions_df = pd.read_csv(train_path)
        logger.info(f"Loaded {len(interactions_df)} training interactions")
        
        # Create features
        feature_sets = engineer.create_all_features(interactions_df)
        
        # Save features
        for feature_name, features_df in feature_sets.items():
            output_path = data_dir / f"{feature_name}_features.csv"
            features_df.to_csv(output_path, index=False)
            logger.info(f"Saved {feature_name} features to {output_path}")
    
    else:
        logger.error(f"Training data not found at {train_path}")


if __name__ == "__main__":
    main()