"""
Sample data generator for testing and demonstration.
Creates realistic synthetic e-commerce interaction data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple

from ..utils.config import get_config
from ..utils.logging import setup_logging

logger = setup_logging(__name__)


class SampleDataGenerator:
    """Generate sample data for testing and demonstration."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        np.random.seed(42)  # For reproducible results
    
    def generate_sample_dataset(
        self,
        n_users: int = 1000,
        n_items: int = 500,
        n_interactions: int = 10000,
        rating_probability: float = 0.3
    ) -> Dict[str, pd.DataFrame]:
        """Generate a complete sample dataset."""
        logger.info(f"Generating sample dataset with {n_users} users, {n_items} items, {n_interactions} interactions")
        
        # Generate users
        users_df = self._generate_users(n_users)
        
        # Generate items
        items_df = self._generate_items(n_items)
        
        # Generate interactions
        interactions_df = self._generate_interactions(
            users_df, items_df, n_interactions, rating_probability
        )
        
        return {
            'users': users_df,
            'items': items_df,
            'interactions': interactions_df
        }
    
    def _generate_users(self, n_users: int) -> pd.DataFrame:
        """Generate user data."""
        logger.info(f"Generating {n_users} users...")
        
        users_data = []
        
        # User demographics
        age_distribution = np.random.normal(35, 12, n_users).clip(18, 80).astype(int)
        genders = np.random.choice(['M', 'F', 'O'], n_users, p=[0.48, 0.48, 0.04])
        
        locations = np.random.choice(
            ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'Other'],
            n_users,
            p=[0.3, 0.15, 0.1, 0.08, 0.08, 0.06, 0.05, 0.18]
        )
        
        # User behavior segments
        segments = np.random.choice(
            ['power_user', 'regular', 'occasional', 'new'],
            n_users,
            p=[0.1, 0.4, 0.35, 0.15]
        )
        
        # Registration dates
        base_date = datetime.now() - timedelta(days=365)
        registration_days_ago = np.random.exponential(100, n_users).clip(1, 365)
        
        for i in range(n_users):
            user_id = i + 1
            
            # Calculate registration date
            reg_date = base_date + timedelta(days=365 - registration_days_ago[i])
            
            # User preferences (affects item interaction patterns)
            preferred_categories = np.random.choice(
                ['Electronics', 'Clothing', 'Books', 'Home', 'Sports', 'Beauty'],
                size=np.random.randint(1, 4),
                replace=False
            )
            
            users_data.append({
                'user_id': user_id,
                'age': age_distribution[i],
                'gender': genders[i],
                'location': locations[i],
                'segment': segments[i],
                'registration_date': reg_date,
                'preferred_categories': ','.join(preferred_categories)
            })
        
        return pd.DataFrame(users_data)
    
    def _generate_items(self, n_items: int) -> pd.DataFrame:
        """Generate item data."""
        logger.info(f"Generating {n_items} items...")
        
        # Item categories and brands
        categories = {
            'Electronics': ['Apple', 'Samsung', 'Sony', 'LG', 'HP'],
            'Clothing': ['Nike', 'Adidas', 'Zara', 'H&M', 'Uniqlo'],
            'Books': ['Penguin', 'Harper', 'Random House', 'Simon & Schuster', 'Hachette'],
            'Home': ['IKEA', 'West Elm', 'CB2', 'Pottery Barn', 'Crate & Barrel'],
            'Sports': ['Nike', 'Adidas', 'Under Armour', 'Puma', 'Reebok'],
            'Beauty': ['L\'Oreal', 'Maybelline', 'MAC', 'Sephora', 'Ulta']
        }
        
        items_data = []
        
        for i in range(n_items):
            item_id = i + 1
            
            # Select category and brand
            category = np.random.choice(list(categories.keys()))
            brand = np.random.choice(categories[category])
            
            # Generate price based on category
            price_ranges = {
                'Electronics': (50, 2000),
                'Clothing': (20, 300),
                'Books': (10, 50),
                'Home': (25, 500),
                'Sports': (30, 400),
                'Beauty': (15, 150)
            }
            
            min_price, max_price = price_ranges[category]
            price = np.random.lognormal(
                np.log(min_price + (max_price - min_price) * 0.3),
                0.5
            )
            price = np.clip(price, min_price, max_price)
            
            # Item launch date
            launch_date = datetime.now() - timedelta(
                days=np.random.exponential(180).clip(1, 1000)
            )
            
            # Item popularity (affects interaction probability)
            popularity_score = np.random.beta(2, 5)  # Skewed towards lower values
            
            # Generate item features
            features = {
                'color': np.random.choice(['Red', 'Blue', 'Black', 'White', 'Green', 'Other']),
                'size': np.random.choice(['S', 'M', 'L', 'XL', 'One Size']),
                'material': np.random.choice(['Cotton', 'Polyester', 'Metal', 'Plastic', 'Wood', 'Other']),
                'rating': np.random.normal(4.0, 0.8).clip(1, 5)
            }
            
            items_data.append({
                'item_id': item_id,
                'category': category,
                'brand': brand,
                'price': round(price, 2),
                'launch_date': launch_date,
                'popularity_score': popularity_score,
                'description': f"{brand} {category.lower()} item with premium quality",
                **features
            })
        
        return pd.DataFrame(items_data)
    
    def _generate_interactions(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        n_interactions: int,
        rating_probability: float
    ) -> pd.DataFrame:
        """Generate interaction data."""
        logger.info(f"Generating {n_interactions} interactions...")
        
        interactions_data = []
        
        # Interaction types and their conversion probabilities
        interaction_funnel = {
            'view': {'weight': 0.6, 'next_prob': {'cart': 0.15, 'purchase': 0.05}},
            'cart': {'weight': 0.2, 'next_prob': {'purchase': 0.3}},
            'purchase': {'weight': 0.15, 'next_prob': {'rating': rating_probability}},
            'rating': {'weight': 0.05, 'next_prob': {}}
        }
        
        # User activity patterns
        user_activity = {}
        for _, user in users_df.iterrows():
            if user['segment'] == 'power_user':
                activity_level = np.random.poisson(20)
            elif user['segment'] == 'regular':
                activity_level = np.random.poisson(8)
            elif user['segment'] == 'occasional':
                activity_level = np.random.poisson(3)
            else:  # new
                activity_level = np.random.poisson(1)
            
            user_activity[user['user_id']] = max(1, activity_level)
        
        # Generate interactions
        for _ in range(n_interactions):
            # Select user based on activity level
            user_weights = [user_activity.get(uid, 1) for uid in users_df['user_id']]
            user_idx = np.random.choice(len(users_df), p=np.array(user_weights)/np.sum(user_weights))
            user = users_df.iloc[user_idx]
            
            # Select item based on user preferences and item popularity
            user_categories = user['preferred_categories'].split(',')
            
            # Higher probability for preferred categories
            item_weights = []
            for _, item in items_df.iterrows():
                weight = item['popularity_score']
                if item['category'] in user_categories:
                    weight *= 3  # 3x more likely for preferred categories
                item_weights.append(weight)
            
            item_weights = np.array(item_weights)
            item_idx = np.random.choice(len(items_df), p=item_weights/item_weights.sum())
            item = items_df.iloc[item_idx]
            
            # Determine interaction type
            interaction_type = np.random.choice(
                list(interaction_funnel.keys()),
                p=[info['weight'] for info in interaction_funnel.values()]
            )
            
            # Generate timestamp
            # Recent activity is more likely
            days_ago = np.random.exponential(7).clip(0, 90)
            timestamp = datetime.now() - timedelta(days=days_ago, 
                                                   hours=np.random.randint(0, 24),
                                                   minutes=np.random.randint(0, 60))
            
            # Generate rating if applicable
            rating = None
            if interaction_type == 'rating' or (interaction_type == 'purchase' and np.random.random() < 0.3):
                # Base rating on item quality, user satisfaction
                base_rating = item['rating']
                user_rating = np.random.normal(base_rating, 0.5).clip(1, 5)
                rating = round(user_rating)
            
            # Session ID (group nearby interactions)
            session_id = f"session_{user['user_id']}_{int(timestamp.timestamp() // 3600)}"
            
            interactions_data.append({
                'user_id': user['user_id'],
                'item_id': item['item_id'],
                'interaction_type': interaction_type,
                'rating': rating,
                'timestamp': timestamp,
                'session_id': session_id,
                'device': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.3, 0.1]),
                'channel': np.random.choice(['organic', 'paid', 'social', 'email'], p=[0.4, 0.3, 0.2, 0.1])
            })
        
        return pd.DataFrame(interactions_data)
    
    def add_seasonal_patterns(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Add seasonal patterns to interaction data."""
        logger.info("Adding seasonal patterns...")
        
        # Add seasonal multipliers
        interactions_df['month'] = interactions_df['timestamp'].dt.month
        interactions_df['day_of_week'] = interactions_df['timestamp'].dt.dayofweek
        interactions_df['hour'] = interactions_df['timestamp'].dt.hour
        
        # Seasonal factors
        seasonal_multipliers = {
            11: 1.5,  # November (pre-holiday)
            12: 2.0,  # December (holiday season)
            1: 1.3,   # January (New Year)
            2: 0.8,   # February
            3: 0.9,   # March
            4: 1.0,   # April
            5: 1.1,   # May
            6: 1.0,   # June
            7: 1.1,   # July
            8: 1.0,   # August
            9: 1.0,   # September
            10: 1.1   # October
        }
        
        # Weekend patterns (higher activity)
        weekend_multiplier = 1.2
        
        # Hour patterns (peak hours)
        hour_multipliers = {
            0: 0.3, 1: 0.2, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
            6: 0.4, 7: 0.6, 8: 0.8, 9: 1.0, 10: 1.1, 11: 1.2,
            12: 1.3, 13: 1.2, 14: 1.1, 15: 1.0, 16: 1.1, 17: 1.2,
            18: 1.4, 19: 1.5, 20: 1.4, 21: 1.2, 22: 1.0, 23: 0.6
        }
        
        return interactions_df
    
    def generate_test_scenarios(self) -> Dict[str, Dict]:
        """Generate specific test scenarios for validation."""
        logger.info("Generating test scenarios...")
        
        scenarios = {
            'cold_start_user': {
                'description': 'New user with no interaction history',
                'user_id': 99999,
                'expected_recommendations': 'popular_items'
            },
            'cold_start_item': {
                'description': 'New item with no interaction history',
                'item_id': 99999,
                'expected_recommendations': 'similar_category_items'
            },
            'active_user': {
                'description': 'Power user with many interactions',
                'user_id': 1,
                'expected_recommendations': 'personalized_diverse'
            },
            'niche_user': {
                'description': 'User with very specific preferences',
                'user_id': 100,
                'expected_recommendations': 'category_specific'
            }
        }
        
        return scenarios
    
    def save_sample_data(self, data: Dict[str, pd.DataFrame], output_dir: str = None):
        """Save sample data to files."""
        if output_dir is None:
            output_dir = Path(self.config.data_dir) / "sample"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving sample data to {output_dir}")
        
        for data_type, df in data.items():
            file_path = output_dir / f"{data_type}.csv"
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {data_type}: {len(df)} records -> {file_path}")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'generator_config': {
                'n_users': len(data.get('users', [])),
                'n_items': len(data.get('items', [])),
                'n_interactions': len(data.get('interactions', []))
            },
            'data_statistics': self._calculate_sample_stats(data)
        }
        
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def _calculate_sample_stats(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate statistics for sample data."""
        stats = {}
        
        if 'interactions' in data:
            interactions_df = data['interactions']
            
            stats['interactions'] = {
                'total': len(interactions_df),
                'unique_users': interactions_df['user_id'].nunique(),
                'unique_items': interactions_df['item_id'].nunique(),
                'interaction_types': interactions_df['interaction_type'].value_counts().to_dict(),
                'ratings_distribution': interactions_df['rating'].value_counts().to_dict(),
                'date_range': {
                    'start': interactions_df['timestamp'].min().isoformat(),
                    'end': interactions_df['timestamp'].max().isoformat()
                }
            }
        
        if 'users' in data:
            users_df = data['users']
            stats['users'] = {
                'total': len(users_df),
                'age_distribution': {
                    'mean': users_df['age'].mean(),
                    'std': users_df['age'].std(),
                    'min': users_df['age'].min(),
                    'max': users_df['age'].max()
                },
                'gender_distribution': users_df['gender'].value_counts().to_dict(),
                'location_distribution': users_df['location'].value_counts().to_dict()
            }
        
        if 'items' in data:
            items_df = data['items']
            stats['items'] = {
                'total': len(items_df),
                'category_distribution': items_df['category'].value_counts().to_dict(),
                'brand_distribution': items_df['brand'].value_counts().to_dict(),
                'price_distribution': {
                    'mean': items_df['price'].mean(),
                    'std': items_df['price'].std(),
                    'min': items_df['price'].min(),
                    'max': items_df['price'].max()
                }
            }
        
        return stats


def main():
    """Generate sample data for testing."""
    config = get_config()
    generator = SampleDataGenerator(config)
    
    # Generate different sized datasets
    datasets = {
        'small': {'n_users': 100, 'n_items': 50, 'n_interactions': 1000},
        'medium': {'n_users': 1000, 'n_items': 500, 'n_interactions': 10000},
        'large': {'n_users': 5000, 'n_items': 2000, 'n_interactions': 50000}
    }
    
    for size_name, params in datasets.items():
        print(f"\nGenerating {size_name} dataset...")
        
        # Generate data
        data = generator.generate_sample_dataset(**params)
        
        # Add seasonal patterns
        data['interactions'] = generator.add_seasonal_patterns(data['interactions'])
        
        # Save data
        output_dir = Path(config.data_dir) / "sample" / size_name
        generator.save_sample_data(data, output_dir)
        
        print(f"Generated {size_name} dataset:")
        for data_type, df in data.items():
            print(f"  {data_type}: {len(df)} records")
    
    # Generate test scenarios
    test_scenarios = generator.generate_test_scenarios()
    scenarios_path = Path(config.data_dir) / "sample" / "test_scenarios.json"
    with open(scenarios_path, 'w') as f:
        json.dump(test_scenarios, f, indent=2)
    
    print(f"\nTest scenarios saved to {scenarios_path}")
    print("Sample data generation completed!")


if __name__ == "__main__":
    main()