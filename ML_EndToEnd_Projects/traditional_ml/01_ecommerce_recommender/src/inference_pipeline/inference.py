"""
Recommendation inference engine for real-time recommendations.
"""

import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional, Union
import asyncio
import logging
from pathlib import Path
import redis
import json
from datetime import datetime
import torch
import torch.nn as nn

from ..utils.config import get_config
from ..utils.logging import setup_logging

logger = setup_logging(__name__)


class RecommendationEngine:
    """Main recommendation engine for real-time inference."""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.encoders = {}
        self.item_features = None
        self.user_features = None
        self.redis_client = None
        
        # Initialize Redis if available
        try:
            self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
    
    async def load_models(self):
        """Load all trained models and encoders."""
        model_dir = Path(self.config.model_dir)
        
        try:
            # Load collaborative filtering model
            cf_path = model_dir / "collaborative" / "svd_model.pkl"
            if cf_path.exists():
                with open(cf_path, 'rb') as f:
                    self.models['collaborative'] = pickle.load(f)
                logger.info("Loaded collaborative filtering model")
            
            # Load content-based model
            cb_path = model_dir / "content_based" / "tfidf_model.pkl"
            if cb_path.exists():
                with open(cb_path, 'rb') as f:
                    self.models['content'] = pickle.load(f)
                logger.info("Loaded content-based model")
            
            # Load encoders
            encoder_dir = model_dir / "encoders"
            if encoder_dir.exists():
                for encoder_file in encoder_dir.glob("*.pkl"):
                    encoder_name = encoder_file.stem
                    with open(encoder_file, 'rb') as f:
                        self.encoders[encoder_name] = pickle.load(f)
                    logger.info(f"Loaded encoder: {encoder_name}")
            
            # Load feature data
            data_dir = Path(self.config.data_dir) / "features"
            if (data_dir / "item_features.csv").exists():
                self.item_features = pd.read_csv(data_dir / "item_features.csv")
                logger.info(f"Loaded item features: {len(self.item_features)} items")
            
            if (data_dir / "user_features.csv").exists():
                self.user_features = pd.read_csv(data_dir / "user_features.csv")
                logger.info(f"Loaded user features: {len(self.user_features)} users")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    async def get_user_recommendations(self,
                                     user_id: int,
                                     num_recommendations: int = 10,
                                     algorithm: str = "hybrid",
                                     include_seen: bool = False,
                                     diversity_lambda: Optional[float] = None,
                                     categories: Optional[List[str]] = None,
                                     min_rating: Optional[float] = None) -> List[Dict]:
        """Generate recommendations for a user."""
        
        # Check cache first
        cache_key = f"user_rec:{user_id}:{algorithm}:{num_recommendations}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        
        try:
            if algorithm == "collaborative":
                recommendations = await self._collaborative_recommendations(
                    user_id, num_recommendations, include_seen
                )
            elif algorithm == "content":
                recommendations = await self._content_based_recommendations(
                    user_id, num_recommendations, categories
                )
            else:  # hybrid
                recommendations = await self._hybrid_recommendations(
                    user_id, num_recommendations, include_seen, categories
                )
            
            # Apply filters
            if min_rating:
                recommendations = [r for r in recommendations if r['score'] >= min_rating]
            
            # Apply diversity if specified
            if diversity_lambda:
                recommendations = self._apply_diversity(recommendations, diversity_lambda)
            
            # Cache results
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        cache_key, 
                        self.config.cache_ttl, 
                        json.dumps(recommendations, default=str)
                    )
                except Exception:
                    pass
            
            return recommendations[:num_recommendations]
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            # Return popular items as fallback
            return await self._popular_recommendations(num_recommendations)
    
    async def _collaborative_recommendations(self, user_id: int, num_recs: int, include_seen: bool) -> List[Dict]:
        """Generate collaborative filtering recommendations."""
        if 'collaborative' not in self.models:
            raise ValueError("Collaborative filtering model not loaded")
        
        model = self.models['collaborative']
        
        # Generate recommendations using SVD
        # This is a simplified implementation - actual would depend on the specific model
        all_items = list(range(1, 1001))  # Assume items 1-1000
        user_ratings = []
        
        for item_id in all_items:
            try:
                # Predict rating for user-item pair
                prediction = model.predict(user_id, item_id)
                user_ratings.append({
                    'item_id': item_id,
                    'score': prediction.est,
                    'algorithm': 'collaborative'
                })
            except:
                continue
        
        # Sort by predicted rating
        user_ratings.sort(key=lambda x: x['score'], reverse=True)
        
        return user_ratings[:num_recs * 2]  # Return extra for filtering
    
    async def _content_based_recommendations(self, user_id: int, num_recs: int, categories: Optional[List[str]]) -> List[Dict]:
        """Generate content-based recommendations."""
        if self.item_features is None:
            raise ValueError("Item features not loaded")
        
        # Get user's historical preferences
        user_profile = self._get_user_profile(user_id)
        
        recommendations = []
        for _, item in self.item_features.iterrows():
            if categories and item.get('category') not in categories:
                continue
            
            # Calculate content similarity score
            score = self._calculate_content_similarity(user_profile, item)
            
            recommendations.append({
                'item_id': int(item['item_id']),
                'score': float(score),
                'algorithm': 'content',
                'category': item.get('category', 'Unknown'),
                'brand': item.get('brand', 'Unknown')
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:num_recs * 2]
    
    async def _hybrid_recommendations(self, user_id: int, num_recs: int, include_seen: bool, categories: Optional[List[str]]) -> List[Dict]:
        """Generate hybrid recommendations combining multiple algorithms."""
        weights = self.config.hybrid_weights
        
        # Get recommendations from different algorithms
        cf_recs = await self._collaborative_recommendations(user_id, num_recs * 2, include_seen)
        cb_recs = await self._content_based_recommendations(user_id, num_recs * 2, categories)
        pop_recs = await self._popular_recommendations(num_recs)
        
        # Combine recommendations with weights
        all_recs = {}
        
        # Add collaborative filtering recommendations
        for rec in cf_recs:
            item_id = rec['item_id']
            all_recs[item_id] = all_recs.get(item_id, 0) + rec['score'] * weights['collaborative']
        
        # Add content-based recommendations
        for rec in cb_recs:
            item_id = rec['item_id']
            all_recs[item_id] = all_recs.get(item_id, 0) + rec['score'] * weights['content']
        
        # Add popularity boost
        for rec in pop_recs:
            item_id = rec['item_id']
            all_recs[item_id] = all_recs.get(item_id, 0) + rec['score'] * weights['popularity']
        
        # Convert to list format
        hybrid_recs = []
        for item_id, score in all_recs.items():
            item_info = self._get_item_info(item_id)
            hybrid_recs.append({
                'item_id': item_id,
                'score': score,
                'algorithm': 'hybrid',
                **item_info
            })
        
        hybrid_recs.sort(key=lambda x: x['score'], reverse=True)
        return hybrid_recs
    
    async def _popular_recommendations(self, num_recs: int) -> List[Dict]:
        """Generate popular item recommendations as fallback."""
        # Simulate popular items (in real implementation, this would come from analytics)
        popular_items = [
            {'item_id': 1, 'score': 0.95, 'algorithm': 'popularity'},
            {'item_id': 2, 'score': 0.92, 'algorithm': 'popularity'},
            {'item_id': 3, 'score': 0.89, 'algorithm': 'popularity'},
            {'item_id': 4, 'score': 0.86, 'algorithm': 'popularity'},
            {'item_id': 5, 'score': 0.83, 'algorithm': 'popularity'},
        ]
        
        # Add item info
        for item in popular_items:
            item.update(self._get_item_info(item['item_id']))
        
        return popular_items[:num_recs]
    
    async def get_similar_items(self, item_id: int, num_similar: int = 10, algorithm: str = "content") -> List[Dict]:
        """Get items similar to a given item."""
        cache_key = f"item_sim:{item_id}:{algorithm}:{num_similar}"
        
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        
        if algorithm == "content":
            similar_items = self._content_similarity(item_id, num_similar)
        else:
            similar_items = self._collaborative_similarity(item_id, num_similar)
        
        # Cache results
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key, 
                    self.config.cache_ttl, 
                    json.dumps(similar_items, default=str)
                )
            except Exception:
                pass
        
        return similar_items
    
    def _get_user_profile(self, user_id: int) -> Dict:
        """Get user profile from features or interactions."""
        if self.user_features is not None:
            user_data = self.user_features[self.user_features['user_id'] == user_id]
            if not user_data.empty:
                return user_data.iloc[0].to_dict()
        
        # Default profile
        return {
            'age': 30,
            'gender': 'M',
            'location': 'US',
            'preferred_categories': ['Electronics', 'Books']
        }
    
    def _get_item_info(self, item_id: int) -> Dict:
        """Get item information."""
        if self.item_features is not None:
            item_data = self.item_features[self.item_features['item_id'] == item_id]
            if not item_data.empty:
                item = item_data.iloc[0]
                return {
                    'category': item.get('category', 'Unknown'),
                    'brand': item.get('brand', 'Unknown'),
                    'price': float(item.get('price', 0.0))
                }
        
        # Default item info
        return {
            'category': 'Electronics',
            'brand': 'Unknown',
            'price': 29.99
        }
    
    def _calculate_content_similarity(self, user_profile: Dict, item: pd.Series) -> float:
        """Calculate content-based similarity score."""
        score = 0.5  # Base score
        
        # Category preference
        if 'preferred_categories' in user_profile:
            if item.get('category') in user_profile['preferred_categories']:
                score += 0.3
        
        # Price preference (simulate price preference based on user profile)
        user_age = user_profile.get('age', 30)
        item_price = item.get('price', 50)
        if user_age > 40 and item_price > 100:
            score += 0.2
        elif user_age <= 25 and item_price < 50:
            score += 0.2
        
        return min(score, 1.0)
    
    def _content_similarity(self, item_id: int, num_similar: int) -> List[Dict]:
        """Find content-based similar items."""
        target_item = self._get_item_info(item_id)
        similar_items = []
        
        if self.item_features is not None:
            for _, item in self.item_features.iterrows():
                if int(item['item_id']) == item_id:
                    continue
                
                similarity = 0.0
                
                # Category similarity
                if item.get('category') == target_item.get('category'):
                    similarity += 0.4
                
                # Brand similarity
                if item.get('brand') == target_item.get('brand'):
                    similarity += 0.3
                
                # Price similarity
                price_diff = abs(item.get('price', 0) - target_item.get('price', 0))
                if price_diff < 20:
                    similarity += 0.3
                
                similar_items.append({
                    'item_id': int(item['item_id']),
                    'similarity': similarity,
                    'category': item.get('category', 'Unknown'),
                    'brand': item.get('brand', 'Unknown'),
                    'price': float(item.get('price', 0))
                })
        
        similar_items.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_items[:num_similar]
    
    def _collaborative_similarity(self, item_id: int, num_similar: int) -> List[Dict]:
        """Find collaborative filtering based similar items."""
        # Simplified implementation
        similar_items = []
        for i in range(1, num_similar + 1):
            similar_item_id = item_id + i
            similar_items.append({
                'item_id': similar_item_id,
                'similarity': max(0.1, 1.0 - i * 0.1),
                **self._get_item_info(similar_item_id)
            })
        
        return similar_items
    
    def _apply_diversity(self, recommendations: List[Dict], diversity_lambda: float) -> List[Dict]:
        """Apply diversity to recommendations."""
        if not recommendations:
            return recommendations
        
        diverse_recs = [recommendations[0]]  # Start with top recommendation
        
        for rec in recommendations[1:]:
            # Calculate diversity score
            diversity_score = 1.0
            for selected in diverse_recs:
                if rec.get('category') == selected.get('category'):
                    diversity_score *= 0.7
                if rec.get('brand') == selected.get('brand'):
                    diversity_score *= 0.8
            
            # Combine relevance and diversity
            rec['score'] = rec['score'] * (1 - diversity_lambda) + diversity_score * diversity_lambda
            diverse_recs.append(rec)
        
        diverse_recs.sort(key=lambda x: x['score'], reverse=True)
        return diverse_recs
    
    async def record_interaction(self, user_id: int, item_id: int, interaction_type: str, 
                               rating: Optional[float] = None, timestamp: Optional[datetime] = None):
        """Record user interaction for online learning."""
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'interaction_type': interaction_type,
            'rating': rating,
            'timestamp': timestamp or datetime.utcnow()
        }
        
        # Store in database or queue for batch processing
        logger.info(f"Recorded interaction: {interaction}")
        
        # Invalidate user's cache
        if self.redis_client:
            try:
                pattern = f"user_rec:{user_id}:*"
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
            except Exception:
                pass
    
    async def get_system_stats(self) -> Dict:
        """Get system statistics and metrics."""
        stats = {
            'models_loaded': len(self.models),
            'encoders_loaded': len(self.encoders),
            'timestamp': datetime.utcnow(),
        }
        
        if self.item_features is not None:
            stats['total_items'] = len(self.item_features)
        
        if self.user_features is not None:
            stats['total_users'] = len(self.user_features)
        
        return stats
    
    async def generate_batch_recommendations(self, user_ids: List[int], 
                                           num_recommendations: int, algorithm: str):
        """Generate recommendations for multiple users (background task)."""
        logger.info(f"Starting batch recommendation generation for {len(user_ids)} users")
        
        for user_id in user_ids:
            try:
                recommendations = await self.get_user_recommendations(
                    user_id, num_recommendations, algorithm
                )
                # Store batch results (in real implementation, save to database)
                logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            except Exception as e:
                logger.error(f"Error generating recommendations for user {user_id}: {e}")
        
        logger.info("Batch recommendation generation completed")