"""
FastAPI application for serving recommendation models.
Provides REST API endpoints for real-time recommendations.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio
import redis
import json
from pathlib import Path

from ..inference_pipeline.inference import RecommendationEngine
from ..utils.config import get_config
from ..utils.logging import setup_logging

# Setup logging
logger = setup_logging(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Recommendation API",
    description="Real-time recommendation system for e-commerce platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
recommendation_engine = None
config = None


# Pydantic models for request/response
class UserRecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = Field(default=10, ge=1, le=100)
    model_type: str = Field(default="hybrid", regex="^(collaborative|content|hybrid|popularity)$")
    include_metadata: bool = Field(default=True)
    exclude_seen: bool = Field(default=True)


class ItemRecommendationRequest(BaseModel):
    item_id: int
    num_recommendations: int = Field(default=10, ge=1, le=50)
    include_metadata: bool = Field(default=True)


class BatchRecommendationRequest(BaseModel):
    user_ids: List[int]
    num_recommendations: int = Field(default=5, ge=1, le=20)
    model_type: str = Field(default="hybrid", regex="^(collaborative|content|hybrid|popularity)$")


class RatingPredictionRequest(BaseModel):
    user_id: int
    item_id: int
    model_type: str = Field(default="hybrid", regex="^(collaborative|content|hybrid)$")


class RecommendationResponse(BaseModel):
    item_id: int
    score: float
    rank: int
    metadata: Optional[Dict] = None


class UserRecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationResponse]
    model_type: str
    timestamp: str
    total_time_ms: float


class BatchRecommendationResponse(BaseModel):
    recommendations: Dict[int, List[RecommendationResponse]]
    model_type: str
    timestamp: str
    total_time_ms: float


class RatingPredictionResponse(BaseModel):
    user_id: int
    item_id: int
    predicted_rating: float
    confidence: Optional[float] = None
    model_type: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    cache_connected: bool
    uptime_seconds: float


class StatsResponse(BaseModel):
    total_users: int
    total_items: int
    total_interactions: int
    model_info: Dict
    cache_stats: Dict


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine and dependencies."""
    global recommendation_engine, config
    
    try:
        logger.info("Starting recommendation API server...")
        
        # Load configuration
        config = get_config()
        
        # Initialize recommendation engine
        recommendation_engine = RecommendationEngine(config)
        recommendation_engine.load_models()
        
        logger.info("Recommendation engine initialized successfully")
        
        # Test cache connection
        try:
            cache = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                password=config.redis_password,
                decode_responses=True
            )
            cache.ping()
            logger.info("Cache connection established")
        except Exception as e:
            logger.warning(f"Cache connection failed: {e}")
        
        logger.info("API server startup completed")
        
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources."""
    logger.info("Shutting down recommendation API server...")


# Dependency injection
async def get_recommendation_engine():
    """Dependency to get recommendation engine."""
    if recommendation_engine is None:
        raise HTTPException(status_code=503, detail="Recommendation engine not initialized")
    return recommendation_engine


# API endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    start_time = getattr(app.state, 'start_time', datetime.utcnow())
    uptime = (datetime.utcnow() - start_time).total_seconds()
    
    # Check model status
    model_loaded = recommendation_engine is not None and recommendation_engine.models_loaded
    
    # Check cache connection
    cache_connected = False
    try:
        cache = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        cache.ping()
        cache_connected = True
    except:
        pass
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model_loaded,
        cache_connected=cache_connected,
        uptime_seconds=uptime
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats(engine: RecommendationEngine = Depends(get_recommendation_engine)):
    """Get system statistics."""
    try:
        stats = engine.get_stats()
        
        # Get cache statistics
        cache_stats = {}
        try:
            cache = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                decode_responses=True
            )
            cache_info = cache.info()
            cache_stats = {
                "used_memory": cache_info.get("used_memory_human", "N/A"),
                "connected_clients": cache_info.get("connected_clients", 0),
                "keyspace_hits": cache_info.get("keyspace_hits", 0),
                "keyspace_misses": cache_info.get("keyspace_misses", 0)
            }
        except Exception as e:
            cache_stats = {"error": str(e)}
        
        return StatsResponse(
            total_users=stats.get('total_users', 0),
            total_items=stats.get('total_items', 0),
            total_interactions=stats.get('total_interactions', 0),
            model_info=stats.get('model_info', {}),
            cache_stats=cache_stats
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/user", response_model=UserRecommendationResponse)
async def recommend_for_user(
    request: UserRecommendationRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get recommendations for a specific user."""
    start_time = datetime.utcnow()
    
    try:
        # Get recommendations
        recommendations = engine.get_user_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            model_type=request.model_type,
            exclude_seen=request.exclude_seen
        )
        
        # Format response
        formatted_recommendations = []
        for i, (item_id, score) in enumerate(recommendations):
            metadata = None
            if request.include_metadata:
                metadata = engine.get_item_metadata(item_id)
            
            formatted_recommendations.append(RecommendationResponse(
                item_id=item_id,
                score=float(score),
                rank=i + 1,
                metadata=metadata
            ))
        
        end_time = datetime.utcnow()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return UserRecommendationResponse(
            user_id=request.user_id,
            recommendations=formatted_recommendations,
            model_type=request.model_type,
            timestamp=end_time.isoformat(),
            total_time_ms=total_time_ms
        )
        
    except ValueError as e:
        logger.warning(f"Invalid request for user {request.user_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/item", response_model=List[RecommendationResponse])
async def recommend_similar_items(
    request: ItemRecommendationRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get similar item recommendations."""
    try:
        # Get similar items
        similar_items = engine.get_similar_items(
            item_id=request.item_id,
            num_items=request.num_recommendations
        )
        
        # Format response
        recommendations = []
        for i, (item_id, similarity) in enumerate(similar_items):
            metadata = None
            if request.include_metadata:
                metadata = engine.get_item_metadata(item_id)
            
            recommendations.append(RecommendationResponse(
                item_id=item_id,
                score=float(similarity),
                rank=i + 1,
                metadata=metadata
            ))
        
        return recommendations
        
    except ValueError as e:
        logger.warning(f"Invalid request for item {request.item_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting similar items for {request.item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/batch", response_model=BatchRecommendationResponse)
async def recommend_batch(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get recommendations for multiple users."""
    start_time = datetime.utcnow()
    
    try:
        # Limit batch size
        if len(request.user_ids) > 1000:
            raise HTTPException(status_code=400, detail="Batch size too large (max 1000 users)")
        
        # Get batch recommendations
        batch_recommendations = {}
        
        for user_id in request.user_ids:
            try:
                recommendations = engine.get_user_recommendations(
                    user_id=user_id,
                    num_recommendations=request.num_recommendations,
                    model_type=request.model_type,
                    exclude_seen=True
                )
                
                formatted_recommendations = []
                for i, (item_id, score) in enumerate(recommendations):
                    formatted_recommendations.append(RecommendationResponse(
                        item_id=item_id,
                        score=float(score),
                        rank=i + 1
                    ))
                
                batch_recommendations[user_id] = formatted_recommendations
                
            except Exception as e:
                logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                batch_recommendations[user_id] = []
        
        end_time = datetime.utcnow()
        total_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return BatchRecommendationResponse(
            recommendations=batch_recommendations,
            model_type=request.model_type,
            timestamp=end_time.isoformat(),
            total_time_ms=total_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/rating", response_model=RatingPredictionResponse)
async def predict_rating(
    request: RatingPredictionRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Predict rating for a user-item pair."""
    try:
        # Get rating prediction
        predicted_rating, confidence = engine.predict_rating(
            user_id=request.user_id,
            item_id=request.item_id,
            model_type=request.model_type
        )
        
        return RatingPredictionResponse(
            user_id=request.user_id,
            item_id=request.item_id,
            predicted_rating=float(predicted_rating),
            confidence=float(confidence) if confidence is not None else None,
            model_type=request.model_type
        )
        
    except ValueError as e:
        logger.warning(f"Invalid prediction request: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting rating: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommend/popular", response_model=List[RecommendationResponse])
async def get_popular_items(
    num_items: int = 20,
    category: Optional[str] = None,
    time_window: Optional[str] = "7d",
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get popular items (trending/bestsellers)."""
    try:
        # Validate parameters
        if num_items > 100:
            raise HTTPException(status_code=400, detail="Too many items requested (max 100)")
        
        # Get popular items
        popular_items = engine.get_popular_items(
            num_items=num_items,
            category=category,
            time_window=time_window
        )
        
        # Format response
        recommendations = []
        for i, (item_id, popularity_score) in enumerate(popular_items):
            metadata = engine.get_item_metadata(item_id)
            
            recommendations.append(RecommendationResponse(
                item_id=item_id,
                score=float(popularity_score),
                rank=i + 1,
                metadata=metadata
            ))
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting popular items: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(
    user_id: int,
    item_id: int,
    feedback_type: str,
    rating: Optional[float] = None,
    background_tasks: BackgroundTasks = None,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Submit user feedback for recommendations."""
    try:
        # Validate feedback type
        valid_feedback_types = ['like', 'dislike', 'rating', 'purchase', 'view', 'cart']
        if feedback_type not in valid_feedback_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid feedback type. Must be one of: {valid_feedback_types}"
            )
        
        # Process feedback
        feedback_data = {
            'user_id': user_id,
            'item_id': item_id,
            'feedback_type': feedback_type,
            'rating': rating,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store feedback (this would typically go to a database)
        if background_tasks:
            background_tasks.add_task(engine.process_feedback, feedback_data)
        
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/reload")
async def reload_models(
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Reload recommendation models (admin endpoint)."""
    try:
        # Reload models in background
        background_tasks.add_task(engine.reload_models)
        
        return {"status": "success", "message": "Model reload initiated"}
        
    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Main function
def main():
    """Run the FastAPI server."""
    # Store start time
    app.state.start_time = datetime.utcnow()
    
    # Get configuration
    config = get_config()
    
    # Run server
    uvicorn.run(
        "src.api.app:app",
        host=config.api_host,
        port=config.api_port,
        workers=config.api_workers,
        reload=config.api_reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()