"""
FastAPI application for e-commerce recommendation system.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GzipMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import redis
import json
import asyncio
from datetime import datetime, timedelta
import uvicorn
import os

from ..inference_pipeline.inference import RecommendationEngine
from ..utils.config import get_config
from ..utils.logging import setup_logging, log_execution_time

# Setup
config = get_config()
logger = setup_logging(__name__)
app = FastAPI(
    title="E-commerce Recommender API",
    description="Production-ready recommendation system with hybrid algorithms",
    version="1.0.0"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GzipMiddleware, minimum_size=1000)

# Redis client for caching
try:
    redis_client = redis.from_url(config.redis_url, decode_responses=True)
except Exception as e:
    logger.warning(f"Could not connect to Redis: {e}")
    redis_client = None

# Global recommendation engine
recommendation_engine = None


# Pydantic models
class UserRecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = Field(default=10, ge=1, le=100)
    algorithm: str = Field(default="hybrid", pattern="^(collaborative|content|hybrid)$")
    include_seen: bool = Field(default=False)
    diversity_lambda: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    categories: Optional[List[str]] = None
    min_rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)


class ItemSimilarityRequest(BaseModel):
    item_id: int
    num_similar: int = Field(default=10, ge=1, le=50)
    algorithm: str = Field(default="content", pattern="^(content|collaborative)$")


class BatchRecommendationRequest(BaseModel):
    user_ids: List[int] = Field(max_items=1000)
    num_recommendations: int = Field(default=10, ge=1, le=100)
    algorithm: str = Field(default="hybrid", pattern="^(collaborative|content|hybrid)$")


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, Union[int, float, str]]]
    algorithm: str
    timestamp: datetime
    execution_time_ms: float


class SimilarItemsResponse(BaseModel):
    item_id: int
    similar_items: List[Dict[str, Union[int, float, str]]]
    algorithm: str
    timestamp: datetime
    execution_time_ms: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    model_status: str
    cache_status: str


class UserInteractionRequest(BaseModel):
    user_id: int
    item_id: int
    interaction_type: str = Field(pattern="^(view|cart|purchase|rating)$")
    rating: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    timestamp: Optional[datetime] = None


# Dependency functions
async def get_recommendation_engine():
    """Get recommendation engine instance."""
    global recommendation_engine
    if recommendation_engine is None:
        try:
            recommendation_engine = RecommendationEngine(config)
            await recommendation_engine.load_models()
        except Exception as e:
            logger.error(f"Failed to initialize recommendation engine: {e}")
            raise HTTPException(status_code=503, detail="Recommendation engine unavailable")
    return recommendation_engine


def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key from parameters."""
    key_parts = [prefix] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
    return ":".join(str(part) for part in key_parts)


async def get_from_cache(key: str) -> Optional[Dict]:
    """Get data from cache."""
    if redis_client is None:
        return None
    
    try:
        cached_data = redis_client.get(key)
        if cached_data:
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    
    return None


async def set_cache(key: str, data: Dict, ttl: int = None):
    """Set data in cache."""
    if redis_client is None:
        return
    
    try:
        ttl = ttl or config.cache_ttl
        redis_client.setex(key, ttl, json.dumps(data, default=str))
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_status = "healthy" if recommendation_engine else "not_loaded"
    cache_status = "healthy" if redis_client else "unavailable"
    
    try:
        if redis_client:
            redis_client.ping()
    except Exception:
        cache_status = "error"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        model_status=model_status,
        cache_status=cache_status
    )


@app.post("/recommend", response_model=RecommendationResponse)
@log_execution_time("user_recommendation")
async def get_user_recommendations(
    request: UserRecommendationRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get personalized recommendations for a user."""
    start_time = datetime.utcnow()
    
    # Check cache first
    cache_key = get_cache_key(
        "user_rec",
        user_id=request.user_id,
        num=request.num_recommendations,
        algo=request.algorithm,
        seen=request.include_seen,
        div=request.diversity_lambda,
        cat=request.categories,
        min_rat=request.min_rating
    )
    
    cached_result = await get_from_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for user {request.user_id}")
        return RecommendationResponse(**cached_result)
    
    try:
        # Generate recommendations
        recommendations = await engine.get_user_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            algorithm=request.algorithm,
            include_seen=request.include_seen,
            diversity_lambda=request.diversity_lambda,
            categories=request.categories,
            min_rating=request.min_rating
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            algorithm=request.algorithm,
            timestamp=datetime.utcnow(),
            execution_time_ms=execution_time
        )
        
        # Cache the result
        await set_cache(cache_key, response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similar", response_model=SimilarItemsResponse)
@log_execution_time("item_similarity")
async def get_similar_items(
    request: ItemSimilarityRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get items similar to a given item."""
    start_time = datetime.utcnow()
    
    # Check cache first
    cache_key = get_cache_key(
        "item_sim",
        item_id=request.item_id,
        num=request.num_similar,
        algo=request.algorithm
    )
    
    cached_result = await get_from_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for item {request.item_id}")
        return SimilarItemsResponse(**cached_result)
    
    try:
        # Get similar items
        similar_items = await engine.get_similar_items(
            item_id=request.item_id,
            num_similar=request.num_similar,
            algorithm=request.algorithm
        )
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=similar_items,
            algorithm=request.algorithm,
            timestamp=datetime.utcnow(),
            execution_time_ms=execution_time
        )
        
        # Cache the result
        await set_cache(cache_key, response.dict())
        
        return response
        
    except Exception as e:
        logger.error(f"Error finding similar items for {request.item_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-recommend")
@log_execution_time("batch_recommendation")
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Generate recommendations for multiple users."""
    if len(request.user_ids) > 1000:
        raise HTTPException(status_code=400, detail="Too many users (max 1000)")
    
    try:
        # Process in background for large batches
        if len(request.user_ids) > 100:
            background_tasks.add_task(
                engine.generate_batch_recommendations,
                request.user_ids,
                request.num_recommendations,
                request.algorithm
            )
            return {"message": "Batch processing started", "user_count": len(request.user_ids)}
        
        # Process immediately for small batches
        results = []
        for user_id in request.user_ids:
            recommendations = await engine.get_user_recommendations(
                user_id=user_id,
                num_recommendations=request.num_recommendations,
                algorithm=request.algorithm
            )
            results.append({
                "user_id": user_id,
                "recommendations": recommendations
            })
        
        return {"results": results, "user_count": len(request.user_ids)}
        
    except Exception as e:
        logger.error(f"Error in batch recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interaction")
async def record_interaction(
    request: UserInteractionRequest,
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Record a user interaction for model updates."""
    try:
        await engine.record_interaction(
            user_id=request.user_id,
            item_id=request.item_id,
            interaction_type=request.interaction_type,
            rating=request.rating,
            timestamp=request.timestamp or datetime.utcnow()
        )
        
        # Invalidate relevant cache entries
        if redis_client:
            pattern = f"user_rec:user_id:{request.user_id}:*"
            try:
                for key in redis_client.scan_iter(match=pattern):
                    redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Cache invalidation error: {e}")
        
        return {"message": "Interaction recorded", "user_id": request.user_id, "item_id": request.item_id}
        
    except Exception as e:
        logger.error(f"Error recording interaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_system_stats(
    engine: RecommendationEngine = Depends(get_recommendation_engine)
):
    """Get system statistics and metrics."""
    try:
        stats = await engine.get_system_stats()
        
        # Add cache stats
        if redis_client:
            try:
                cache_info = redis_client.info("memory")
                stats["cache"] = {
                    "used_memory": cache_info.get("used_memory_human"),
                    "keyspace_hits": cache_info.get("keyspace_hits", 0),
                    "keyspace_misses": cache_info.get("keyspace_misses", 0)
                }
            except Exception:
                stats["cache"] = {"status": "error"}
        else:
            stats["cache"] = {"status": "unavailable"}
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting E-commerce Recommender API...")
    
    # Initialize recommendation engine
    try:
        global recommendation_engine
        recommendation_engine = RecommendationEngine(config)
        await recommendation_engine.load_models()
        logger.info("Recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {e}")
    
    # Test Redis connection
    if redis_client:
        try:
            redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down E-commerce Recommender API...")
    
    # Close Redis connection
    if redis_client:
        try:
            redis_client.close()
        except Exception:
            pass


def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "src.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True if os.getenv("DEBUG") else False,
        log_level=config.log_level.lower()
    )


if __name__ == "__main__":
    main()