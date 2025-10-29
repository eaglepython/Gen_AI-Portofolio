"""
Configuration management for the e-commerce recommender system.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for the recommender system."""
    
    # Data paths
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    # Data source configuration
    data_source: str = "synthetic"  # 'synthetic' or 'movielens'
    dataset_size: str = "25m"  # For MovieLens: '100k', '1m', '25m'
    
    # Model configuration
    collaborative_model: str = "svd"  # 'svd', 'nmf', 'ncf'
    content_model: str = "tfidf"  # 'tfidf', 'bert'
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        "collaborative": 0.6,
        "content": 0.3,
        "popularity": 0.1
    })
    
    # Training parameters
    n_factors: int = 100
    n_epochs: int = 20
    learning_rate: float = 0.01
    reg_all: float = 0.02
    batch_size: int = 256
    
    # Recommendation parameters
    n_recommendations: int = 10
    min_rating: float = 3.0
    diversity_lambda: float = 0.1
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour
    
    # MLflow configuration
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "ecommerce_recommender"
    
    # AWS configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-west-2"
    s3_bucket: str = "ecommerce-recommender-data"
    
    # Database configuration
    database_url: str = "postgresql://user:password@localhost:5432/recommender"
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Load environment variables
        self._load_from_env()
        
        # Load from config file if exists
        config_file = Path("configs/config.yaml")
        if config_file.exists():
            self._load_from_file(config_file)
        
        # Create directories
        self._create_directories()
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mapping = {
            "DATA_DIR": "data_dir",
            "MODEL_DIR": "model_dir",
            "LOG_DIR": "log_dir",
            "DATA_SOURCE": "data_source",
            "DATASET_SIZE": "dataset_size",
            "API_HOST": "api_host",
            "API_PORT": "api_port",
            "REDIS_URL": "redis_url",
            "MLFLOW_TRACKING_URI": "mlflow_tracking_uri",
            "AWS_ACCESS_KEY_ID": "aws_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
            "AWS_REGION": "aws_region",
            "S3_BUCKET": "s3_bucket",
            "DATABASE_URL": "database_url",
            "LOG_LEVEL": "log_level",
        }
        
        for env_var, attr_name in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert to appropriate type
                if attr_name == "api_port":
                    env_value = int(env_value)
                elif attr_name == "cache_ttl":
                    env_value = int(env_value)
                
                setattr(self, attr_name, env_value)
    
    def _load_from_file(self, config_file: Path):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir,
            f"{self.data_dir}/raw",
            f"{self.data_dir}/processed",
            f"{self.data_dir}/features",
            self.model_dir,
            f"{self.model_dir}/collaborative",
            f"{self.model_dir}/content_based",
            f"{self.model_dir}/hybrid",
            f"{self.model_dir}/encoders",
            self.log_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return getattr(self, key, default)
    
    def update(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    def save(self, file_path: Path):
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global configuration instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config