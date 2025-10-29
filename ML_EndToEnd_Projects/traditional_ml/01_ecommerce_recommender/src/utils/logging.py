"""
Logging utilities for the e-commerce recommender system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from rich.logging import RichHandler
from rich.console import Console


def setup_logging(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[Path] = None,
    use_structured: bool = True,
    use_rich: bool = True
) -> logging.Logger:
    """
    Set up logging configuration with structured logging and rich formatting.
    
    Args:
        name: Logger name (defaults to calling module)
        level: Logging level
        log_file: Optional log file path
        use_structured: Whether to use structured logging
        use_rich: Whether to use rich formatting for console output
    
    Returns:
        Configured logger instance
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if use_structured:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        logger = structlog.get_logger(name)
    else:
        # Standard logging setup
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        if use_rich:
            console = Console()
            console_handler = RichHandler(
                console=console,
                show_time=True,
                show_path=True,
                markup=True,
                rich_tracebacks=True
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the standard configuration."""
    return setup_logging(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


# Context managers for logging
class LogExecutionTime:
    """Context manager to log execution time of a block of code."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = "INFO"):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.log(
            getattr(logging, self.level),
            f"Starting {self.operation}..."
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(
                getattr(logging, self.level),
                f"Completed {self.operation} in {execution_time:.2f} seconds"
            )
        else:
            self.logger.error(
                f"Failed {self.operation} after {execution_time:.2f} seconds: {exc_val}"
            )


# Decorators for logging
def log_execution_time(operation: str = None, level: str = "INFO"):
    """Decorator to log execution time of a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            operation_name = operation or f"{func.__module__}.{func.__name__}"
            
            # Try to get logger from self if it's a method
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            else:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            logger.log(
                getattr(logging, level),
                f"Starting {operation_name}..."
            )
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log(
                    getattr(logging, level),
                    f"Completed {operation_name} in {execution_time:.2f} seconds"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Failed {operation_name} after {execution_time:.2f} seconds: {e}"
                )
                raise
        
        return wrapper
    return decorator


def log_errors(logger: Optional[logging.Logger] = None):
    """Decorator to log exceptions from a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Try to get logger from self if it's a method
            if logger is not None:
                log = logger
            elif args and hasattr(args[0], 'logger'):
                log = args[0].logger
            else:
                log = get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log.error(f"Error in {func.__name__}: {e}", exc_info=True)
                raise
        
        return wrapper
    return decorator