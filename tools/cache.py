"""
Caching utilities for financial data.
"""

import functools
import os
import pickle
import time
from typing import Any, Callable
from loguru import logger

CACHE_DIR = "data/cache"

def ensure_cache_dir():
    """Ensure the cache directory exists."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def disk_cache(expiry_seconds: int = 3600):
    """
    Simple disk-based cache decorator.
    
    Args:
        expiry_seconds: Cache expiry time in seconds (default 1 hour)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ensure_cache_dir()
            
            # Create a simple cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}.pkl"
            cache_path = os.path.join(CACHE_DIR, cache_key)
            
            # Check if cache exists and is not expired
            if os.path.exists(cache_path):
                mtime = os.path.getmtime(cache_path)
                if (time.time() - mtime) < expiry_seconds:
                    try:
                        with open(cache_path, "rb") as f:
                            logger.debug(f"Cache hit for {func.__name__}")
                            return pickle.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading cache for {func.__name__}: {e}")
            
            # Call the actual function
            result = func(*args, **kwargs)
            
            # Save to cache
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                    logger.debug(f"Cache saved for {func.__name__}")
            except Exception as e:
                logger.warning(f"Error saving cache for {func.__name__}: {e}")
                
            return result
        return wrapper
    return decorator
