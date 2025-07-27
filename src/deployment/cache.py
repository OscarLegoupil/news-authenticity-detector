"""
Caching system for model predictions to improve performance.
"""

import hashlib
import time
import logging
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: Optional[float] = None


class PredictionCache:
    """LRU cache with TTL for prediction results."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum number of entries to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: Dict[str, float] = {}  # For LRU tracking
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info(f"PredictionCache initialized with max_size={max_size}, ttl={ttl_seconds}s")
    
    def get_cache_key(self, text: str, model_type: str, **kwargs) -> str:
        """
        Generate cache key for text and model type.
        
        Args:
            text: Input text
            model_type: Model type used
            **kwargs: Additional parameters affecting prediction
            
        Returns:
            Cache key string
        """
        # Create a deterministic hash of input parameters
        cache_data = {
            'text': text,
            'model_type': model_type,
            **kwargs
        }
        
        # Sort keys for consistent hashing
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()
        
        return f"{model_type}:{cache_hash[:16]}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        current_time = time.time()
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check if entry has expired
        if current_time - entry.timestamp > self.ttl_seconds:
            del self.cache[key]
            if key in self.access_order:
                del self.access_order[key]
            self.misses += 1
            return None
        
        # Update access tracking
        entry.access_count += 1
        entry.last_access = current_time
        self.access_order[key] = current_time
        self.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        current_time = time.time()
        
        # If cache is full, evict LRU entry
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        # Create cache entry
        entry = CacheEntry(
            value=value,
            timestamp=current_time,
            access_count=1,
            last_access=current_time
        )
        
        self.cache[key] = entry
        self.access_order[key] = current_time
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.access_order:
            return
        
        # Find least recently used key
        lru_key = min(self.access_order.keys(), key=self.access_order.get)
        
        # Remove from cache
        del self.cache[lru_key]
        del self.access_order[lru_key]
        self.evictions += 1
        
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if key was found and removed
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.access_order:
                del self.access_order[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if current_time - entry.timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_order:
                del self.access_order[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'ttl_seconds': self.ttl_seconds
        }
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific cache entry."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        current_time = time.time()
        
        return {
            'key': key,
            'age_seconds': current_time - entry.timestamp,
            'access_count': entry.access_count,
            'last_access': entry.last_access,
            'expires_in_seconds': self.ttl_seconds - (current_time - entry.timestamp)
        }
    
    def get_top_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cache entries by access count."""
        entries_info = []
        
        for key, entry in self.cache.items():
            entries_info.append({
                'key': key,
                'access_count': entry.access_count,
                'age_seconds': time.time() - entry.timestamp
            })
        
        # Sort by access count descending
        entries_info.sort(key=lambda x: x['access_count'], reverse=True)
        
        return entries_info[:limit]


class ModelCache:
    """Cache for storing trained models and their metadata."""
    
    def __init__(self, cache_dir: str = "cache/models"):
        """
        Initialize model cache.
        
        Args:
            cache_dir: Directory to store cached models
        """
        self.cache_dir = cache_dir
        self.ensure_cache_directory()
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ModelCache initialized with directory: {cache_dir}")
    
    def ensure_cache_directory(self):
        """Ensure cache directory exists."""
        import os
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def cache_model(self, model_id: str, model_path: str, metadata: Dict[str, Any]) -> str:
        """
        Cache a trained model.
        
        Args:
            model_id: Unique identifier for the model
            model_path: Path to the model file
            metadata: Model metadata (training date, performance, etc.)
            
        Returns:
            Path to cached model
        """
        import os
        import shutil
        
        cached_path = os.path.join(self.cache_dir, f"{model_id}")
        
        # Copy model to cache
        if os.path.isfile(model_path):
            shutil.copy2(model_path, cached_path)
        elif os.path.isdir(model_path):
            if os.path.exists(cached_path):
                shutil.rmtree(cached_path)
            shutil.copytree(model_path, cached_path)
        
        # Store metadata
        self.model_registry[model_id] = {
            'cached_path': cached_path,
            'original_path': model_path,
            'cached_timestamp': time.time(),
            **metadata
        }
        
        logger.info(f"Model {model_id} cached successfully")
        return cached_path
    
    def get_model(self, model_id: str) -> Optional[str]:
        """
        Get cached model path.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to cached model or None if not found
        """
        if model_id not in self.model_registry:
            return None
        
        cached_path = self.model_registry[model_id]['cached_path']
        
        # Check if cached file/directory exists
        import os
        if not os.path.exists(cached_path):
            # Remove from registry if file doesn't exist
            del self.model_registry[model_id]
            return None
        
        return cached_path
    
    def list_cached_models(self) -> Dict[str, Dict[str, Any]]:
        """List all cached models and their metadata."""
        return self.model_registry.copy()
    
    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model from cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model was removed
        """
        if model_id not in self.model_registry:
            return False
        
        import os
        import shutil
        
        cached_path = self.model_registry[model_id]['cached_path']
        
        # Remove file/directory
        try:
            if os.path.isfile(cached_path):
                os.remove(cached_path)
            elif os.path.isdir(cached_path):
                shutil.rmtree(cached_path)
            
            # Remove from registry
            del self.model_registry[model_id]
            logger.info(f"Model {model_id} removed from cache")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model {model_id}: {e}")
            return False
    
    def cleanup_old_models(self, max_age_hours: int = 168) -> int:  # 1 week default
        """
        Remove models older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of models removed
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        old_models = []
        for model_id, metadata in self.model_registry.items():
            age = current_time - metadata['cached_timestamp']
            if age > max_age_seconds:
                old_models.append(model_id)
        
        removed_count = 0
        for model_id in old_models:
            if self.remove_model(model_id):
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old models")
        
        return removed_count


class FeatureCache:
    """Cache for expensive feature computations."""
    
    def __init__(self, max_size: int = 5000):
        """
        Initialize feature cache.
        
        Args:
            max_size: Maximum number of feature sets to cache
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        
    def get_feature_key(self, text: str, feature_type: str) -> str:
        """Generate key for feature caching."""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
        return f"{feature_type}:{text_hash}"
    
    def get_features(self, text: str, feature_type: str) -> Optional[Any]:
        """Get cached features."""
        key = self.get_feature_key(text, feature_type)
        return self.cache.get(key)
    
    def set_features(self, text: str, feature_type: str, features: Any) -> None:
        """Cache computed features."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_feature_key(text, feature_type)
        self.cache[key] = features
    
    def clear(self) -> None:
        """Clear feature cache."""
        self.cache.clear()