"""
Redis Client Management for Riad Concierge AI
Handles Redis connection, configuration, and utilities
"""

import asyncio
from typing import Optional, Any, Dict, List
import json
from datetime import datetime, timedelta

import redis.asyncio as redis
from loguru import logger

from app.core.config import get_settings


class RedisManager:
    """Redis connection and utility manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[redis.Redis] = None
        self._initialized = False
        self._connection_pool: Optional[redis.ConnectionPool] = None
    
    async def initialize(self):
        """Initialize Redis connection."""
        if self._initialized:
            return self.client
            
        try:
            # Create connection pool
            self._connection_pool = redis.ConnectionPool.from_url(
                self.settings.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                max_connections=20
            )
            
            # Create Redis client
            self.client = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self.client.ping()
            
            self._initialized = True
            logger.info("Redis client initialized successfully")
            
            return self.client
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
            # Continue without Redis for testing
            self.client = None
            self._initialized = True
            return None
    
    async def get_client(self) -> Optional[redis.Redis]:
        """Get Redis client instance."""
        if not self._initialized:
            await self.initialize()
        return self.client
    
    async def set_json(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set JSON value in Redis."""
        try:
            if not self.client:
                return False
                
            json_value = json.dumps(value, default=str)
            
            if ttl:
                await self.client.setex(key, ttl, json_value)
            else:
                await self.client.set(key, json_value)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to set JSON in Redis: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis."""
        try:
            if not self.client:
                return None
                
            value = await self.client.get(key)
            if value:
                return json.loads(value)
                
        except Exception as e:
            logger.error(f"Failed to get JSON from Redis: {e}")
        
        return None
    
    async def set_with_ttl(self, key: str, value: str, ttl: int):
        """Set value with TTL."""
        try:
            if not self.client:
                return False
                
            await self.client.setex(key, ttl, value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set value with TTL: {e}")
            return False
    
    async def get_keys_pattern(self, pattern: str) -> List[str]:
        """Get keys matching pattern."""
        try:
            if not self.client:
                return []
                
            keys = await self.client.keys(pattern)
            return keys
            
        except Exception as e:
            logger.error(f"Failed to get keys with pattern: {e}")
            return []
    
    async def delete_keys(self, keys: List[str]) -> int:
        """Delete multiple keys."""
        try:
            if not self.client or not keys:
                return 0
                
            return await self.client.delete(*keys)
            
        except Exception as e:
            logger.error(f"Failed to delete keys: {e}")
            return 0
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter."""
        try:
            if not self.client:
                return None
                
            return await self.client.incrby(key, amount)
            
        except Exception as e:
            logger.error(f"Failed to increment counter: {e}")
            return None
    
    async def add_to_set(self, key: str, *values) -> int:
        """Add values to set."""
        try:
            if not self.client:
                return 0
                
            return await self.client.sadd(key, *values)
            
        except Exception as e:
            logger.error(f"Failed to add to set: {e}")
            return 0
    
    async def get_set_members(self, key: str) -> List[str]:
        """Get set members."""
        try:
            if not self.client:
                return []
                
            members = await self.client.smembers(key)
            return list(members)
            
        except Exception as e:
            logger.error(f"Failed to get set members: {e}")
            return []
    
    async def push_to_list(self, key: str, *values) -> int:
        """Push values to list."""
        try:
            if not self.client:
                return 0
                
            return await self.client.lpush(key, *values)
            
        except Exception as e:
            logger.error(f"Failed to push to list: {e}")
            return 0
    
    async def pop_from_list(self, key: str, timeout: int = 0) -> Optional[str]:
        """Pop value from list."""
        try:
            if not self.client:
                return None
                
            if timeout > 0:
                result = await self.client.brpop(key, timeout=timeout)
                return result[1] if result else None
            else:
                return await self.client.rpop(key)
                
        except Exception as e:
            logger.error(f"Failed to pop from list: {e}")
            return None
    
    async def get_list_length(self, key: str) -> int:
        """Get list length."""
        try:
            if not self.client:
                return 0
                
            return await self.client.llen(key)
            
        except Exception as e:
            logger.error(f"Failed to get list length: {e}")
            return 0
    
    async def set_hash_field(self, key: str, field: str, value: str):
        """Set hash field."""
        try:
            if not self.client:
                return False
                
            await self.client.hset(key, field, value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set hash field: {e}")
            return False
    
    async def get_hash_field(self, key: str, field: str) -> Optional[str]:
        """Get hash field."""
        try:
            if not self.client:
                return None
                
            return await self.client.hget(key, field)
            
        except Exception as e:
            logger.error(f"Failed to get hash field: {e}")
            return None
    
    async def get_hash_all(self, key: str) -> Dict[str, str]:
        """Get all hash fields."""
        try:
            if not self.client:
                return {}
                
            return await self.client.hgetall(key)
            
        except Exception as e:
            logger.error(f"Failed to get hash: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            if not self.client:
                return False
                
            await self.client.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup Redis connections."""
        try:
            if self.client:
                await self.client.close()
                
            if self._connection_pool:
                await self._connection_pool.disconnect()
                
            logger.info("Redis connections closed")
            
        except Exception as e:
            logger.error(f"Error during Redis cleanup: {e}")


# Global Redis manager instance
_redis_manager: Optional[RedisManager] = None


async def get_redis_manager() -> RedisManager:
    """Get Redis manager instance with lazy initialization."""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
        await _redis_manager.initialize()
    return _redis_manager


async def init_redis() -> Optional[redis.Redis]:
    """Initialize Redis - called during application startup."""
    try:
        redis_manager = await get_redis_manager()
        client = await redis_manager.get_client()
        logger.info("Redis initialization completed")
        return client
        
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        return None


async def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client instance."""
    try:
        redis_manager = await get_redis_manager()
        return await redis_manager.get_client()
    except Exception as e:
        logger.error(f"Failed to get Redis client: {e}")
        return None


async def close_redis():
    """Close Redis connections - called during application shutdown."""
    global _redis_manager
    if _redis_manager:
        await _redis_manager.cleanup()
        _redis_manager = None
