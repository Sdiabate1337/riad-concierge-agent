"""
Database initialization and management for Riad Concierge AI
Handles database connections, migrations, and setup
"""

import asyncio
from typing import Optional
import redis.asyncio as redis
from loguru import logger

from app.core.config import get_settings


class DatabaseManager:
    """Database connection and management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return
            
        try:
            # Initialize Redis connection
            await self._init_redis()
            
            # Run any necessary migrations or setup
            await self._run_migrations()
            
            self._initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            # Continue without database for testing
            self._initialized = True
    
    async def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            # Continue without Redis for testing
            self.redis_client = None
    
    async def _run_migrations(self):
        """Run database migrations if needed."""
        try:
            if not self.redis_client:
                return
                
            # Check if migrations have been run
            migration_key = "migrations:version"
            current_version = await self.redis_client.get(migration_key)
            
            if not current_version:
                # Run initial setup
                await self._create_initial_schema()
                await self.redis_client.set(migration_key, "1.0.0")
                logger.info("Initial database schema created")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
    
    async def _create_initial_schema(self):
        """Create initial database schema/indexes."""
        try:
            if not self.redis_client:
                return
                
            # Create any necessary Redis indexes or initial data
            # For now, just ensure basic keys exist
            await self.redis_client.set("system:initialized", "true")
            
        except Exception as e:
            logger.error(f"Failed to create initial schema: {e}")
    
    async def get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client instance."""
        if not self._initialized:
            await self.initialize()
        return self.redis_client
    
    async def health_check(self) -> dict:
        """Perform database health check."""
        health = {
            "redis": False,
            "overall": False
        }
        
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health["redis"] = True
            
            health["overall"] = health["redis"]
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        return health
    
    async def cleanup(self):
        """Cleanup database connections."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Database connections closed")
                
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def get_database_manager() -> DatabaseManager:
    """Get database manager instance with lazy initialization."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        await _db_manager.initialize()
    return _db_manager


async def init_db():
    """Initialize database - called during application startup."""
    try:
        db_manager = await get_database_manager()
        logger.info("Database initialization completed")
        return db_manager
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Return a mock manager for testing
        return DatabaseManager()


async def get_redis_client() -> Optional[redis.Redis]:
    """Get Redis client instance."""
    try:
        db_manager = await get_database_manager()
        return await db_manager.get_redis_client()
    except Exception as e:
        logger.error(f"Failed to get Redis client: {e}")
        return None


async def close_db():
    """Close database connections - called during application shutdown."""
    global _db_manager
    if _db_manager:
        await _db_manager.cleanup()
        _db_manager = None
