"""
Rate Limiting Middleware for Riad Concierge AI
Handles API rate limiting, throttling, and abuse prevention
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import hashlib

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.redis_client import get_redis_client


class RateLimitConfig:
    """Rate limit configuration."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000,
        burst_limit: int = 10,
        window_size: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_limit = burst_limit
        self.window_size = window_size


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with multiple algorithms."""
    
    def __init__(
        self, 
        app, 
        default_config: Optional[RateLimitConfig] = None,
        endpoint_configs: Optional[Dict[str, RateLimitConfig]] = None
    ):
        super().__init__(app)
        self.settings = get_settings()
        self.default_config = default_config or RateLimitConfig()
        self.endpoint_configs = endpoint_configs or {}
        
        # Paths to exclude from rate limiting
        self.exclude_paths = [
            "/health",
            "/docs",
            "/redoc", 
            "/openapi.json"
        ]
        
        # Different limits for different endpoint types
        self.endpoint_configs.update({
            "/webhook/whatsapp": RateLimitConfig(
                requests_per_minute=300,  # High limit for WhatsApp webhooks
                requests_per_hour=5000,
                burst_limit=50
            ),
            "/api/chat": RateLimitConfig(
                requests_per_minute=30,   # Lower limit for chat endpoints
                requests_per_hour=500,
                burst_limit=5
            ),
            "/api/admin": RateLimitConfig(
                requests_per_minute=100,  # Higher limit for admin
                requests_per_hour=2000,
                burst_limit=20
            )
        })
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through rate limiting middleware."""
        try:
            # Skip rate limiting for excluded paths
            if any(request.url.path.startswith(path) for path in self.exclude_paths):
                return await call_next(request)
            
            # Get client identifier
            client_id = await self._get_client_identifier(request)
            
            # Get rate limit config for this endpoint
            config = self._get_endpoint_config(request.url.path)
            
            # Check rate limits
            rate_limit_result = await self._check_rate_limits(client_id, request.url.path, config)
            
            if not rate_limit_result["allowed"]:
                # Rate limit exceeded
                response = Response(
                    content=f"Rate limit exceeded. Try again in {rate_limit_result['retry_after']} seconds.",
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    headers={
                        "Retry-After": str(rate_limit_result["retry_after"]),
                        "X-RateLimit-Limit": str(config.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(rate_limit_result["reset_time"])
                    }
                )
                
                # Log rate limit violation
                await self._log_rate_limit_violation(request, client_id, rate_limit_result)
                
                return response
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            response.headers["X-RateLimit-Limit"] = str(config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(rate_limit_result["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_limit_result["reset_time"])
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}")
            # Continue without rate limiting on error
            return await call_next(request)
    
    async def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier for rate limiting."""
        # Priority order: API key > User ID > IP address
        
        # Try API key first
        api_key = request.headers.get("X-API-Key")
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header[7:]
        
        if api_key:
            return f"api_key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        
        # Try user ID from request state (if authenticated)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Fallback to IP address
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def _get_endpoint_config(self, path: str) -> RateLimitConfig:
        """Get rate limit configuration for endpoint."""
        # Check for exact matches first
        if path in self.endpoint_configs:
            return self.endpoint_configs[path]
        
        # Check for prefix matches
        for endpoint_path, config in self.endpoint_configs.items():
            if path.startswith(endpoint_path):
                return config
        
        # Return default config
        return self.default_config
    
    async def _check_rate_limits(
        self, 
        client_id: str, 
        endpoint: str, 
        config: RateLimitConfig
    ) -> Dict[str, Any]:
        """Check if request is within rate limits."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                # Allow if Redis is not available
                return {
                    "allowed": True,
                    "remaining": config.requests_per_minute,
                    "reset_time": int(time.time() + 60),
                    "retry_after": 0
                }
            
            current_time = int(time.time())
            
            # Check multiple time windows
            checks = [
                ("minute", 60, config.requests_per_minute),
                ("hour", 3600, config.requests_per_hour),
                ("day", 86400, config.requests_per_day)
            ]
            
            for window_name, window_seconds, limit in checks:
                window_start = current_time - (current_time % window_seconds)
                key = f"rate_limit:{client_id}:{endpoint}:{window_name}:{window_start}"
                
                # Get current count
                current_count = await redis_client.get(key)
                current_count = int(current_count) if current_count else 0
                
                if current_count >= limit:
                    # Rate limit exceeded
                    retry_after = window_start + window_seconds - current_time
                    return {
                        "allowed": False,
                        "remaining": 0,
                        "reset_time": window_start + window_seconds,
                        "retry_after": retry_after,
                        "window": window_name,
                        "limit": limit,
                        "current": current_count
                    }
            
            # Check burst limit (sliding window)
            burst_key = f"rate_limit_burst:{client_id}:{endpoint}"
            burst_window = current_time - config.window_size
            
            # Remove old entries
            await redis_client.zremrangebyscore(burst_key, 0, burst_window)
            
            # Count current requests in burst window
            burst_count = await redis_client.zcard(burst_key)
            
            if burst_count >= config.burst_limit:
                return {
                    "allowed": False,
                    "remaining": 0,
                    "reset_time": current_time + config.window_size,
                    "retry_after": config.window_size,
                    "window": "burst",
                    "limit": config.burst_limit,
                    "current": burst_count
                }
            
            # Increment counters
            await self._increment_counters(redis_client, client_id, endpoint, current_time, config)
            
            # Calculate remaining requests (use most restrictive)
            remaining = min(
                config.requests_per_minute - (current_count if checks[0][0] == "minute" else 0),
                config.burst_limit - burst_count
            )
            
            return {
                "allowed": True,
                "remaining": max(0, remaining),
                "reset_time": current_time + 60,  # Next minute
                "retry_after": 0
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Allow on error
            return {
                "allowed": True,
                "remaining": config.requests_per_minute,
                "reset_time": int(time.time() + 60),
                "retry_after": 0
            }
    
    async def _increment_counters(
        self, 
        redis_client, 
        client_id: str, 
        endpoint: str, 
        current_time: int,
        config: RateLimitConfig
    ):
        """Increment rate limit counters."""
        try:
            # Increment time window counters
            windows = [
                ("minute", 60),
                ("hour", 3600), 
                ("day", 86400)
            ]
            
            for window_name, window_seconds in windows:
                window_start = current_time - (current_time % window_seconds)
                key = f"rate_limit:{client_id}:{endpoint}:{window_name}:{window_start}"
                
                # Increment and set expiration
                await redis_client.incr(key)
                await redis_client.expire(key, window_seconds * 2)  # Double window for safety
            
            # Add to burst window (sorted set with timestamps)
            burst_key = f"rate_limit_burst:{client_id}:{endpoint}"
            await redis_client.zadd(burst_key, {str(current_time): current_time})
            await redis_client.expire(burst_key, config.window_size * 2)
            
        except Exception as e:
            logger.error(f"Failed to increment rate limit counters: {e}")
    
    async def _log_rate_limit_violation(
        self, 
        request: Request, 
        client_id: str, 
        rate_limit_result: Dict[str, Any]
    ):
        """Log rate limit violation."""
        try:
            violation_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "client_id": client_id,
                "endpoint": request.url.path,
                "method": request.method,
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("User-Agent", ""),
                "window": rate_limit_result.get("window", "unknown"),
                "limit": rate_limit_result.get("limit", 0),
                "current_count": rate_limit_result.get("current", 0),
                "retry_after": rate_limit_result.get("retry_after", 0)
            }
            
            logger.warning(
                f"Rate limit exceeded for {client_id} on {request.method} {request.url.path}",
                extra={"rate_limit_violation": violation_data}
            )
            
            # Store violation in Redis for monitoring
            redis_client = await get_redis_client()
            if redis_client:
                violation_key = f"rate_limit_violations:{int(time.time())}"
                await redis_client.setex(
                    violation_key,
                    86400,  # 24 hours TTL
                    str(violation_data)
                )
                
                # Increment daily violation counter
                daily_key = f"daily_violations:{datetime.utcnow().strftime('%Y-%m-%d')}"
                await redis_client.incr(daily_key)
                await redis_client.expire(daily_key, 86400 * 7)  # 7 days TTL
                
        except Exception as e:
            logger.error(f"Failed to log rate limit violation: {e}")


# Utility functions
async def get_client_rate_limit_status(client_id: str, endpoint: str = "*") -> Dict[str, Any]:
    """Get current rate limit status for a client."""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            return {}
        
        current_time = int(time.time())
        status = {}
        
        # Get counts for different windows
        windows = [("minute", 60), ("hour", 3600), ("day", 86400)]
        
        for window_name, window_seconds in windows:
            window_start = current_time - (current_time % window_seconds)
            key = f"rate_limit:{client_id}:{endpoint}:{window_name}:{window_start}"
            
            count = await redis_client.get(key)
            status[f"{window_name}_count"] = int(count) if count else 0
            status[f"{window_name}_reset"] = window_start + window_seconds
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {e}")
        return {}


async def reset_client_rate_limits(client_id: str, endpoint: str = "*"):
    """Reset rate limits for a client (admin function)."""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            return False
        
        # Find and delete all rate limit keys for this client
        pattern = f"rate_limit:{client_id}:{endpoint}:*"
        keys = await redis_client.keys(pattern)
        
        if keys:
            await redis_client.delete(*keys)
        
        # Also clear burst limits
        burst_pattern = f"rate_limit_burst:{client_id}:{endpoint}"
        burst_keys = await redis_client.keys(burst_pattern)
        
        if burst_keys:
            await redis_client.delete(*burst_keys)
        
        logger.info(f"Reset rate limits for client {client_id} on endpoint {endpoint}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to reset rate limits: {e}")
        return False


async def get_rate_limit_violations(hours: int = 24) -> list:
    """Get recent rate limit violations."""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            return []
        
        # Get violation keys from the last N hours
        start_time = int(time.time()) - (hours * 3600)
        pattern = "rate_limit_violations:*"
        keys = await redis_client.keys(pattern)
        
        violations = []
        for key in keys:
            # Extract timestamp from key
            try:
                timestamp = int(key.split(":")[-1])
                if timestamp >= start_time:
                    violation_data = await redis_client.get(key)
                    if violation_data:
                        violations.append(eval(violation_data))  # Note: Use json.loads in production
            except (ValueError, SyntaxError):
                continue
        
        # Sort by timestamp
        violations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return violations
        
    except Exception as e:
        logger.error(f"Failed to get rate limit violations: {e}")
        return []
