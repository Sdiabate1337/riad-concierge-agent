"""
Logging Middleware for Riad Concierge AI
Handles request/response logging, performance monitoring, and audit trails
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.redis_client import get_redis_client


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging and monitoring middleware."""
    
    def __init__(self, app, log_requests: bool = True, log_responses: bool = True):
        super().__init__(app)
        self.settings = get_settings()
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.exclude_paths = [
            "/health",
            "/metrics", 
            "/docs",
            "/redoc",
            "/openapi.json"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process request through logging middleware."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Skip logging for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request, request_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 3))
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, process_time, request_id)
            
            # Store performance metrics
            await self._store_performance_metrics(request, response, process_time)
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            await self._log_error(request, e, process_time, request_id)
            raise
    
    async def _log_request(self, request: Request, request_id: str):
        """Log incoming request details."""
        try:
            # Get client info
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            
            # Prepare request data
            request_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "headers": self._sanitize_headers(dict(request.headers)),
                "content_type": request.headers.get("Content-Type", ""),
                "content_length": request.headers.get("Content-Length", "0")
            }
            
            # Log request body for POST/PUT requests (sanitized)
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        # Try to parse as JSON
                        try:
                            body_json = json.loads(body.decode())
                            request_data["body"] = self._sanitize_body(body_json)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            request_data["body_size"] = len(body)
                except Exception:
                    pass
            
            # Log structured request
            logger.info(
                f"REQUEST {request.method} {request.url.path}",
                extra={"request_data": request_data}
            )
            
            # Store in Redis for analytics
            await self._store_request_log(request_data)
            
        except Exception as e:
            logger.error(f"Request logging failed: {e}")
    
    async def _log_response(self, request: Request, response: Response, process_time: float, request_id: str):
        """Log response details."""
        try:
            response_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "status_code": response.status_code,
                "process_time": round(process_time, 3),
                "response_headers": self._sanitize_headers(dict(response.headers)),
                "content_type": response.headers.get("Content-Type", ""),
                "content_length": response.headers.get("Content-Length", "0")
            }
            
            # Determine log level based on status code
            if response.status_code >= 500:
                log_level = "error"
            elif response.status_code >= 400:
                log_level = "warning"
            else:
                log_level = "info"
            
            # Log structured response
            getattr(logger, log_level)(
                f"RESPONSE {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s",
                extra={"response_data": response_data}
            )
            
            # Store in Redis for analytics
            await self._store_response_log(response_data)
            
        except Exception as e:
            logger.error(f"Response logging failed: {e}")
    
    async def _log_error(self, request: Request, error: Exception, process_time: float, request_id: str):
        """Log error details."""
        try:
            error_data = {
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "process_time": round(process_time, 3),
                "client_ip": self._get_client_ip(request)
            }
            
            logger.error(
                f"ERROR {request.method} {request.url.path} - {type(error).__name__}: {error}",
                extra={"error_data": error_data}
            )
            
            # Store error in Redis for monitoring
            await self._store_error_log(error_data)
            
        except Exception as e:
            logger.error(f"Error logging failed: {e}")
    
    async def _store_request_log(self, request_data: Dict[str, Any]):
        """Store request log in Redis."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return
            
            # Store with TTL
            log_key = f"request_log:{request_data['request_id']}"
            await redis_client.setex(
                log_key,
                86400,  # 24 hours TTL
                json.dumps(request_data, default=str)
            )
            
            # Add to daily request count
            date_key = f"daily_requests:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await redis_client.incr(date_key)
            await redis_client.expire(date_key, 86400 * 7)  # 7 days TTL
            
        except Exception as e:
            logger.error(f"Failed to store request log: {e}")
    
    async def _store_response_log(self, response_data: Dict[str, Any]):
        """Store response log in Redis."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return
            
            # Store with TTL
            log_key = f"response_log:{response_data['request_id']}"
            await redis_client.setex(
                log_key,
                86400,  # 24 hours TTL
                json.dumps(response_data, default=str)
            )
            
            # Update status code counters
            status_key = f"status_codes:{datetime.utcnow().strftime('%Y-%m-%d')}:{response_data['status_code']}"
            await redis_client.incr(status_key)
            await redis_client.expire(status_key, 86400 * 7)  # 7 days TTL
            
        except Exception as e:
            logger.error(f"Failed to store response log: {e}")
    
    async def _store_error_log(self, error_data: Dict[str, Any]):
        """Store error log in Redis."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return
            
            # Store error details
            error_key = f"error_log:{error_data['request_id']}"
            await redis_client.setex(
                error_key,
                86400 * 7,  # 7 days TTL
                json.dumps(error_data, default=str)
            )
            
            # Update error counters
            error_count_key = f"errors:{datetime.utcnow().strftime('%Y-%m-%d')}:{error_data['error_type']}"
            await redis_client.incr(error_count_key)
            await redis_client.expire(error_count_key, 86400 * 7)  # 7 days TTL
            
        except Exception as e:
            logger.error(f"Failed to store error log: {e}")
    
    async def _store_performance_metrics(self, request: Request, response: Response, process_time: float):
        """Store performance metrics."""
        try:
            redis_client = await get_redis_client()
            if not redis_client:
                return
            
            # Store response time metrics
            endpoint = f"{request.method}:{request.url.path}"
            
            # Add to response time list (for calculating averages)
            time_key = f"response_times:{endpoint}:{datetime.utcnow().strftime('%Y-%m-%d-%H')}"
            await redis_client.lpush(time_key, str(process_time))
            await redis_client.ltrim(time_key, 0, 999)  # Keep last 1000 entries
            await redis_client.expire(time_key, 86400)  # 24 hours TTL
            
            # Update endpoint counters
            endpoint_key = f"endpoint_calls:{endpoint}:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await redis_client.incr(endpoint_key)
            await redis_client.expire(endpoint_key, 86400 * 7)  # 7 days TTL
            
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
    
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
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers by removing sensitive information."""
        sensitive_headers = {
            "authorization", "x-api-key", "cookie", "x-auth-token",
            "x-access-token", "x-refresh-token"
        }
        
        sanitized = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_body(self, body: Any) -> Any:
        """Sanitize request body by removing sensitive information."""
        if isinstance(body, dict):
            sensitive_fields = {
                "password", "token", "api_key", "secret", "auth",
                "authorization", "credential", "key"
            }
            
            sanitized = {}
            for key, value in body.items():
                if any(sensitive in key.lower() for sensitive in sensitive_fields):
                    sanitized[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    sanitized[key] = self._sanitize_body(value)
                elif isinstance(value, list):
                    sanitized[key] = [self._sanitize_body(item) for item in value]
                else:
                    sanitized[key] = value
            
            return sanitized
        
        return body


# Utility functions for log analysis
async def get_request_logs(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 100
) -> list:
    """Get request logs from Redis."""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            return []
        
        # Get all request log keys
        keys = await redis_client.keys("request_log:*")
        
        # Fetch logs
        logs = []
        for key in keys[:limit]:
            log_data = await redis_client.get(key)
            if log_data:
                try:
                    log_entry = json.loads(log_data)
                    logs.append(log_entry)
                except json.JSONDecodeError:
                    continue
        
        # Sort by timestamp
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return logs[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get request logs: {e}")
        return []


async def get_performance_stats(endpoint: str, hours: int = 24) -> Dict[str, Any]:
    """Get performance statistics for an endpoint."""
    try:
        redis_client = await get_redis_client()
        if not redis_client:
            return {}
        
        stats = {
            "endpoint": endpoint,
            "total_requests": 0,
            "avg_response_time": 0.0,
            "min_response_time": 0.0,
            "max_response_time": 0.0,
            "error_rate": 0.0
        }
        
        # Get response times for the last N hours
        response_times = []
        for hour in range(hours):
            time_key = f"response_times:{endpoint}:{(datetime.utcnow() - timedelta(hours=hour)).strftime('%Y-%m-%d-%H')}"
            times = await redis_client.lrange(time_key, 0, -1)
            response_times.extend([float(t) for t in times])
        
        if response_times:
            stats["total_requests"] = len(response_times)
            stats["avg_response_time"] = sum(response_times) / len(response_times)
            stats["min_response_time"] = min(response_times)
            stats["max_response_time"] = max(response_times)
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {}
