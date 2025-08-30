"""
Authentication Middleware for Riad Concierge AI
Handles authentication, authorization, and security for API endpoints
"""

import asyncio
from typing import Optional, Dict, Any, Callable
import hmac
import hashlib
import time
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger

from app.core.config import get_settings
from app.core.redis_client import get_redis_client


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication and security middleware."""
    
    def __init__(self, app, skip_auth_paths: Optional[list] = None):
        super().__init__(app)
        self.settings = get_settings()
        self.skip_auth_paths = skip_auth_paths or [
            "/health",
            "/docs", 
            "/redoc",
            "/openapi.json",
            "/webhook/whatsapp"  # WhatsApp webhook uses signature verification
        ]
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_max_requests = 100
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through authentication middleware."""
        start_time = time.time()
        
        try:
            # Skip authentication for certain paths
            if any(request.url.path.startswith(path) for path in self.skip_auth_paths):
                response = await call_next(request)
                return self._add_security_headers(response)
            
            # Rate limiting
            if not await self._check_rate_limit(request):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # WhatsApp webhook signature verification
            if request.url.path.startswith("/webhook/whatsapp"):
                if not await self._verify_whatsapp_signature(request):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid webhook signature"
                    )
            
            # API key authentication for other endpoints
            elif not await self._verify_api_key(request):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API key"
                )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response = self._add_security_headers(response)
            
            # Log request
            process_time = time.time() - start_time
            await self._log_request(request, response, process_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Auth middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
    
    async def _verify_api_key(self, request: Request) -> bool:
        """Verify API key authentication."""
        try:
            # Get API key from header
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                # Try Authorization header
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    api_key = auth_header[7:]
            
            if not api_key:
                return False
            
            # Verify against configured API key
            # In production, this would check against a database of valid keys
            valid_keys = [
                self.settings.api_key,
                self.settings.whatsapp_access_token  # Allow WhatsApp token for API access
            ]
            
            return api_key in valid_keys
            
        except Exception as e:
            logger.error(f"API key verification failed: {e}")
            return False
    
    async def _verify_whatsapp_signature(self, request: Request) -> bool:
        """Verify WhatsApp webhook signature."""
        try:
            # Get signature from header
            signature = request.headers.get("X-Hub-Signature-256")
            if not signature:
                return False
            
            # Get request body
            body = await request.body()
            
            # Calculate expected signature
            expected_signature = hmac.new(
                self.settings.whatsapp_app_secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            
            expected_signature = f"sha256={expected_signature}"
            
            # Compare signatures
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"WhatsApp signature verification failed: {e}")
            return False
    
    async def _check_rate_limit(self, request: Request) -> bool:
        """Check rate limiting for client."""
        try:
            # Get client identifier
            client_ip = self._get_client_ip(request)
            api_key = request.headers.get("X-API-Key", "anonymous")
            
            # Use API key if available, otherwise IP
            client_id = api_key if api_key != "anonymous" else client_ip
            
            redis_client = await get_redis_client()
            if not redis_client:
                # Allow if Redis is not available
                return True
            
            # Rate limit key
            rate_limit_key = f"rate_limit:{client_id}:{int(time.time() // self.rate_limit_window)}"
            
            # Increment counter
            current_requests = await redis_client.incr(rate_limit_key)
            
            # Set expiration on first request
            if current_requests == 1:
                await redis_client.expire(rate_limit_key, self.rate_limit_window)
            
            return current_requests <= self.rate_limit_max_requests
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Allow on error
            return True
    
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
    
    def _add_security_headers(self, response: Response) -> Response:
        """Add security headers to response."""
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        
        return response
    
    async def _log_request(self, request: Request, response: Response, process_time: float):
        """Log request details."""
        try:
            client_ip = self._get_client_ip(request)
            
            log_data = {
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_ip,
                "status_code": response.status_code,
                "process_time": round(process_time, 3),
                "user_agent": request.headers.get("User-Agent", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Log to structured logger
            logger.info(
                f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s",
                extra=log_data
            )
            
            # Store in Redis for analytics (optional)
            redis_client = await get_redis_client()
            if redis_client:
                log_key = f"request_log:{int(time.time())}"
                await redis_client.setex(log_key, 86400, str(log_data))  # 24h TTL
                
        except Exception as e:
            logger.error(f"Request logging failed: {e}")


class APIKeyAuth(HTTPBearer):
    """API Key authentication scheme."""
    
    def __init__(self):
        super().__init__(auto_error=False)
        self.settings = get_settings()
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        """Authenticate request with API key."""
        try:
            # Try Bearer token first
            credentials = await super().__call__(request)
            if credentials:
                return credentials
            
            # Try X-API-Key header
            api_key = request.headers.get("X-API-Key")
            if api_key:
                return HTTPAuthorizationCredentials(
                    scheme="Bearer",
                    credentials=api_key
                )
            
            return None
            
        except Exception as e:
            logger.error(f"API key authentication failed: {e}")
            return None


# Security utilities
def verify_api_key(api_key: str) -> bool:
    """Verify if API key is valid."""
    try:
        settings = get_settings()
        valid_keys = [
            settings.api_key,
            settings.whatsapp_access_token
        ]
        return api_key in valid_keys
    except Exception as e:
        logger.error(f"API key verification failed: {e}")
        return False


def create_api_key_hash(api_key: str) -> str:
    """Create hash of API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_current_user(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
    """Get current user from credentials."""
    try:
        if not verify_api_key(credentials.credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        # Return user info based on API key
        # In production, this would fetch from database
        return {
            "api_key": credentials.credentials,
            "permissions": ["read", "write"],
            "authenticated": True
        }
        
    except Exception as e:
        logger.error(f"Get current user failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )
