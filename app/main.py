"""
Riad Concierge AI - Main Application Entry Point
Production-ready WhatsApp AI agent for Moroccan riad hospitality
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger
from prometheus_client import make_asgi_app

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.core.database import init_db
from app.core.redis_client import init_redis
from app.middleware.auth import AuthMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.services.agent_service import AgentService
from app.services.proactive_service import ProactiveService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    settings = get_settings()
    
    # Initialize core services
    logger.info("🚀 Starting Riad Concierge AI Agent...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("✅ Database initialized")
        
        # Initialize Redis
        await init_redis()
        logger.info("✅ Redis initialized")
        
        # Initialize agent service
        agent_service = AgentService()
        await agent_service.initialize()
        app.state.agent_service = agent_service
        logger.info("✅ Agent service initialized")
        
        # Initialize proactive service
        proactive_service = ProactiveService(agent_service)
        await proactive_service.start()
        app.state.proactive_service = proactive_service
        logger.info("✅ Proactive service started")
        
        logger.info("🎉 Riad Concierge AI Agent started successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("🛑 Shutting down Riad Concierge AI Agent...")
        
        if hasattr(app.state, 'proactive_service'):
            await app.state.proactive_service.stop()
            logger.info("✅ Proactive service stopped")
        
        if hasattr(app.state, 'agent_service'):
            await app.state.agent_service.cleanup()
            logger.info("✅ Agent service cleaned up")
        
        logger.info("👋 Riad Concierge AI Agent stopped")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Riad Concierge AI",
        description="Production-ready WhatsApp AI agent for Moroccan riad hospitality",
        version="1.0.0",
        docs_url="/docs" if settings.environment == "development" else None,
        redoc_url="/redoc" if settings.environment == "development" else None,
        lifespan=lifespan,
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(AuthMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint for load balancers."""
        return {
            "status": "healthy",
            "service": "riad-concierge-ai",
            "version": "1.0.0",
            "environment": settings.environment,
        }
    
    # Metrics endpoint for Prometheus
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    
    return app


# Create the FastAPI app instance
app = create_app()


def main() -> None:
    """Main entry point for the application."""
    settings = get_settings()
    
    # Configure logging
    logger.remove()
    logger.add(
        sink=settings.log_file,
        level=settings.log_level.upper(),
        rotation="10 MB",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        serialize=settings.environment == "production",
    )
    
    if settings.environment == "development":
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level="DEBUG",
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        )
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        workers=1 if settings.environment == "development" else settings.workers,
        log_config=None,  # Use loguru instead
        access_log=False,  # Use custom logging middleware
    )


if __name__ == "__main__":
    main()
