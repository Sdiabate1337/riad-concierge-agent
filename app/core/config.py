"""
Application configuration management using Pydantic settings.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = Field(default="Riad Concierge AI", env="APP_NAME")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=4, env="WORKERS")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["*"], env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    
    # WhatsApp Business API
    whatsapp_phone_number_id: str = Field(..., env="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_access_token: str = Field(..., env="WHATSAPP_ACCESS_TOKEN")
    whatsapp_webhook_verify_token: str = Field(..., env="WHATSAPP_WEBHOOK_VERIFY_TOKEN")
    whatsapp_app_secret: str = Field(..., env="WHATSAPP_APP_SECRET")
    whatsapp_api_version: str = Field(default="v18.0", env="WHATSAPP_API_VERSION")
    whatsapp_webhook_url: str = Field(..., env="WHATSAPP_WEBHOOK_URL")
    
    # AI Services
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.3, env="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=2000, env="OPENAI_MAX_TOKENS")
    openai_embedding_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    
    # Pinecone Vector Database
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="riad-concierge-knowledge", env="PINECONE_INDEX_NAME")
    
    # PMS Integration (Octorate)
    octorate_api_key: str = Field(..., env="OCTORATE_API_KEY")
    octorate_property_id: str = Field(..., env="OCTORATE_PROPERTY_ID")
    octorate_api_url: str = Field(default="https://api.octorate.com/v1", env="OCTORATE_API_URL")
    pms_sync_interval: int = Field(default=300, env="PMS_SYNC_INTERVAL")
    
    # External APIs
    openweather_api_key: str = Field(..., env="OPENWEATHER_API_KEY")
    default_city: str = Field(default="Marrakech", env="DEFAULT_CITY")
    google_maps_api_key: str = Field(..., env="GOOGLE_MAPS_API_KEY")
    google_translate_api_key: Optional[str] = Field(default=None, env="GOOGLE_TRANSLATE_API_KEY")
    
    # Cultural Settings
    default_language: str = Field(default="en", env="DEFAULT_LANGUAGE")
    supported_languages: List[str] = Field(default=["ar", "fr", "en", "es"], env="SUPPORTED_LANGUAGES")
    timezone: str = Field(default="Africa/Casablanca", env="TIMEZONE")
    
    # Revenue Optimization
    upselling_enabled: bool = Field(default=True, env="UPSELLING_ENABLED")
    upselling_success_threshold: float = Field(default=0.7, env="UPSELLING_SUCCESS_THRESHOLD")
    upselling_timing_optimization: bool = Field(default=True, env="UPSELLING_TIMING_OPTIMIZATION")
    direct_booking_tracking: bool = Field(default=True, env="DIRECT_BOOKING_TRACKING")
    loyalty_program_enabled: bool = Field(default=True, env="LOYALTY_PROGRAM_ENABLED")
    commission_savings_share: float = Field(default=0.3, env="COMMISSION_SAVINGS_SHARE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/app.log", env="LOG_FILE")
    
    # Performance
    max_concurrent_messages: int = Field(default=50, env="MAX_CONCURRENT_MESSAGES")
    message_processing_timeout: int = Field(default=30, env="MESSAGE_PROCESSING_TIMEOUT")
    vector_search_timeout: int = Field(default=5, env="VECTOR_SEARCH_TIMEOUT")
    
    # Rate Limiting
    rate_limit_window: int = Field(default=900, env="RATE_LIMIT_WINDOW")  # 15 minutes
    rate_limit_max_requests: int = Field(default=100, env="RATE_LIMIT_MAX_REQUESTS")
    
    @validator("supported_languages", pre=True)
    def parse_supported_languages(cls, v):
        """Parse comma-separated languages from environment variable."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    @validator("allowed_hosts", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse comma-separated hosts from environment variable."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse comma-separated origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value."""
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
