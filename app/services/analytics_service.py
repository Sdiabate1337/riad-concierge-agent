"""
Analytics Service for Riad Concierge AI
Handles metrics collection, performance tracking, and business intelligence
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import redis.asyncio as redis
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings


class MetricType(str):
    """Metric type constants."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class Metric(BaseModel):
    """Metric data model."""
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    metric_type: str = Field(MetricType.COUNTER, description="Type of metric")
    tags: Dict[str, str] = Field(default_factory=dict, description="Metric tags")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric timestamp")


class ConversationMetrics(BaseModel):
    """Conversation-level metrics."""
    conversation_id: str
    guest_phone: str
    message_count: int = 0
    response_time_avg: float = 0.0
    cultural_accuracy_score: float = 0.0
    satisfaction_score: Optional[float] = None
    revenue_generated: float = 0.0
    upsells_successful: int = 0
    escalations: int = 0
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None


class AnalyticsService:
    """Analytics and metrics collection service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        self.metrics_buffer: List[Metric] = []
        self.buffer_size = 100
        self.flush_interval = 60  # seconds
        
    async def initialize(self):
        """Initialize the analytics service."""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                retry_on_timeout=True
            )
            await self.redis_client.ping()
            logger.info("Analytics service initialized successfully")
            
            # Start background tasks
            asyncio.create_task(self._flush_metrics_periodically())
            
        except Exception as e:
            logger.error(f"Failed to initialize analytics service: {e}")
            # Continue without Redis for testing
            self.redis_client = None
    
    async def record_metric(
        self, 
        name: str, 
        value: float, 
        metric_type: str = MetricType.COUNTER,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a metric."""
        try:
            metric = Metric(
                name=name,
                value=value,
                metric_type=metric_type,
                tags=tags or {},
                timestamp=datetime.utcnow()
            )
            
            self.metrics_buffer.append(metric)
            
            # Flush if buffer is full
            if len(self.metrics_buffer) >= self.buffer_size:
                await self._flush_metrics()
                
        except Exception as e:
            logger.error(f"Failed to record metric {name}: {e}")
    
    async def record_conversation_start(self, conversation_id: str, guest_phone: str):
        """Record the start of a conversation."""
        try:
            metrics = ConversationMetrics(
                conversation_id=conversation_id,
                guest_phone=guest_phone
            )
            
            if self.redis_client:
                await self.redis_client.setex(
                    f"conversation_metrics:{conversation_id}",
                    3600,  # 1 hour TTL
                    metrics.model_dump_json()
                )
            
            await self.record_metric("conversations_started", 1.0)
            
        except Exception as e:
            logger.error(f"Failed to record conversation start: {e}")
    
    async def record_message_processed(
        self, 
        conversation_id: str, 
        response_time: float,
        cultural_accuracy: float = 0.0
    ):
        """Record a processed message."""
        try:
            # Update conversation metrics
            if self.redis_client:
                key = f"conversation_metrics:{conversation_id}"
                data = await self.redis_client.get(key)
                
                if data:
                    metrics = ConversationMetrics.model_validate_json(data)
                    metrics.message_count += 1
                    
                    # Update running average
                    total_time = metrics.response_time_avg * (metrics.message_count - 1)
                    metrics.response_time_avg = (total_time + response_time) / metrics.message_count
                    
                    # Update cultural accuracy
                    if cultural_accuracy > 0:
                        total_accuracy = metrics.cultural_accuracy_score * (metrics.message_count - 1)
                        metrics.cultural_accuracy_score = (total_accuracy + cultural_accuracy) / metrics.message_count
                    
                    await self.redis_client.setex(key, 3600, metrics.model_dump_json())
            
            # Record individual metrics
            await self.record_metric("messages_processed", 1.0)
            await self.record_metric("response_time", response_time, MetricType.TIMER)
            
            if cultural_accuracy > 0:
                await self.record_metric("cultural_accuracy", cultural_accuracy, MetricType.GAUGE)
                
        except Exception as e:
            logger.error(f"Failed to record message processing: {e}")
    
    async def record_revenue_event(
        self, 
        conversation_id: str, 
        amount: float, 
        event_type: str = "upsell"
    ):
        """Record a revenue-generating event."""
        try:
            # Update conversation metrics
            if self.redis_client:
                key = f"conversation_metrics:{conversation_id}"
                data = await self.redis_client.get(key)
                
                if data:
                    metrics = ConversationMetrics.model_validate_json(data)
                    metrics.revenue_generated += amount
                    
                    if event_type == "upsell":
                        metrics.upsells_successful += 1
                    
                    await self.redis_client.setex(key, 3600, metrics.model_dump_json())
            
            # Record metrics
            await self.record_metric("revenue_generated", amount, MetricType.COUNTER)
            await self.record_metric(f"{event_type}_events", 1.0)
            
        except Exception as e:
            logger.error(f"Failed to record revenue event: {e}")
    
    async def record_escalation(self, conversation_id: str, reason: str):
        """Record a human escalation."""
        try:
            # Update conversation metrics
            if self.redis_client:
                key = f"conversation_metrics:{conversation_id}"
                data = await self.redis_client.get(key)
                
                if data:
                    metrics = ConversationMetrics.model_validate_json(data)
                    metrics.escalations += 1
                    await self.redis_client.setex(key, 3600, metrics.model_dump_json())
            
            await self.record_metric("escalations", 1.0, tags={"reason": reason})
            
        except Exception as e:
            logger.error(f"Failed to record escalation: {e}")
    
    async def get_conversation_metrics(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get metrics for a specific conversation."""
        try:
            if not self.redis_client:
                return None
                
            key = f"conversation_metrics:{conversation_id}"
            data = await self.redis_client.get(key)
            
            if data:
                return ConversationMetrics.model_validate_json(data)
                
        except Exception as e:
            logger.error(f"Failed to get conversation metrics: {e}")
        
        return None
    
    async def get_daily_metrics(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get aggregated daily metrics."""
        try:
            if not date:
                date = datetime.utcnow().date()
            
            # This would typically query a time-series database
            # For now, return mock data for testing
            return {
                "date": date.isoformat(),
                "conversations_started": 45,
                "messages_processed": 234,
                "avg_response_time": 1.8,
                "cultural_accuracy_avg": 0.92,
                "revenue_generated": 1250.0,
                "escalation_rate": 0.03,
                "satisfaction_score": 4.6
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily metrics: {e}")
            return {}
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data."""
        try:
            # This would aggregate data from various sources
            return {
                "real_time": {
                    "active_conversations": await self._get_active_conversations_count(),
                    "avg_response_time_1h": 1.9,
                    "cultural_accuracy_1h": 0.91,
                    "messages_per_minute": 12.5
                },
                "today": await self.get_daily_metrics(),
                "trends": {
                    "response_time_trend": "improving",
                    "satisfaction_trend": "stable",
                    "revenue_trend": "increasing"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get performance dashboard: {e}")
            return {}
    
    async def _get_active_conversations_count(self) -> int:
        """Get count of active conversations."""
        try:
            if not self.redis_client:
                return 0
                
            # Count conversation metrics keys
            keys = await self.redis_client.keys("conversation_metrics:*")
            return len(keys)
            
        except Exception as e:
            logger.error(f"Failed to get active conversations count: {e}")
            return 0
    
    async def _flush_metrics(self):
        """Flush metrics buffer to storage."""
        if not self.metrics_buffer:
            return
            
        try:
            # In a real implementation, this would send to a metrics backend
            # like Prometheus, InfluxDB, or CloudWatch
            
            if self.redis_client:
                # Store metrics in Redis for now
                for metric in self.metrics_buffer:
                    key = f"metric:{metric.name}:{int(metric.timestamp.timestamp())}"
                    await self.redis_client.setex(
                        key, 
                        86400,  # 24 hours TTL
                        metric.model_dump_json()
                    )
            
            logger.debug(f"Flushed {len(self.metrics_buffer)} metrics")
            self.metrics_buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush metrics: {e}")
    
    async def _flush_metrics_periodically(self):
        """Periodically flush metrics buffer."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
            except Exception as e:
                logger.error(f"Error in periodic metrics flush: {e}")
    
    async def cleanup(self):
        """Cleanup analytics service."""
        try:
            # Flush remaining metrics
            await self._flush_metrics()
            
            if self.redis_client:
                await self.redis_client.close()
                
        except Exception as e:
            logger.error(f"Error during analytics cleanup: {e}")


# Global analytics service instance - lazy initialization for testing
analytics_service = None

def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance with lazy initialization."""
    global analytics_service
    if analytics_service is None:
        analytics_service = AnalyticsService()
    return analytics_service
