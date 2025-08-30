"""
FastAPI routes for Riad Concierge AI webhooks and API endpoints.
"""

import asyncio
import hmac
import hashlib
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
from pydantic import BaseModel, Field

from app.agents.riad_agent import RiadConciergeAgent
from app.core.config import get_settings
from app.models.agent_state import AgentState
from app.services.analytics_service import AnalyticsService
from app.services.guest_service import GuestService

router = APIRouter()
security = HTTPBearer(auto_error=False)


class WebhookVerification(BaseModel):
    """WhatsApp webhook verification model."""
    hub_mode: str = Field(..., alias="hub.mode")
    hub_challenge: str = Field(..., alias="hub.challenge") 
    hub_verify_token: str = Field(..., alias="hub.verify_token")


class WhatsAppMessage(BaseModel):
    """WhatsApp message webhook payload."""
    object: str
    entry: list


class MessageProcessingResponse(BaseModel):
    """Response for message processing."""
    success: bool
    conversation_id: str
    processing_time: float
    actions_executed: int
    errors: Optional[list] = None


class GuestProfileResponse(BaseModel):
    """Guest profile API response."""
    phone_number: str
    name: Optional[str]
    nationality: Optional[str]
    language_preference: str
    total_interactions: int
    satisfaction_score: Optional[float]
    loyalty_tier: str


class AnalyticsResponse(BaseModel):
    """Analytics data response."""
    total_conversations: int
    avg_response_time: float
    satisfaction_score: float
    conversion_rate: float
    revenue_impact: float


def get_agent_service(request: Request) -> RiadConciergeAgent:
    """Get agent service from app state."""
    return request.app.state.agent_service


def get_analytics_service() -> AnalyticsService:
    """Get analytics service."""
    return AnalyticsService()


def get_guest_service() -> GuestService:
    """Get guest service."""
    return GuestService()


def verify_whatsapp_signature(payload: bytes, signature: str) -> bool:
    """Verify WhatsApp webhook signature."""
    settings = get_settings()
    
    if not signature:
        return False
    
    try:
        # Remove 'sha256=' prefix
        signature = signature.replace('sha256=', '')
        
        # Calculate expected signature
        expected_signature = hmac.new(
            settings.whatsapp_app_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        # Compare signatures
        return hmac.compare_digest(signature, expected_signature)
        
    except Exception as e:
        logger.error(f"Signature verification failed: {e}")
        return False


@router.get("/webhook/whatsapp")
async def verify_whatsapp_webhook(
    request: Request,
    hub_mode: str = None,
    hub_challenge: str = None, 
    hub_verify_token: str = None
):
    """Verify WhatsApp webhook during setup."""
    settings = get_settings()
    
    # Check if this is a verification request
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_webhook_verify_token:
        logger.info("WhatsApp webhook verification successful")
        return Response(content=hub_challenge, media_type="text/plain")
    
    logger.warning("WhatsApp webhook verification failed")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/webhook/whatsapp", response_model=MessageProcessingResponse)
async def handle_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    agent_service: RiadConciergeAgent = Depends(get_agent_service)
):
    """Handle incoming WhatsApp messages."""
    
    try:
        # Get request body and signature
        body = await request.body()
        signature = request.headers.get('x-hub-signature-256', '')
        
        # Verify webhook signature
        if not verify_whatsapp_signature(body, signature):
            logger.warning("Invalid webhook signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        # Parse webhook payload
        payload = await request.json()
        
        # Extract message data
        entry = payload.get('entry', [])
        if not entry:
            return MessageProcessingResponse(
                success=True,
                conversation_id="",
                processing_time=0.0,
                actions_executed=0
            )
        
        changes = entry[0].get('changes', [])
        if not changes:
            return MessageProcessingResponse(
                success=True,
                conversation_id="",
                processing_time=0.0,
                actions_executed=0
            )
        
        value = changes[0].get('value', {})
        messages = value.get('messages', [])
        
        if not messages:
            return MessageProcessingResponse(
                success=True,
                conversation_id="",
                processing_time=0.0,
                actions_executed=0
            )
        
        # Process each message
        results = []
        for message in messages:
            # Add contact information
            contacts = value.get('contacts', [])
            contact = contacts[0] if contacts else {}
            
            # Prepare message data for agent
            message_data = {
                **message,
                'contact': contact,
                'metadata': value.get('metadata', {})
            }
            
            # Process message through agent workflow
            background_tasks.add_task(
                process_message_async,
                agent_service,
                message_data
            )
            
            results.append({
                'message_id': message.get('id'),
                'from': message.get('from'),
                'queued': True
            })
        
        return MessageProcessingResponse(
            success=True,
            conversation_id=f"batch_{len(results)}",
            processing_time=0.0,
            actions_executed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


async def process_message_async(
    agent_service: RiadConciergeAgent,
    message_data: Dict[str, Any]
) -> None:
    """Process message asynchronously in background."""
    try:
        result = await agent_service.process_message(message_data)
        
        logger.info(
            f"Message processed: {result.conversation_id}, "
            f"time: {result.processing_time:.2f}s, "
            f"actions: {len(result.actions)}, "
            f"errors: {len(result.errors)}"
        )
        
    except Exception as e:
        logger.error(f"Async message processing failed: {e}")


@router.get("/guest/{phone_number}", response_model=GuestProfileResponse)
async def get_guest_profile(
    phone_number: str,
    guest_service: GuestService = Depends(get_guest_service),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get guest profile information."""
    
    try:
        # Validate phone number format
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
        
        profile = await guest_service.get_guest_profile(phone_number)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Guest not found")
        
        return GuestProfileResponse(
            phone_number=profile.phone_number,
            name=profile.name,
            nationality=profile.nationality,
            language_preference=profile.language_preference.value,
            total_interactions=await guest_service.get_interaction_count(phone_number),
            satisfaction_score=profile.satisfaction_score,
            loyalty_tier=profile.loyalty_tier
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Guest profile retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Profile retrieval failed")


@router.post("/guest/{phone_number}/update")
async def update_guest_profile(
    phone_number: str,
    updates: Dict[str, Any],
    guest_service: GuestService = Depends(get_guest_service),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Update guest profile information."""
    
    try:
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
        
        updated_profile = await guest_service.update_guest_profile(
            phone_number, 
            updates
        )
        
        return {"success": True, "updated_fields": list(updates.keys())}
        
    except Exception as e:
        logger.error(f"Guest profile update failed: {e}")
        raise HTTPException(status_code=500, detail="Profile update failed")


@router.post("/message/send")
async def send_proactive_message(
    phone_number: str,
    message: str,
    language: str = "en",
    agent_service: RiadConciergeAgent = Depends(get_agent_service),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Send proactive message to guest."""
    
    try:
        if not phone_number.startswith('+'):
            phone_number = '+' + phone_number
        
        # Send message through WhatsApp service
        success = await agent_service.whatsapp_service.send_message(
            to=phone_number,
            message=message,
            language=language
        )
        
        return {"success": success, "message_sent": True}
        
    except Exception as e:
        logger.error(f"Proactive message sending failed: {e}")
        raise HTTPException(status_code=500, detail="Message sending failed")


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(
    days: int = 7,
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get analytics data for the specified period."""
    
    try:
        analytics_data = await analytics_service.get_analytics(days=days)
        
        return AnalyticsResponse(
            total_conversations=analytics_data.get('total_conversations', 0),
            avg_response_time=analytics_data.get('avg_response_time', 0.0),
            satisfaction_score=analytics_data.get('satisfaction_score', 0.0),
            conversion_rate=analytics_data.get('conversion_rate', 0.0),
            revenue_impact=analytics_data.get('revenue_impact', 0.0)
        )
        
    except Exception as e:
        logger.error(f"Analytics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Analytics retrieval failed")


@router.post("/analytics/interaction")
async def log_interaction(
    interaction_data: Dict[str, Any],
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Log interaction data for analytics."""
    
    try:
        await analytics_service.log_interaction(interaction_data)
        return {"success": True, "logged": True}
        
    except Exception as e:
        logger.error(f"Interaction logging failed: {e}")
        raise HTTPException(status_code=500, detail="Interaction logging failed")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "riad-concierge-ai",
        "version": "1.0.0"
    }


@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    # This will be handled by Prometheus middleware
    pass
