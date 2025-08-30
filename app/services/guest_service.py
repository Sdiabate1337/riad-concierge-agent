"""
Guest Service for Riad Concierge AI
Manages guest profiles, preferences, and interaction history
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import redis.asyncio as redis
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.agent_state import GuestProfile, CulturalContext, Language


class GuestPreference(BaseModel):
    """Guest preference model."""
    category: str = Field(..., description="Preference category")
    value: str = Field(..., description="Preference value")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in preference")
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class GuestInteraction(BaseModel):
    """Guest interaction history."""
    interaction_id: str = Field(..., description="Unique interaction ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str = Field(..., description="Type of message")
    intent: str = Field(..., description="Detected intent")
    satisfaction_score: Optional[float] = Field(None, ge=0.0, le=5.0)
    cultural_markers: List[str] = Field(default_factory=list)
    revenue_generated: float = Field(0.0, description="Revenue from this interaction")


class EnhancedGuestProfile(GuestProfile):
    """Enhanced guest profile with additional analytics."""
    total_interactions: int = Field(0, description="Total number of interactions")
    avg_satisfaction: float = Field(0.0, description="Average satisfaction score")
    total_revenue: float = Field(0.0, description="Total revenue generated")
    preferred_communication_style: str = Field("neutral", description="Preferred communication style")
    cultural_preferences: List[str] = Field(default_factory=list)
    last_interaction: Optional[datetime] = Field(None)
    loyalty_score: float = Field(0.0, ge=0.0, le=1.0, description="Guest loyalty score")
    preferences: List[GuestPreference] = Field(default_factory=list)
    interaction_history: List[GuestInteraction] = Field(default_factory=list)


class GuestService:
    """Guest management and profiling service."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize the guest service."""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                retry_on_timeout=True
            )
            await self.redis_client.ping()
            logger.info("Guest service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize guest service: {e}")
            # Continue without Redis for testing
            self.redis_client = None
    
    async def get_guest_profile(self, phone_number: str) -> Optional[EnhancedGuestProfile]:
        """Get enhanced guest profile by phone number."""
        try:
            if not self.redis_client:
                return self._create_default_profile(phone_number)
                
            key = f"guest_profile:{phone_number}"
            data = await self.redis_client.get(key)
            
            if data:
                return EnhancedGuestProfile.model_validate_json(data)
            else:
                # Create new profile
                profile = self._create_default_profile(phone_number)
                await self.save_guest_profile(profile)
                return profile
                
        except Exception as e:
            logger.error(f"Failed to get guest profile for {phone_number}: {e}")
            return self._create_default_profile(phone_number)
    
    async def save_guest_profile(self, profile: EnhancedGuestProfile):
        """Save guest profile to storage."""
        try:
            if not self.redis_client:
                return
                
            key = f"guest_profile:{profile.phone_number}"
            await self.redis_client.setex(
                key,
                86400 * 30,  # 30 days TTL
                profile.model_dump_json()
            )
            
        except Exception as e:
            logger.error(f"Failed to save guest profile: {e}")
    
    async def update_guest_interaction(
        self, 
        phone_number: str, 
        interaction: GuestInteraction
    ):
        """Update guest profile with new interaction."""
        try:
            profile = await self.get_guest_profile(phone_number)
            if not profile:
                return
            
            # Add interaction to history
            profile.interaction_history.append(interaction)
            
            # Keep only last 50 interactions
            if len(profile.interaction_history) > 50:
                profile.interaction_history = profile.interaction_history[-50:]
            
            # Update aggregate metrics
            profile.total_interactions += 1
            profile.last_interaction = interaction.timestamp
            
            if interaction.satisfaction_score:
                # Update average satisfaction
                total_satisfaction = profile.avg_satisfaction * (profile.total_interactions - 1)
                profile.avg_satisfaction = (total_satisfaction + interaction.satisfaction_score) / profile.total_interactions
            
            # Update revenue
            profile.total_revenue += interaction.revenue_generated
            
            # Update loyalty score based on interactions and satisfaction
            profile.loyalty_score = self._calculate_loyalty_score(profile)
            
            await self.save_guest_profile(profile)
            
        except Exception as e:
            logger.error(f"Failed to update guest interaction: {e}")
    
    async def add_guest_preference(
        self, 
        phone_number: str, 
        category: str, 
        value: str, 
        confidence: float = 1.0
    ):
        """Add or update guest preference."""
        try:
            profile = await self.get_guest_profile(phone_number)
            if not profile:
                return
            
            # Find existing preference or create new one
            existing_pref = None
            for pref in profile.preferences:
                if pref.category == category:
                    existing_pref = pref
                    break
            
            if existing_pref:
                # Update existing preference
                existing_pref.value = value
                existing_pref.confidence = max(existing_pref.confidence, confidence)
                existing_pref.last_updated = datetime.utcnow()
            else:
                # Add new preference
                new_pref = GuestPreference(
                    category=category,
                    value=value,
                    confidence=confidence
                )
                profile.preferences.append(new_pref)
            
            await self.save_guest_profile(profile)
            
        except Exception as e:
            logger.error(f"Failed to add guest preference: {e}")
    
    async def get_guest_preferences(self, phone_number: str) -> Dict[str, str]:
        """Get guest preferences as a dictionary."""
        try:
            profile = await self.get_guest_profile(phone_number)
            if not profile:
                return {}
            
            return {
                pref.category: pref.value 
                for pref in profile.preferences
                if pref.confidence > 0.5
            }
            
        except Exception as e:
            logger.error(f"Failed to get guest preferences: {e}")
            return {}
    
    async def update_cultural_context(
        self, 
        phone_number: str, 
        cultural_context: CulturalContext
    ):
        """Update guest profile with cultural context."""
        try:
            profile = await self.get_guest_profile(phone_number)
            if not profile:
                return
            
            # Update basic profile info
            if cultural_context.nationality:
                profile.nationality = cultural_context.nationality
            
            profile.language = cultural_context.language
            profile.preferred_communication_style = cultural_context.communication_style
            
            # Update cultural preferences
            profile.cultural_preferences = cultural_context.cultural_markers
            
            # Add cultural preferences
            for marker in cultural_context.cultural_markers:
                await self.add_guest_preference(
                    phone_number, 
                    "cultural_marker", 
                    marker, 
                    0.8
                )
            
            # Add religious considerations
            for consideration in cultural_context.religious_considerations:
                await self.add_guest_preference(
                    phone_number, 
                    "religious_consideration", 
                    consideration, 
                    0.9
                )
            
            await self.save_guest_profile(profile)
            
        except Exception as e:
            logger.error(f"Failed to update cultural context: {e}")
    
    async def get_guest_analytics(self, phone_number: str) -> Dict[str, Any]:
        """Get guest analytics and insights."""
        try:
            profile = await self.get_guest_profile(phone_number)
            if not profile:
                return {}
            
            # Calculate analytics
            recent_interactions = [
                interaction for interaction in profile.interaction_history
                if interaction.timestamp > datetime.utcnow() - timedelta(days=30)
            ]
            
            return {
                "total_interactions": profile.total_interactions,
                "recent_interactions": len(recent_interactions),
                "avg_satisfaction": profile.avg_satisfaction,
                "total_revenue": profile.total_revenue,
                "loyalty_score": profile.loyalty_score,
                "preferred_language": profile.language,
                "cultural_markers": profile.cultural_preferences,
                "last_interaction": profile.last_interaction.isoformat() if profile.last_interaction else None,
                "engagement_level": self._calculate_engagement_level(profile),
                "revenue_potential": self._calculate_revenue_potential(profile)
            }
            
        except Exception as e:
            logger.error(f"Failed to get guest analytics: {e}")
            return {}
    
    async def search_guests(
        self, 
        criteria: Dict[str, Any], 
        limit: int = 50
    ) -> List[EnhancedGuestProfile]:
        """Search guests by criteria."""
        try:
            if not self.redis_client:
                return []
            
            # In a real implementation, this would use a proper search index
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Failed to search guests: {e}")
            return []
    
    def _create_default_profile(self, phone_number: str) -> EnhancedGuestProfile:
        """Create a default guest profile."""
        return EnhancedGuestProfile(
            phone_number=phone_number,
            name="",
            nationality="Unknown",
            language=Language.ENGLISH,
            preferences={}
        )
    
    def _calculate_loyalty_score(self, profile: EnhancedGuestProfile) -> float:
        """Calculate guest loyalty score."""
        try:
            # Base score from interactions
            interaction_score = min(profile.total_interactions / 10.0, 0.4)
            
            # Satisfaction score
            satisfaction_score = (profile.avg_satisfaction / 5.0) * 0.3
            
            # Revenue score
            revenue_score = min(profile.total_revenue / 1000.0, 0.2)
            
            # Recency score
            recency_score = 0.1
            if profile.last_interaction:
                days_since = (datetime.utcnow() - profile.last_interaction).days
                recency_score = max(0.1 - (days_since / 365.0), 0.0)
            
            return min(interaction_score + satisfaction_score + revenue_score + recency_score, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate loyalty score: {e}")
            return 0.0
    
    def _calculate_engagement_level(self, profile: EnhancedGuestProfile) -> str:
        """Calculate guest engagement level."""
        try:
            if profile.total_interactions >= 20:
                return "high"
            elif profile.total_interactions >= 5:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Failed to calculate engagement level: {e}")
            return "unknown"
    
    def _calculate_revenue_potential(self, profile: EnhancedGuestProfile) -> str:
        """Calculate guest revenue potential."""
        try:
            if profile.total_revenue >= 500:
                return "high"
            elif profile.total_revenue >= 100:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            logger.error(f"Failed to calculate revenue potential: {e}")
            return "unknown"
    
    async def cleanup(self):
        """Cleanup guest service."""
        try:
            if self.redis_client:
                await self.redis_client.close()
                
        except Exception as e:
            logger.error(f"Error during guest service cleanup: {e}")


# Global guest service instance - lazy initialization for testing
guest_service = None

def get_guest_service() -> GuestService:
    """Get guest service instance with lazy initialization."""
    global guest_service
    if guest_service is None:
        guest_service = GuestService()
    return guest_service
