"""
Instructor models for structured LLM outputs with validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

import instructor
from pydantic import BaseModel, Field, validator


class CulturalResponse(BaseModel):
    """Culturally appropriate response with validation."""
    
    message: str = Field(..., description="Culturally appropriate response message")
    language: Literal["ar", "fr", "en", "es"] = Field(..., description="Response language")
    cultural_markers: List[str] = Field(
        default_factory=list, 
        description="Cultural elements included in response"
    )
    tone_adaptation: str = Field(..., description="Applied communication style")
    formality_level: Literal["low", "medium", "high"] = Field(
        default="medium", 
        description="Formality level used"
    )
    religious_sensitivity: bool = Field(
        default=False, 
        description="Whether religious considerations were applied"
    )
    local_context_included: bool = Field(
        default=False, 
        description="Whether local Moroccan context was included"
    )
    
    @validator("message")
    def validate_message_length(cls, v):
        """Ensure message is not too long for WhatsApp."""
        if len(v) > 4096:
            raise ValueError("Message too long for WhatsApp (max 4096 characters)")
        return v
    
    @validator("cultural_markers")
    def validate_cultural_markers(cls, v):
        """Validate cultural markers are appropriate."""
        valid_markers = [
            "moroccan_hospitality", "islamic_greeting", "family_oriented",
            "respectful_tone", "traditional_values", "french_sophistication",
            "english_enthusiasm", "spanish_warmth", "ramadan_awareness",
            "prayer_time_consideration", "halal_emphasis", "local_pride"
        ]
        for marker in v:
            if marker not in valid_markers:
                raise ValueError(f"Invalid cultural marker: {marker}")
        return v


class IntentClassificationResult(BaseModel):
    """Structured intent classification with confidence scores."""
    
    primary_intent: Literal[
        "greeting", "booking_inquiry", "service_request", "complaint", 
        "compliment", "information_request", "emergency", "upselling_opportunity",
        "direct_booking_interest", "cultural_inquiry", "local_recommendation", 
        "farewell", "unclear"
    ] = Field(..., description="Primary intent of the message")
    
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Classification confidence (0-1)"
    )
    
    secondary_intents: List[str] = Field(
        default_factory=list, 
        description="Additional detected intents"
    )
    
    urgency_level: Literal["low", "medium", "high", "critical"] = Field(
        default="medium", 
        description="Message urgency level"
    )
    
    requires_human_escalation: bool = Field(
        default=False, 
        description="Whether human intervention is needed"
    )
    
    cultural_sensitivity_required: bool = Field(
        default=False, 
        description="Whether cultural awareness is critical"
    )
    
    reasoning: str = Field(..., description="Explanation for the classification")
    
    @validator("confidence_score")
    def validate_confidence(cls, v):
        """Ensure confidence score is reasonable."""
        if v < 0.3:
            raise ValueError("Confidence score too low - classification may be unreliable")
        return v


class EmotionalAnalysis(BaseModel):
    """Emotional state analysis with cultural context."""
    
    primary_emotion: Literal[
        "joy", "satisfaction", "excitement", "gratitude", "curiosity",
        "neutral", "confusion", "concern", "frustration", "anger", 
        "disappointment", "anxiety", "urgency"
    ] = Field(..., description="Primary detected emotion")
    
    emotion_intensity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Emotion intensity (0-1)"
    )
    
    sentiment_score: float = Field(
        ..., 
        ge=-1.0, 
        le=1.0, 
        description="Overall sentiment (-1 to 1)"
    )
    
    cultural_emotion_markers: List[str] = Field(
        default_factory=list, 
        description="Culture-specific emotional indicators"
    )
    
    requires_empathy: bool = Field(
        default=False, 
        description="Whether empathetic response is needed"
    )
    
    escalation_risk: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Risk of situation escalating"
    )
    
    recommended_response_tone: Literal[
        "warm", "professional", "empathetic", "enthusiastic", 
        "apologetic", "reassuring", "celebratory"
    ] = Field(..., description="Recommended tone for response")
    
    cultural_considerations: List[str] = Field(
        default_factory=list, 
        description="Cultural factors to consider in response"
    )


class ServiceAction(BaseModel):
    """Structured service action with business logic."""
    
    action_type: Literal[
        "booking", "information", "escalation", "upselling", 
        "direct_booking_conversion", "service_request", "follow_up"
    ] = Field(..., description="Type of service action")
    
    priority: Literal["immediate", "scheduled", "proactive"] = Field(
        default="immediate", 
        description="Action execution priority"
    )
    
    staff_notification: bool = Field(
        default=False, 
        description="Whether to notify staff"
    )
    
    guest_confirmation_required: bool = Field(
        default=False, 
        description="Whether guest confirmation is needed"
    )
    
    revenue_impact: Optional[float] = Field(
        None, 
        description="Expected revenue impact in local currency"
    )
    
    service_details: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Specific service parameters"
    )
    
    cultural_adaptations: List[str] = Field(
        default_factory=list, 
        description="Cultural adaptations for service delivery"
    )
    
    timing_considerations: Optional[str] = Field(
        None, 
        description="Timing considerations (e.g., prayer times, meal times)"
    )
    
    success_probability: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Estimated success probability"
    )


class UpsellOpportunity(BaseModel):
    """Structured upselling opportunity analysis."""
    
    opportunity_type: Literal[
        "room_upgrade", "spa_services", "dining", "excursions", 
        "transportation", "extended_stay", "additional_services"
    ] = Field(..., description="Type of upselling opportunity")
    
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence in opportunity success"
    )
    
    recommended_services: List[str] = Field(
        ..., 
        description="Specific services to recommend"
    )
    
    pricing_strategy: Literal[
        "standard", "discount", "bundle", "loyalty_reward", "cultural_package"
    ] = Field(default="standard", description="Recommended pricing approach")
    
    optimal_timing: Literal[
        "immediate", "check_in", "during_stay", "before_checkout", "post_stay"
    ] = Field(default="immediate", description="Best time to present offer")
    
    cultural_appeal_factors: List[str] = Field(
        default_factory=list, 
        description="Cultural factors that make offer appealing"
    )
    
    expected_conversion_rate: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Expected conversion probability"
    )
    
    revenue_potential: float = Field(
        ..., 
        ge=0.0, 
        description="Potential additional revenue"
    )
    
    personalization_elements: List[str] = Field(
        default_factory=list, 
        description="Elements to personalize the offer"
    )


class DirectBookingConversion(BaseModel):
    """Direct booking conversion opportunity analysis."""
    
    conversion_potential: Literal["low", "medium", "high"] = Field(
        ..., 
        description="Potential for converting to direct booking"
    )
    
    guest_receptivity_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Guest's receptivity to direct booking"
    )
    
    conversion_tactics: List[str] = Field(
        ..., 
        description="Recommended conversion strategies"
    )
    
    incentive_recommendations: List[str] = Field(
        default_factory=list, 
        description="Incentives to offer for direct booking"
    )
    
    cultural_motivators: List[str] = Field(
        default_factory=list, 
        description="Culture-specific motivating factors"
    )
    
    commission_savings_offer: Optional[float] = Field(
        None, 
        description="Percentage of commission savings to share"
    )
    
    loyalty_benefits: List[str] = Field(
        default_factory=list, 
        description="Loyalty program benefits to highlight"
    )
    
    follow_up_schedule: List[str] = Field(
        default_factory=list, 
        description="Recommended follow-up timing"
    )
    
    expected_lifetime_value: Optional[float] = Field(
        None, 
        description="Expected customer lifetime value"
    )


class KnowledgeRetrieval(BaseModel):
    """Structured knowledge retrieval result."""
    
    query_understanding: str = Field(
        ..., 
        description="Understanding of what information is needed"
    )
    
    relevant_knowledge: List[str] = Field(
        ..., 
        description="Relevant knowledge pieces found"
    )
    
    confidence_scores: List[float] = Field(
        ..., 
        description="Confidence score for each knowledge piece"
    )
    
    cultural_relevance: List[float] = Field(
        default_factory=list, 
        description="Cultural relevance score for each piece"
    )
    
    knowledge_gaps: List[str] = Field(
        default_factory=list, 
        description="Identified gaps in available knowledge"
    )
    
    requires_real_time_data: bool = Field(
        default=False, 
        description="Whether real-time data lookup is needed"
    )
    
    fallback_responses: List[str] = Field(
        default_factory=list, 
        description="Fallback responses if knowledge is insufficient"
    )


class ResponsePlan(BaseModel):
    """Comprehensive response generation plan."""
    
    response_strategy: Literal[
        "direct_answer", "guided_discovery", "upselling_focus", 
        "cultural_storytelling", "problem_solving", "relationship_building"
    ] = Field(..., description="Overall response strategy")
    
    message_structure: List[str] = Field(
        ..., 
        description="Ordered list of message components"
    )
    
    cultural_adaptations: CulturalResponse = Field(
        ..., 
        description="Cultural adaptations to apply"
    )
    
    include_upselling: bool = Field(
        default=False, 
        description="Whether to include upselling elements"
    )
    
    include_local_recommendations: bool = Field(
        default=False, 
        description="Whether to include local recommendations"
    )
    
    follow_up_required: bool = Field(
        default=False, 
        description="Whether follow-up is needed"
    )
    
    estimated_satisfaction_impact: float = Field(
        default=0.0, 
        ge=-1.0, 
        le=1.0, 
        description="Expected impact on guest satisfaction"
    )
    
    success_metrics: List[str] = Field(
        default_factory=list, 
        description="Metrics to track for response success"
    )


class ProactiveAction(BaseModel):
    """Proactive action recommendation with timing and context."""
    
    action_type: Literal[
        "welcome_message", "check_in_reminder", "service_offer", 
        "local_recommendation", "weather_update", "cultural_event_notification",
        "upsell_opportunity", "feedback_request", "checkout_assistance",
        "follow_up_message", "special_occasion_greeting", "emergency_notification"
    ] = Field(..., description="Type of proactive action")
    
    trigger_condition: str = Field(
        ..., 
        description="Condition that triggered this proactive action"
    )
    
    optimal_timing: Literal[
        "immediate", "within_hour", "within_day", "scheduled", "event_based"
    ] = Field(default="immediate", description="Optimal timing for action")
    
    priority_level: Literal["low", "medium", "high", "urgent"] = Field(
        default="medium", 
        description="Priority level of the action"
    )
    
    personalization_data: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Data for personalizing the action"
    )
    
    cultural_considerations: List[str] = Field(
        default_factory=list, 
        description="Cultural factors to consider"
    )
    
    expected_impact: Literal[
        "satisfaction_boost", "revenue_generation", "problem_prevention",
        "relationship_building", "information_sharing", "service_enhancement"
    ] = Field(..., description="Expected impact of the action")
    
    success_probability: float = Field(
        default=0.5, 
        ge=0.0, 
        le=1.0, 
        description="Estimated probability of positive outcome"
    )
    
    message_template: Optional[str] = Field(
        None, 
        description="Template for the proactive message"
    )
    
    requires_staff_approval: bool = Field(
        default=False, 
        description="Whether staff approval is needed before execution"
    )
    
    follow_up_actions: List[str] = Field(
        default_factory=list, 
        description="Recommended follow-up actions"
    )
    
    @validator("success_probability")
    def validate_success_probability(cls, v):
        """Ensure success probability is reasonable."""
        if v < 0.1:
            raise ValueError("Success probability too low for proactive action")
        return v


# Instructor client setup for structured outputs
def get_instructor_client():
    """Get configured instructor client for structured outputs."""
    from openai import OpenAI
    from app.core.config import get_settings
    
    settings = get_settings()
    
    client = instructor.from_openai(
        OpenAI(api_key=settings.openai_api_key),
        mode=instructor.Mode.TOOLS
    )
    
    return client
