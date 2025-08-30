"""
LangGraph Agent State Models with Instructor validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, validator


class MessageType(str, Enum):
    """WhatsApp message types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    LOCATION = "location"
    CONTACT = "contact"
    INTERACTIVE = "interactive"
    BUTTON = "button"
    LIST = "list"


class Language(str, Enum):
    """Supported languages."""
    ARABIC = "ar"
    FRENCH = "fr"
    ENGLISH = "en"
    SPANISH = "es"


class BookingStatus(str, Enum):
    """Booking status types."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CHECKED_IN = "checked_in"
    CHECKED_OUT = "checked_out"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"


class RoomType(str, Enum):
    """Room types available."""
    STANDARD = "standard"
    DELUXE = "deluxe"
    SUITE = "suite"
    FAMILY = "family"
    PRESIDENTIAL = "presidential"


class CulturalContext(BaseModel):
    """Cultural context for personalized responses."""
    
    nationality: Optional[str] = Field(None, description="Guest's nationality")
    language: Language = Field(Language.ENGLISH, description="Detected/preferred language")
    communication_style: str = Field("neutral", description="Preferred communication style")
    cultural_markers: List[str] = Field(default_factory=list, description="Cultural elements to include")
    religious_considerations: List[str] = Field(default_factory=list, description="Religious sensitivities")
    service_preferences: List[str] = Field(default_factory=list, description="Service preferences")
    formality_level: str = Field("medium", description="Formality level (low/medium/high)")


class GuestProfile(BaseModel):
    """Guest profile information."""
    
    phone_number: str = Field(..., description="WhatsApp phone number")
    name: Optional[str] = Field(None, description="Guest name")
    nationality: Optional[str] = Field(None, description="Guest nationality")
    language_preference: Language = Field(Language.ENGLISH, description="Preferred language")
    stay_dates: Optional[Dict[str, str]] = Field(None, description="Check-in/out dates")
    room_type: Optional[str] = Field(None, description="Room type")
    guest_count: Optional[int] = Field(None, description="Number of guests")
    special_requests: List[str] = Field(default_factory=list, description="Special requests")
    dietary_restrictions: List[str] = Field(default_factory=list, description="Dietary restrictions")
    previous_stays: int = Field(0, description="Number of previous stays")
    loyalty_tier: str = Field("standard", description="Loyalty program tier")
    total_spent: float = Field(0.0, description="Total amount spent")
    satisfaction_score: Optional[float] = Field(None, description="Average satisfaction score")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Guest preferences")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class IntentType(str, Enum):
    """Message intent classifications."""
    GREETING = "greeting"
    BOOKING_INQUIRY = "booking_inquiry"
    SERVICE_REQUEST = "service_request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    INFORMATION_REQUEST = "information_request"
    EMERGENCY = "emergency"
    UPSELLING_OPPORTUNITY = "upselling_opportunity"
    DIRECT_BOOKING_INTEREST = "direct_booking_interest"
    CULTURAL_INQUIRY = "cultural_inquiry"
    LOCAL_RECOMMENDATION = "local_recommendation"
    FAREWELL = "farewell"
    UNCLEAR = "unclear"


class UrgencyLevel(str, Enum):
    """Message urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IntentClassification(BaseModel):
    """Intent classification result."""
    
    primary_intent: IntentType = Field(..., description="Primary intent of the message")
    secondary_intents: List[IntentType] = Field(default_factory=list, description="Secondary intents")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Classification confidence")
    urgency_level: UrgencyLevel = Field(UrgencyLevel.MEDIUM, description="Message urgency")
    requires_human_escalation: bool = Field(False, description="Needs human intervention")
    cultural_sensitivity_required: bool = Field(False, description="Requires cultural awareness")


class RetrievalResult(BaseModel):
    """RAG retrieval result."""
    
    content: str = Field(..., description="Retrieved content")
    source: str = Field(..., description="Content source")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    cultural_relevance: float = Field(0.0, ge=0.0, le=1.0, description="Cultural relevance score")


class ActionType(str, Enum):
    """Types of actions the agent can execute."""
    SEND_MESSAGE = "send_message"
    BOOK_SERVICE = "book_service"
    ESCALATE_TO_HUMAN = "escalate_to_human"
    UPDATE_GUEST_PROFILE = "update_guest_profile"
    SEND_PROACTIVE_MESSAGE = "send_proactive_message"
    CREATE_UPSELLING_OPPORTUNITY = "create_upselling_opportunity"
    LOG_INTERACTION = "log_interaction"
    SCHEDULE_FOLLOW_UP = "schedule_follow_up"


class Action(BaseModel):
    """Action to be executed by the agent."""
    
    action_type: ActionType = Field(..., description="Type of action")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")
    priority: int = Field(1, ge=1, le=10, description="Action priority (1=highest)")
    scheduled_time: Optional[datetime] = Field(None, description="When to execute (None=immediate)")
    requires_confirmation: bool = Field(False, description="Requires guest confirmation")
    staff_notification: bool = Field(False, description="Notify staff")
    revenue_impact: Optional[float] = Field(None, description="Expected revenue impact")


class ResponsePlan(BaseModel):
    """Plan for generating the response."""
    
    response_strategy: str = Field(..., description="Overall response strategy")
    cultural_adaptation: CulturalContext = Field(..., description="Cultural adaptations to apply")
    tone: str = Field("friendly", description="Response tone")
    include_upselling: bool = Field(False, description="Include upselling elements")
    include_local_recommendations: bool = Field(False, description="Include local recommendations")
    follow_up_required: bool = Field(False, description="Requires follow-up")
    estimated_satisfaction_impact: float = Field(0.0, ge=-1.0, le=1.0, description="Expected satisfaction impact")


class EmotionalState(BaseModel):
    """Guest's emotional state analysis."""
    
    primary_emotion: str = Field(..., description="Primary detected emotion")
    emotion_intensity: float = Field(..., ge=0.0, le=1.0, description="Emotion intensity")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall sentiment")
    cultural_emotion_markers: List[str] = Field(default_factory=list, description="Cultural emotion indicators")
    requires_empathy: bool = Field(False, description="Requires empathetic response")
    escalation_risk: float = Field(0.0, ge=0.0, le=1.0, description="Risk of escalation")


class AgentState(BaseModel):
    """Complete agent state for LangGraph workflow."""
    
    # Input message and context
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        default_factory=list, 
        description="Conversation messages"
    )
    raw_message: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Raw WhatsApp message data"
    )
    
    # Guest and cultural context
    guest_profile: Optional[GuestProfile] = Field(None, description="Guest profile information")
    cultural_context: Optional[CulturalContext] = Field(None, description="Cultural context")
    emotional_state: Optional[EmotionalState] = Field(None, description="Guest emotional state")
    
    # Intent and classification
    intent: Optional[IntentClassification] = Field(None, description="Message intent classification")
    
    # Knowledge retrieval
    cag_knowledge: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Static CAG knowledge results"
    )
    rag_results: List[RetrievalResult] = Field(
        default_factory=list, 
        description="Dynamic RAG retrieval results"
    )
    
    # Response planning
    response_plan: Optional[ResponsePlan] = Field(None, description="Response generation plan")
    
    # Actions and execution
    actions: List[Action] = Field(default_factory=list, description="Actions to execute")
    
    # Metadata and tracking
    conversation_id: str = Field(..., description="Unique conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    
    # Error handling
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    fallback_triggered: bool = Field(False, description="Whether fallback was triggered")
    
    # Performance tracking
    cag_retrieval_time: Optional[float] = Field(None, description="CAG retrieval time")
    rag_retrieval_time: Optional[float] = Field(None, description="RAG retrieval time")
    llm_processing_time: Optional[float] = Field(None, description="LLM processing time")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @validator("messages", pre=True)
    def validate_messages(cls, v):
        """Ensure messages are properly formatted."""
        if not isinstance(v, list):
            return []
        return v
    
    def add_message(self, message: Union[HumanMessage, AIMessage, SystemMessage]) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_latest_human_message(self) -> Optional[HumanMessage]:
        """Get the most recent human message."""
        for message in reversed(self.messages):
            if isinstance(message, HumanMessage):
                return message
        return None
    
    def get_conversation_history(self, limit: int = 10) -> List[Union[HumanMessage, AIMessage]]:
        """Get recent conversation history."""
        relevant_messages = [
            msg for msg in self.messages 
            if isinstance(msg, (HumanMessage, AIMessage))
        ]
        return relevant_messages[-limit:] if relevant_messages else []
