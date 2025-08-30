"""
Advanced Proactive Intelligence Service for Anticipatory Guest Experience
Sophisticated proactive engagement with cultural intelligence and revenue optimization
"""

import asyncio
import json
from datetime import datetime, timedelta, time
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
import uuid
from dataclasses import dataclass

import httpx
from loguru import logger
from pydantic import BaseModel, Field
import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import get_settings
from app.models.agent_state import GuestProfile, CulturalContext, Language
from app.models.instructor_models import ProactiveAction, get_instructor_client


class ProactiveEventType(str, Enum):
    """Types of proactive events to monitor."""
    ARRIVAL_PREPARATION = "arrival_preparation"
    WELCOME_SEQUENCE = "welcome_sequence"
    EXPERIENCE_ENHANCEMENT = "experience_enhancement"
    DEPARTURE_PREPARATION = "departure_preparation"
    SATISFACTION_CHECK = "satisfaction_check"
    UPSELLING_OPPORTUNITY = "upselling_opportunity"
    CULTURAL_MOMENT = "cultural_moment"
    WEATHER_ALERT = "weather_alert"
    LOCAL_EVENT_NOTIFICATION = "local_event_notification"
    LOYALTY_ENGAGEMENT = "loyalty_engagement"


class ProactiveTrigger(str, Enum):
    """Triggers for proactive actions."""
    TIME_BASED = "time_based"
    BEHAVIOR_BASED = "behavior_based"
    CONTEXT_BASED = "context_based"
    CULTURAL_BASED = "cultural_based"
    REVENUE_BASED = "revenue_based"
    SATISFACTION_BASED = "satisfaction_based"


class ProactiveMessage(BaseModel):
    """Proactive message with cultural and contextual intelligence."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    guest_id: str
    event_type: ProactiveEventType
    trigger: ProactiveTrigger
    message_content: str
    cultural_adaptation: Dict[str, str] = Field(default_factory=dict)
    timing: datetime
    priority: int = Field(ge=1, le=10, default=5)
    revenue_impact: Optional[float] = None
    satisfaction_impact: Optional[float] = None
    cultural_relevance: float = Field(ge=0.0, le=1.0, default=0.5)
    personalization_score: float = Field(ge=0.0, le=1.0, default=0.5)
    delivery_channel: str = "whatsapp"
    status: str = "pending"  # pending, sent, delivered, responded
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GuestJourneyStage(str, Enum):
    """Guest journey stages for proactive intelligence."""
    PRE_ARRIVAL = "pre_arrival"
    ARRIVAL = "arrival"
    EARLY_STAY = "early_stay"
    MID_STAY = "mid_stay"
    LATE_STAY = "late_stay"
    DEPARTURE = "departure"
    POST_DEPARTURE = "post_departure"


class SatisfactionTrend(BaseModel):
    """Guest satisfaction trend analysis."""
    guest_id: str
    current_score: float = Field(ge=0.0, le=10.0)
    trend: str  # improving, declining, stable
    risk_level: str  # low, medium, high
    intervention_recommended: bool = False
    cultural_factors: List[str] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class ProactiveService:
    """Sophisticated proactive intelligence service for anticipatory guest experience."""
    
    def __init__(self):
        self.settings = get_settings()
        self.instructor_client = get_instructor_client()
        
        # Redis for real-time data and caching
        self.redis_client: Optional[redis.Redis] = None
        
        # Scheduler for proactive actions
        self.scheduler = AsyncIOScheduler()
        
        # Guest journey tracking
        self.guest_journeys: Dict[str, Dict[str, Any]] = {}
        self.satisfaction_trends: Dict[str, SatisfactionTrend] = {}
        
        # Proactive intelligence rules
        self.proactive_rules = self._initialize_proactive_rules()
        self.cultural_moments = self._initialize_cultural_moments()
        
        # Message queue and delivery tracking
        self.proactive_queue = asyncio.Queue(maxsize=500)
        self.delivery_tracking: Dict[str, ProactiveMessage] = {}
        
        # External data sources
        self.weather_client = httpx.AsyncClient(
            base_url="https://api.openweathermap.org/data/2.5",
            timeout=10.0
        )
        
        # Performance metrics
        self.proactive_metrics = {
            "messages_sent": 0,
            "satisfaction_improvements": 0,
            "revenue_generated": 0.0,
            "cultural_moments_captured": 0,
            "avg_response_rate": 0.0
        }
        
        # Background monitoring
        self._background_tasks: List[asyncio.Task] = []
        self._running = False
    
    async def initialize(self) -> None:
        """Initialize proactive service with monitoring and scheduling."""
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                password=self.settings.redis_password,
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Start scheduler
            self.scheduler.start()
            
            # Initialize proactive monitoring
            await self._initialize_proactive_monitoring()
            
            # Start background tasks
            self._running = True
            self._background_tasks.extend([
                asyncio.create_task(self._process_proactive_queue()),
                asyncio.create_task(self._monitor_guest_journeys()),
                asyncio.create_task(self._analyze_satisfaction_trends()),
                asyncio.create_task(self._monitor_cultural_moments()),
                asyncio.create_task(self._monitor_external_events())
            ])
            
            logger.info("✅ Advanced Proactive Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Proactive service initialization failed: {e}")
            raise
    
    async def start_guest_journey_monitoring(
        self,
        guest_profile: GuestProfile,
        cultural_context: CulturalContext,
        booking_details: Dict[str, Any]
    ) -> None:
        """Start comprehensive guest journey monitoring with cultural intelligence."""
        
        try:
            guest_id = guest_profile.guest_id
            
            # Create guest journey tracking
            journey_data = {
                "guest_profile": guest_profile.dict(),
                "cultural_context": cultural_context.dict(),
                "booking_details": booking_details,
                "current_stage": GuestJourneyStage.PRE_ARRIVAL,
                "journey_start": datetime.now(),
                "touchpoints": [],
                "satisfaction_scores": [],
                "proactive_actions": [],
                "cultural_preferences": {},
                "revenue_opportunities": []
            }
            
            self.guest_journeys[guest_id] = journey_data
            
            # Schedule proactive actions based on booking timeline
            await self._schedule_journey_based_actions(guest_id, booking_details, cultural_context)
            
            # Initialize satisfaction monitoring
            await self._initialize_satisfaction_monitoring(guest_id)
            
            # Cache journey data
            if self.redis_client:
                await self.redis_client.setex(
                    f"guest_journey:{guest_id}",
                    86400 * 7,  # 7 days
                    json.dumps(journey_data, default=str)
                )
            
            logger.info(f"Guest journey monitoring started for {guest_id}")
            
        except Exception as e:
            logger.error(f"Guest journey monitoring initialization failed: {e}")
    
    async def trigger_proactive_action(
        self,
        event_type: ProactiveEventType,
        guest_id: str,
        context: Optional[Dict[str, Any]] = None,
        immediate: bool = False
    ) -> Optional[str]:
        """Trigger sophisticated proactive action with cultural intelligence."""
        
        try:
            # Get guest journey data
            journey_data = self.guest_journeys.get(guest_id)
            if not journey_data:
                logger.warning(f"No journey data found for guest {guest_id}")
                return None
            
            # Use Instructor for intelligent proactive action planning
            proactive_action = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=ProactiveAction,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_proactive_action_prompt(journey_data)
                    },
                    {
                        "role": "user",
                        "content": f"Plan proactive action for event: {event_type.value}\nContext: {context or {}}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Create culturally adapted proactive message
            proactive_message = await self._create_culturally_adapted_message(
                proactive_action, journey_data, event_type
            )
            
            # Schedule or queue the message
            if immediate:
                await self.proactive_queue.put(proactive_message)
            else:
                await self._schedule_proactive_message(proactive_message)
            
            # Update journey tracking
            journey_data["proactive_actions"].append({
                "event_type": event_type.value,
                "message_id": proactive_message.message_id,
                "timestamp": datetime.now(),
                "context": context
            })
            
            logger.info(f"Proactive action triggered: {proactive_message.message_id}")
            return proactive_message.message_id
            
        except Exception as e:
            logger.error(f"Proactive action triggering failed: {e}")
            return None
    
    async def analyze_satisfaction_and_intervene(
        self,
        guest_id: str,
        interaction_data: Dict[str, Any]
    ) -> Optional[ProactiveMessage]:
        """Analyze satisfaction trends and intervene proactively."""
        
        try:
            # Update satisfaction trend
            satisfaction_trend = await self._update_satisfaction_trend(guest_id, interaction_data)
            
            # Check if intervention is needed
            if not satisfaction_trend.intervention_recommended:
                return None
            
            # Get cultural context for intervention
            journey_data = self.guest_journeys.get(guest_id, {})
            cultural_context = journey_data.get("cultural_context", {})
            
            # Create culturally appropriate intervention
            intervention_message = await self._create_satisfaction_intervention(
                satisfaction_trend, cultural_context
            )
            
            # Queue immediate intervention
            await self.proactive_queue.put(intervention_message)
            
            # Update metrics
            self.proactive_metrics["satisfaction_improvements"] += 1
            
            logger.info(f"Satisfaction intervention triggered for {guest_id}")
            return intervention_message
            
        except Exception as e:
            logger.error(f"Satisfaction analysis and intervention failed: {e}")
            return None
    
    async def identify_and_act_on_cultural_moments(
        self,
        guest_id: str,
        current_context: Dict[str, Any]
    ) -> List[ProactiveMessage]:
        """Identify and act on cultural moments for enhanced experience."""
        
        try:
            journey_data = self.guest_journeys.get(guest_id, {})
            cultural_context = journey_data.get("cultural_context", {})
            
            if not cultural_context:
                return []
            
            # Identify relevant cultural moments
            cultural_moments = await self._identify_cultural_moments(
                cultural_context, current_context
            )
            
            proactive_messages = []
            
            for moment in cultural_moments:
                # Create culturally intelligent message
                message = await self._create_cultural_moment_message(
                    moment, guest_id, cultural_context
                )
                
                if message:
                    proactive_messages.append(message)
                    await self.proactive_queue.put(message)
            
            # Update metrics
            self.proactive_metrics["cultural_moments_captured"] += len(proactive_messages)
            
            return proactive_messages
            
        except Exception as e:
            logger.error(f"Cultural moments identification failed: {e}")
            return []
    
    async def _process_proactive_queue(self) -> None:
        """Process proactive messages with intelligent delivery timing."""
        
        while self._running:
            try:
                # Get message from queue
                message = await asyncio.wait_for(
                    self.proactive_queue.get(), timeout=1.0
                )
                
                # Check delivery timing
                if not await self._is_appropriate_delivery_time(message):
                    # Reschedule for later
                    await self._reschedule_message(message)
                    continue
                
                # Deliver the message
                success = await self._deliver_proactive_message(message)
                
                if success:
                    message.status = "sent"
                    self.delivery_tracking[message.message_id] = message
                    self.proactive_metrics["messages_sent"] += 1
                else:
                    # Retry logic
                    if message.priority >= 7:  # High priority messages
                        await asyncio.sleep(300)  # Wait 5 minutes
                        await self.proactive_queue.put(message)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Proactive queue processing error: {e}")
                await asyncio.sleep(1.0)
    
    async def _monitor_guest_journeys(self) -> None:
        """Monitor guest journeys and trigger stage-based actions."""
        
        while self._running:
            try:
                for guest_id, journey_data in self.guest_journeys.items():
                    # Update journey stage
                    new_stage = await self._determine_journey_stage(journey_data)
                    current_stage = journey_data.get("current_stage")
                    
                    if new_stage != current_stage:
                        # Stage transition detected
                        await self._handle_journey_stage_transition(
                            guest_id, current_stage, new_stage
                        )
                        journey_data["current_stage"] = new_stage
                
                # Sleep before next monitoring cycle
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Guest journey monitoring error: {e}")
                await asyncio.sleep(60)
    
    def _initialize_proactive_rules(self) -> Dict[str, Any]:
        """Initialize sophisticated proactive intelligence rules."""
        
        return {
            "arrival_preparation": {
                "trigger_time": "24_hours_before",
                "cultural_adaptations": {
                    "arabic": {
                        "greeting": "أهلاً وسهلاً بكم في رياضنا التقليدي",
                        "elements": ["prayer_direction", "halal_dining", "cultural_respect"]
                    },
                    "french": {
                        "greeting": "Bienvenue dans notre riad authentique",
                        "elements": ["sophistication", "cultural_heritage", "refined_service"]
                    },
                    "english": {
                        "greeting": "Welcome to our traditional Moroccan riad",
                        "elements": ["authentic_experience", "local_insights", "personalized_service"]
                    }
                }
            },
            "experience_enhancement": {
                "triggers": ["mid_stay", "positive_interaction", "cultural_interest"],
                "opportunities": [
                    {
                        "type": "cultural_tour",
                        "timing": "morning_preferred",
                        "cultural_relevance": {
                            "arabic": "Traditional medina heritage walk",
                            "french": "Découverte culturelle authentique",
                            "english": "Authentic cultural immersion"
                        }
                    },
                    {
                        "type": "cooking_class",
                        "timing": "afternoon_preferred",
                        "cultural_adaptation": True
                    }
                ]
            },
            "satisfaction_monitoring": {
                "check_intervals": ["day_1", "day_3", "pre_departure"],
                "intervention_triggers": {
                    "satisfaction_drop": 2.0,
                    "negative_sentiment": 0.3,
                    "cultural_mismatch": 0.4
                }
            }
        }
    
    def _initialize_cultural_moments(self) -> Dict[str, Any]:
        """Initialize cultural moments for proactive engagement."""
        
        return {
            "ramadan": {
                "detection": ["islamic_calendar", "guest_nationality"],
                "actions": ["iftar_arrangements", "prayer_facilities", "cultural_respect"]
            },
            "national_holidays": {
                "morocco": ["throne_day", "independence_day", "green_march"],
                "france": ["bastille_day", "christmas", "new_year"],
                "actions": ["cultural_acknowledgment", "special_arrangements"]
            },
            "personal_celebrations": {
                "detection": ["booking_notes", "special_requests", "guest_profile"],
                "actions": ["personalized_surprise", "cultural_celebration"]
            },
            "weather_events": {
                "triggers": ["rain", "extreme_heat", "sandstorm"],
                "cultural_responses": {
                    "arabic": "Traditional indoor activities and mint tea",
                    "french": "Sophisticated indoor experiences",
                    "english": "Cozy indoor cultural activities"
                }
            }
        }
    
    def _get_proactive_action_prompt(self, journey_data: Dict[str, Any]) -> str:
        """Get system prompt for proactive action planning."""
        
        cultural_context = journey_data.get("cultural_context", {})
        guest_profile = journey_data.get("guest_profile", {})
        
        return f"""You are an expert in proactive hospitality and Moroccan riad guest experience.
        
        Guest Context:
        - Nationality: {cultural_context.get('nationality', 'International')}
        - Language: {cultural_context.get('language', 'English')}
        - Previous Stays: {guest_profile.get('previous_stays', 0)}
        - Loyalty Tier: {guest_profile.get('loyalty_tier', 'standard')}
        - Journey Stage: {journey_data.get('current_stage', 'unknown')}
        
        Plan proactive actions that:
        1. Anticipate guest needs before they arise
        2. Respect cultural preferences and sensitivities
        3. Enhance the authentic Moroccan experience
        4. Create memorable moments aligned with guest expectations
        5. Identify revenue opportunities naturally and appropriately
        
        Focus on genuine hospitality that feels personal and culturally intelligent."""
    
    async def cleanup(self) -> None:
        """Cleanup proactive service resources."""
        
        try:
            self._running = False
            
            # Shutdown scheduler
            self.scheduler.shutdown()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Close HTTP clients
            await self.weather_client.aclose()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Proactive service cleanup completed")
            
        except Exception as e:
            logger.error(f"Proactive service cleanup failed: {e}")
    
    # Additional sophisticated methods would continue here...
    # Including cultural moment detection, satisfaction analysis, delivery timing, etc.
    
    async def _is_appropriate_delivery_time(self, message: ProactiveMessage) -> bool:
        """Check if it's appropriate time to deliver proactive message."""
        
        current_time = datetime.now().time()
        
        # Respect quiet hours (10 PM - 8 AM)
        if current_time >= time(22, 0) or current_time <= time(8, 0):
            return message.priority >= 9  # Only urgent messages
        
        # Respect cultural prayer times for Muslim guests
        journey_data = self.guest_journeys.get(message.guest_id, {})
        cultural_context = journey_data.get("cultural_context", {})
        
        if cultural_context.get("nationality") in ["Morocco", "Saudi Arabia", "UAE"]:
            # Check prayer times (simplified)
            prayer_times = [
                time(5, 30),   # Fajr
                time(12, 30),  # Dhuhr
                time(15, 30),  # Asr
                time(18, 30),  # Maghrib
                time(20, 0)    # Isha
            ]
            
            for prayer_time in prayer_times:
                if abs((current_time.hour * 60 + current_time.minute) - 
                      (prayer_time.hour * 60 + prayer_time.minute)) <= 15:
                    return False
        
        return True
