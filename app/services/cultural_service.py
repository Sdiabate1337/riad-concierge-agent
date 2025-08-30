"""
Advanced Cultural Intelligence Service for Moroccan Riad Hospitality
Sophisticated cultural adaptation with deep understanding of guest backgrounds
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import re

from langdetect import detect, DetectorFactory
import pycountry
from loguru import logger
from pydantic import BaseModel, Field
import redis.asyncio as redis

from app.core.config import get_settings
from app.models.agent_state import CulturalContext, Language, GuestProfile
from app.models.instructor_models import CulturalResponse, get_instructor_client


# Set seed for consistent language detection
DetectorFactory.seed = 0


class CulturalDimension(str, Enum):
    """Hofstede's cultural dimensions for sophisticated analysis."""
    POWER_DISTANCE = "power_distance"
    INDIVIDUALISM = "individualism"
    MASCULINITY = "masculinity"
    UNCERTAINTY_AVOIDANCE = "uncertainty_avoidance"
    LONG_TERM_ORIENTATION = "long_term_orientation"
    INDULGENCE = "indulgence"


class CommunicationStyle(str, Enum):
    """Communication style preferences."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    FORMAL = "formal"
    INFORMAL = "informal"
    HIGH_CONTEXT = "high_context"
    LOW_CONTEXT = "low_context"


class ReligiousConsideration(BaseModel):
    """Religious and cultural considerations."""
    religion: Optional[str] = None
    dietary_restrictions: List[str] = Field(default_factory=list)
    prayer_requirements: bool = False
    religious_holidays: List[str] = Field(default_factory=list)
    cultural_sensitivities: List[str] = Field(default_factory=list)


class CulturalProfile(BaseModel):
    """Comprehensive cultural profile for guests."""
    nationality: str
    language: Language
    communication_style: CommunicationStyle
    cultural_dimensions: Dict[CulturalDimension, float]
    religious_considerations: ReligiousConsideration
    hospitality_expectations: Dict[str, Any]
    service_preferences: Dict[str, Any]
    cultural_markers: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class CulturalService:
    """Sophisticated cultural intelligence service for personalized hospitality."""
    
    def __init__(self):
        self.settings = get_settings()
        self.instructor_client = get_instructor_client()
        
        # Redis for caching cultural profiles
        self.redis_client: Optional[redis.Redis] = None
        
        # Cultural knowledge databases
        self.cultural_database = self._initialize_cultural_database()
        self.nationality_patterns = self._initialize_nationality_patterns()
        self.communication_patterns = self._initialize_communication_patterns()
        
        # Language detection confidence threshold
        self.language_confidence_threshold = 0.8
        
        # Cultural adaptation cache
        self.adaptation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=24)
        
        # Performance metrics
        self.cultural_metrics = {
            "profiles_created": 0,
            "adaptations_performed": 0,
            "cache_hits": 0,
            "avg_confidence_score": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize cultural service with connections and caches."""
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                password=self.settings.redis_password,
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Warm up cultural adaptation cache
            await self._warm_up_cultural_cache()
            
            logger.info("✅ Advanced Cultural Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Cultural service initialization failed: {e}")
            raise
    
    async def create_comprehensive_cultural_profile(
        self,
        phone_number: str,
        message_text: str,
        guest_profile: Optional[GuestProfile] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> CulturalProfile:
        """Create comprehensive cultural profile with sophisticated analysis."""
        
        try:
            # Check cache first
            cache_key = f"cultural_profile:{phone_number}"
            
            if self.redis_client:
                cached_profile = await self.redis_client.get(cache_key)
                if cached_profile:
                    self.cultural_metrics["cache_hits"] += 1
                    return CulturalProfile.parse_raw(cached_profile)
            
            # Detect language with confidence scoring
            language, language_confidence = await self._detect_language_advanced(message_text)
            
            # Infer nationality with multiple methods
            nationality, nationality_confidence = await self._infer_nationality_sophisticated(
                message_text, phone_number, guest_profile, additional_context
            )
            
            # Analyze communication style
            communication_style = await self._analyze_communication_style(
                message_text, nationality, language
            )
            
            # Get cultural dimensions
            cultural_dimensions = await self._get_cultural_dimensions(nationality)
            
            # Determine religious considerations
            religious_considerations = await self._determine_religious_considerations(
                nationality, additional_context
            )
            
            # Build hospitality expectations
            hospitality_expectations = await self._build_hospitality_expectations(
                nationality, cultural_dimensions, religious_considerations
            )
            
            # Determine service preferences
            service_preferences = await self._determine_service_preferences(
                nationality, communication_style, guest_profile
            )
            
            # Extract cultural markers
            cultural_markers = await self._extract_cultural_markers(
                message_text, nationality, language
            )
            
            # Calculate overall confidence score
            confidence_score = (language_confidence + nationality_confidence) / 2
            
            # Create comprehensive profile
            cultural_profile = CulturalProfile(
                nationality=nationality,
                language=language,
                communication_style=communication_style,
                cultural_dimensions=cultural_dimensions,
                religious_considerations=religious_considerations,
                hospitality_expectations=hospitality_expectations,
                service_preferences=service_preferences,
                cultural_markers=cultural_markers,
                confidence_score=confidence_score
            )
            
            # Cache the profile
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    int(self.cache_ttl.total_seconds()),
                    cultural_profile.json()
                )
            
            # Update metrics
            self.cultural_metrics["profiles_created"] += 1
            self.cultural_metrics["avg_confidence_score"] = (
                (self.cultural_metrics["avg_confidence_score"] * 
                 (self.cultural_metrics["profiles_created"] - 1) + confidence_score) /
                self.cultural_metrics["profiles_created"]
            )
            
            logger.info(f"Cultural profile created with {confidence_score:.2f} confidence")
            return cultural_profile
            
        except Exception as e:
            logger.error(f"Cultural profile creation failed: {e}")
            return await self._create_fallback_cultural_profile(phone_number)
    
    async def adapt_message_culturally(
        self,
        message: str,
        cultural_profile: CulturalProfile,
        intent: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Adapt message with sophisticated cultural intelligence."""
        
        try:
            # Check adaptation cache
            cache_key = f"adaptation:{hash(message)}:{cultural_profile.nationality}:{intent}"
            
            if self.redis_client:
                cached_adaptation = await self.redis_client.get(cache_key)
                if cached_adaptation:
                    return json.loads(cached_adaptation)
            
            # Use Instructor for sophisticated cultural adaptation
            cultural_response = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=CulturalResponse,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_cultural_adaptation_prompt(cultural_profile)
                    },
                    {
                        "role": "user",
                        "content": f"Adapt this message culturally:\n\nMessage: {message}\nIntent: {intent}\nContext: {context or {}}"
                    }
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            # Enhance with cultural formatting
            adapted_message = await self._enhance_cultural_formatting(
                cultural_response.message,
                cultural_profile,
                cultural_response.cultural_markers
            )
            
            # Build comprehensive adaptation result
            adaptation_result = {
                "adapted_message": adapted_message,
                "cultural_markers": cultural_response.cultural_markers,
                "tone_adjustments": cultural_response.tone_adjustments,
                "formality_level": cultural_response.formality_level,
                "cultural_elements": await self._get_cultural_elements(cultural_profile),
                "communication_style": cultural_profile.communication_style.value,
                "religious_considerations": cultural_profile.religious_considerations.dict(),
                "adaptation_confidence": cultural_response.confidence_score
            }
            
            # Cache the adaptation
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    3600,  # 1 hour cache for adaptations
                    json.dumps(adaptation_result, default=str)
                )
            
            self.cultural_metrics["adaptations_performed"] += 1
            return adaptation_result
            
        except Exception as e:
            logger.error(f"Cultural message adaptation failed: {e}")
            return {"adapted_message": message, "cultural_markers": []}
    
    async def _detect_language_advanced(self, text: str) -> Tuple[Language, float]:
        """Advanced language detection with confidence scoring."""
        
        try:
            # Clean text for better detection
            cleaned_text = re.sub(r'[^\w\s]', ' ', text).strip()
            
            if len(cleaned_text) < 10:
                # Default to English for short messages
                return Language.ENGLISH, 0.5
            
            # Detect language
            detected_lang = detect(cleaned_text)
            confidence = 0.8  # Base confidence
            
            # Map to our Language enum
            language_mapping = {
                'ar': Language.ARABIC,
                'fr': Language.FRENCH,
                'en': Language.ENGLISH,
                'es': Language.SPANISH
            }
            
            language = language_mapping.get(detected_lang, Language.ENGLISH)
            
            # Adjust confidence based on text characteristics
            if any(char in text for char in 'أبتثجحخدذرزسشصضطظعغفقكلمنهوي'):
                language = Language.ARABIC
                confidence = 0.95
            elif any(word in text.lower() for word in ['bonjour', 'merci', 'salut', 'bonsoir']):
                language = Language.FRENCH
                confidence = 0.9
            elif any(word in text.lower() for word in ['hello', 'thanks', 'please', 'good']):
                language = Language.ENGLISH
                confidence = 0.85
            
            return language, confidence
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return Language.ENGLISH, 0.3
    
    async def _infer_nationality_sophisticated(
        self,
        message_text: str,
        phone_number: str,
        guest_profile: Optional[GuestProfile] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """Sophisticated nationality inference using multiple signals."""
        
        try:
            confidence_scores = {}
            
            # Phone number analysis
            phone_nationality, phone_confidence = self._infer_from_phone_number(phone_number)
            if phone_nationality:
                confidence_scores[phone_nationality] = phone_confidence * 0.4
            
            # Guest profile analysis
            if guest_profile and guest_profile.nationality:
                confidence_scores[guest_profile.nationality] = 0.8
            
            # Message content analysis
            content_nationality, content_confidence = await self._infer_from_message_content(message_text)
            if content_nationality:
                existing_score = confidence_scores.get(content_nationality, 0)
                confidence_scores[content_nationality] = max(existing_score, content_confidence * 0.3)
            
            # Additional context analysis
            if additional_context:
                context_nationality, context_confidence = self._infer_from_context(additional_context)
                if context_nationality:
                    existing_score = confidence_scores.get(context_nationality, 0)
                    confidence_scores[context_nationality] = max(existing_score, context_confidence * 0.2)
            
            # Select best nationality
            if confidence_scores:
                best_nationality = max(confidence_scores, key=confidence_scores.get)
                best_confidence = confidence_scores[best_nationality]
                return best_nationality, min(best_confidence, 1.0)
            
            # Default fallback
            return "International", 0.3
            
        except Exception as e:
            logger.error(f"Nationality inference failed: {e}")
            return "International", 0.2
    
    def _initialize_cultural_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive cultural database."""
        
        return {
            "Morocco": {
                "communication_style": CommunicationStyle.HIGH_CONTEXT,
                "hospitality_values": ["warmth", "generosity", "respect", "family_focus"],
                "religious_considerations": {
                    "primary_religion": "Islam",
                    "dietary": ["halal"],
                    "prayer_times": True,
                    "religious_holidays": ["Ramadan", "Eid al-Fitr", "Eid al-Adha"]
                },
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.7,
                    CulturalDimension.INDIVIDUALISM: 0.3,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6
                }
            },
            "France": {
                "communication_style": CommunicationStyle.DIRECT,
                "hospitality_values": ["sophistication", "quality", "cultural_appreciation"],
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.6,
                    CulturalDimension.INDIVIDUALISM: 0.7,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.8
                }
            },
            "United States": {
                "communication_style": CommunicationStyle.LOW_CONTEXT,
                "hospitality_values": ["efficiency", "friendliness", "personalization"],
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.4,
                    CulturalDimension.INDIVIDUALISM: 0.9,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.5
                }
            },
            "United Kingdom": {
                "communication_style": CommunicationStyle.INDIRECT,
                "hospitality_values": ["politeness", "tradition", "understatement"],
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.35,
                    CulturalDimension.INDIVIDUALISM: 0.89,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.35
                }
            },
            "Germany": {
                "communication_style": CommunicationStyle.DIRECT,
                "hospitality_values": ["efficiency", "quality", "punctuality"],
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.35,
                    CulturalDimension.INDIVIDUALISM: 0.67,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.65
                }
            },
            "Japan": {
                "communication_style": CommunicationStyle.HIGH_CONTEXT,
                "hospitality_values": ["respect", "harmony", "attention_to_detail"],
                "cultural_dimensions": {
                    CulturalDimension.POWER_DISTANCE: 0.54,
                    CulturalDimension.INDIVIDUALISM: 0.46,
                    CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.92
                }
            }
        }
    
    def _get_cultural_adaptation_prompt(self, cultural_profile: CulturalProfile) -> str:
        """Get sophisticated cultural adaptation prompt."""
        
        cultural_data = self.cultural_database.get(cultural_profile.nationality, {})
        
        return f"""You are an expert in cross-cultural communication and Moroccan hospitality.
        
        Guest Cultural Profile:
        - Nationality: {cultural_profile.nationality}
        - Language: {cultural_profile.language.value}
        - Communication Style: {cultural_profile.communication_style.value}
        - Religious Considerations: {cultural_profile.religious_considerations.dict()}
        - Hospitality Expectations: {cultural_profile.hospitality_expectations}
        
        Adapt messages to be culturally appropriate while maintaining:
        1. Authentic Moroccan hospitality warmth
        2. Respect for cultural and religious sensitivities
        3. Appropriate formality level and communication style
        4. Cultural markers that resonate with the guest's background
        5. Service excellence aligned with cultural expectations
        
        Focus on creating genuine connections while honoring both Moroccan traditions
        and the guest's cultural background."""
    
    async def cleanup(self) -> None:
        """Cleanup cultural service resources."""
        
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Cultural service cleanup completed")
            
        except Exception as e:
            logger.error(f"Cultural service cleanup failed: {e}")
    
    # Additional sophisticated methods would continue here...
    # Including phone number analysis, message content analysis, cultural formatting, etc.
    
    def _infer_from_phone_number(self, phone_number: str) -> Tuple[Optional[str], float]:
        """Infer nationality from phone number country code."""
        
        try:
            # Extract country code
            cleaned_number = ''.join(c for c in phone_number if c.isdigit() or c == '+')
            
            if cleaned_number.startswith('+212') or cleaned_number.startswith('212'):
                return "Morocco", 0.9
            elif cleaned_number.startswith('+33') or cleaned_number.startswith('33'):
                return "France", 0.8
            elif cleaned_number.startswith('+1'):
                return "United States", 0.6  # Could be Canada too
            elif cleaned_number.startswith('+44'):
                return "United Kingdom", 0.8
            elif cleaned_number.startswith('+49'):
                return "Germany", 0.8
            elif cleaned_number.startswith('+81'):
                return "Japan", 0.8
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Phone number nationality inference failed: {e}")
            return None, 0.0
