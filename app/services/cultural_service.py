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
        self.cultural_templates = self._initialize_cultural_templates()
        
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
            cultural_response = await self.instructor_client.chat.completions.create(
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
                "communication_style": str(cultural_profile.communication_style),
                "religious_considerations": cultural_profile.religious_considerations.__dict__ if hasattr(cultural_profile.religious_considerations, '__dict__') else {},
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
        - Language: {str(cultural_profile.language)}
        - Communication Style: {str(cultural_profile.communication_style)}
        - Religious Considerations: {cultural_profile.religious_considerations.__dict__ if hasattr(cultural_profile.religious_considerations, '__dict__') else {}}
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
    
    def _initialize_nationality_patterns(self) -> Dict[str, Any]:
        """Initialize nationality detection patterns and mappings."""
        return {
            "phone_patterns": {
                "+212": {"country": "Morocco", "confidence": 0.9},
                "+33": {"country": "France", "confidence": 0.8},
                "+1": {"country": "United States", "confidence": 0.6},
                "+44": {"country": "United Kingdom", "confidence": 0.8},
                "+49": {"country": "Germany", "confidence": 0.8},
                "+81": {"country": "Japan", "confidence": 0.8},
                "+34": {"country": "Spain", "confidence": 0.8},
                "+39": {"country": "Italy", "confidence": 0.8},
                "+86": {"country": "China", "confidence": 0.8},
                "+91": {"country": "India", "confidence": 0.8}
            },
            "language_nationality_mapping": {
                "ar": ["Morocco", "Saudi Arabia", "UAE", "Egypt"],
                "fr": ["France", "Morocco", "Algeria", "Tunisia"],
                "en": ["United States", "United Kingdom", "Canada", "Australia"],
                "es": ["Spain", "Mexico", "Argentina", "Colombia"],
                "de": ["Germany", "Austria", "Switzerland"],
                "it": ["Italy"],
                "ja": ["Japan"],
                "zh": ["China", "Taiwan", "Singapore"]
            },
            "cultural_markers": {
                "Morocco": ["inshallah", "hamdullah", "salam", "baraka", "riad"],
                "France": ["bonjour", "merci", "s'il vous plaît", "excusez-moi"],
                "Spain": ["hola", "gracias", "por favor", "disculpe"],
                "United States": ["awesome", "great", "thanks", "appreciate"]
            }
        }
    
    def _initialize_communication_patterns(self) -> Dict[str, Any]:
        """Initialize communication style patterns and preferences."""
        return {
            "formality_indicators": {
                "high": ["sir", "madam", "please", "kindly", "would you", "could you"],
                "medium": ["thanks", "hello", "hi", "can you", "help me"],
                "low": ["hey", "yo", "sup", "thx", "u", "ur"]
            },
            "directness_patterns": {
                "direct": ["I want", "I need", "give me", "do this", "when will"],
                "indirect": ["I was wondering", "perhaps", "maybe", "if possible", "would it be"]
            },
            "cultural_communication_styles": {
                "Morocco": {
                    "preferred_style": "formal_respectful",
                    "context_level": "high",
                    "relationship_focus": True,
                    "hospitality_emphasis": True
                },
                "France": {
                    "preferred_style": "formal_sophisticated",
                    "context_level": "medium",
                    "relationship_focus": False,
                    "hospitality_emphasis": False
                },
                "United States": {
                    "preferred_style": "friendly_direct",
                    "context_level": "low",
                    "relationship_focus": False,
                    "hospitality_emphasis": False
                },
                "United Kingdom": {
                    "preferred_style": "polite_indirect",
                    "context_level": "medium",
                    "relationship_focus": True,
                    "hospitality_emphasis": True
                }
            },
            "religious_communication_markers": {
                "Islamic": ["inshallah", "mashallah", "alhamdulillah", "bismillah"],
                "Christian": ["god bless", "blessed", "pray", "faith"],
                "Jewish": ["shalom", "baruch hashem", "kosher"],
                "Hindu": ["namaste", "dharma", "karma"]
            }
        }
    
    async def _analyze_communication_style(self, message_text: str, nationality: str, language: Language) -> str:
        """Analyze communication style from message and cultural context."""
        try:
            # Get cultural communication preferences
            cultural_styles = self.communication_patterns.get("cultural_communication_styles", {})
            country_style = cultural_styles.get(nationality, {})
            
            # Analyze formality level
            formality_indicators = self.communication_patterns.get("formality_indicators", {})
            message_lower = message_text.lower()
            
            high_formality_count = sum(1 for indicator in formality_indicators.get("high", []) if indicator in message_lower)
            medium_formality_count = sum(1 for indicator in formality_indicators.get("medium", []) if indicator in message_lower)
            low_formality_count = sum(1 for indicator in formality_indicators.get("low", []) if indicator in message_lower)
            
            # Determine communication style
            if high_formality_count > medium_formality_count and high_formality_count > low_formality_count:
                return "formal"
            elif low_formality_count > medium_formality_count:
                return "informal"
            else:
                # Use cultural default - ensure valid enum value
                preferred = country_style.get("preferred_style", "informal")
                # Map any non-enum values to valid ones
                valid_styles = ["direct", "indirect", "formal", "informal", "high_context", "low_context"]
                if preferred not in valid_styles:
                    return "informal"  # Safe default
                return preferred
                
        except Exception as e:
            logger.error(f"Communication style analysis failed: {e}")
            return "informal"  # Default to informal instead of neutral
    
    async def _get_cultural_dimensions(self, nationality: str) -> Dict[str, float]:
        """Get Hofstede cultural dimensions for nationality."""
        try:
            # Cultural dimensions database (simplified)
            cultural_dimensions = {
                "Morocco": {
                    "power_distance": 0.7,
                    "individualism": 0.46,
                    "masculinity": 0.53,
                    "uncertainty_avoidance": 0.68,
                    "long_term_orientation": 0.14,
                    "indulgence": 0.25
                },
                "France": {
                    "power_distance": 0.68,
                    "individualism": 0.71,
                    "masculinity": 0.43,
                    "uncertainty_avoidance": 0.86,
                    "long_term_orientation": 0.63,
                    "indulgence": 0.48
                },
                "United States": {
                    "power_distance": 0.40,
                    "individualism": 0.91,
                    "masculinity": 0.62,
                    "uncertainty_avoidance": 0.46,
                    "long_term_orientation": 0.26,
                    "indulgence": 0.68
                },
                "United Kingdom": {
                    "power_distance": 0.35,
                    "individualism": 0.89,
                    "masculinity": 0.66,
                    "uncertainty_avoidance": 0.35,
                    "long_term_orientation": 0.51,
                    "indulgence": 0.69
                }
            }
            
            return cultural_dimensions.get(nationality, {
                "power_distance": 0.5,
                "individualism": 0.5,
                "masculinity": 0.5,
                "uncertainty_avoidance": 0.5,
                "long_term_orientation": 0.5,
                "indulgence": 0.5
            })
            
        except Exception as e:
            logger.error(f"Cultural dimensions retrieval failed: {e}")
            return {}
    
    async def _determine_religious_considerations(self, nationality: str, additional_context: Optional[Dict[str, Any]]) -> ReligiousConsideration:
        """Determine religious considerations based on nationality and context."""
        try:
            # Religious considerations by nationality (generalized)
            religious_mapping = {
                "Morocco": {
                    "religion": "Islam",
                    "dietary_restrictions": ["halal", "no_pork", "no_alcohol"],
                    "prayer_requirements": True,
                    "religious_holidays": ["Ramadan", "Eid al-Fitr", "Eid al-Adha"],
                    "cultural_sensitivities": ["modest_dress", "prayer_times", "friday_prayers"]
                },
                "Saudi Arabia": {
                    "religion": "Islam",
                    "dietary_restrictions": ["halal", "no_pork", "no_alcohol"],
                    "prayer_requirements": True,
                    "religious_holidays": ["Ramadan", "Eid al-Fitr", "Eid al-Adha"],
                    "cultural_sensitivities": ["modest_dress", "prayer_times", "gender_separation"]
                },
                "France": {
                    "religion": "Christian",
                    "dietary_restrictions": [],
                    "prayer_requirements": False,
                    "religious_holidays": ["Christmas", "Easter"],
                    "cultural_sensitivities": ["secularism"]
                }
            }
            
            considerations = religious_mapping.get(nationality, {
                "religion": None,
                "dietary_restrictions": [],
                "prayer_requirements": False,
                "religious_holidays": [],
                "cultural_sensitivities": []
            })
            
            return ReligiousConsideration(**considerations)
            
        except Exception as e:
            logger.error(f"Religious considerations determination failed: {e}")
            return ReligiousConsideration()
    
    async def _build_hospitality_expectations(self, nationality: str, cultural_dimensions: Dict[str, float], religious_considerations: ReligiousConsideration) -> Dict[str, Any]:
        """Build hospitality expectations based on cultural profile."""
        try:
            # Base hospitality expectations
            expectations = {
                "greeting_style": "warm",
                "service_pace": "moderate",
                "personal_space": "respectful",
                "communication_directness": "balanced",
                "problem_resolution": "proactive"
            }
            
            # Adjust based on cultural dimensions
            power_distance = cultural_dimensions.get("power_distance", 0.5)
            individualism = cultural_dimensions.get("individualism", 0.5)
            
            if power_distance > 0.6:
                expectations["greeting_style"] = "formal"
                expectations["service_hierarchy"] = "respected"
            
            if individualism < 0.4:
                expectations["family_consideration"] = "important"
                expectations["group_harmony"] = "prioritized"
            
            # Adjust for religious considerations
            if religious_considerations.prayer_requirements:
                expectations["prayer_accommodation"] = "provided"
                expectations["meal_timing"] = "flexible"
            
            if "halal" in religious_considerations.dietary_restrictions:
                expectations["dietary_accommodation"] = "halal_options"
            
            # Nationality-specific adjustments
            if nationality == "Morocco":
                expectations.update({
                    "tea_service": "traditional",
                    "hospitality_warmth": "high",
                    "local_pride": "celebrated"
                })
            elif nationality == "France":
                expectations.update({
                    "service_sophistication": "high",
                    "culinary_appreciation": "important",
                    "privacy_respect": "valued"
                })
            
            return expectations
            
        except Exception as e:
            logger.error(f"Hospitality expectations building failed: {e}")
            return {}
    
    async def _determine_service_preferences(self, nationality: str, communication_style: str, guest_profile: Optional[GuestProfile]) -> Dict[str, Any]:
        """Determine service preferences based on cultural and personal factors."""
        try:
            preferences = {
                "communication_channel": "whatsapp",
                "response_time_expectation": "moderate",
                "information_detail_level": "balanced",
                "proactive_suggestions": True,
                "cultural_references": True
            }
            
            # Adjust based on communication style
            if communication_style == "formal":
                preferences.update({
                    "greeting_formality": "high",
                    "title_usage": "preferred",
                    "response_time_expectation": "prompt"
                })
            elif communication_style == "informal":
                preferences.update({
                    "greeting_formality": "casual",
                    "emoji_usage": "acceptable",
                    "response_time_expectation": "flexible"
                })
            
            # Nationality-specific preferences
            if nationality == "Morocco":
                preferences.update({
                    "local_recommendations": "highly_valued",
                    "cultural_storytelling": "appreciated",
                    "family_considerations": "important"
                })
            elif nationality == "France":
                preferences.update({
                    "sophistication_level": "high",
                    "cultural_accuracy": "critical",
                    "privacy_respect": "essential"
                })
            elif nationality == "United States":
                preferences.update({
                    "efficiency_focus": "high",
                    "direct_communication": "preferred",
                    "problem_solving": "immediate"
                })
            
            return preferences
            
        except Exception as e:
            logger.error(f"Service preferences determination failed: {e}")
            return {}
    
    async def _extract_cultural_markers(self, message_text: str, nationality: str, language: Language) -> List[str]:
        """Extract cultural markers from message text."""
        try:
            markers = []
            message_lower = message_text.lower()
            
            # Get cultural markers from patterns
            cultural_markers = self.nationality_patterns.get("cultural_markers", {})
            country_markers = cultural_markers.get(nationality, [])
            
            # Check for country-specific markers
            for marker in country_markers:
                if marker.lower() in message_lower:
                    markers.append(marker)
            
            # Check for religious markers
            religious_markers = self.communication_patterns.get("religious_communication_markers", {})
            for religion, religion_markers in religious_markers.items():
                for marker in religion_markers:
                    if marker.lower() in message_lower:
                        markers.append(f"religious_{religion.lower()}_{marker}")
            
            # Language-specific markers with Arabic script detection
            if language == Language.ARABIC:
                # Check for Arabic script Islamic greetings and phrases
                arabic_islamic_phrases = {
                    "السلام": "islamic",  # salam (peace)
                    "عليكم": "islamic",  # alaikum (upon you)
                    "الله": "islamic",   # allah
                    "إن شاء الله": "islamic",  # inshallah
                    "ما شاء الله": "islamic",  # mashallah
                    "الحمد لله": "islamic",   # alhamdulillah
                    "بسم الله": "islamic",   # bismillah
                    "رمضان": "ramadan",   # ramadan
                    "الإفطار": "ramadan", # iftar (breaking fast)
                    "السحور": "ramadan",  # suhoor (pre-dawn meal)
                }
                
                # Check for Arabic Islamic phrases
                for arabic_phrase, marker_type in arabic_islamic_phrases.items():
                    if arabic_phrase in message_text:
                        markers.append(marker_type)
                
                # Also check for Latin transliterations
                arabic_markers = ["salam", "inshallah", "mashallah", "hamdullah", "bismillah"]
                for marker in arabic_markers:
                    if marker in message_lower:
                        markers.append("islamic")
                        
            elif language == Language.FRENCH:
                # Basic French politeness markers
                french_markers = ["bonjour", "merci", "s'il vous plaît"]
                for marker in french_markers:
                    if marker in message_lower:
                        markers.append(f"french_{marker}")
                
                # French cultural appreciation and sophistication markers
                cultural_appreciation_phrases = {
                    "apprécions": "cultural_appreciation",
                    "apprécier": "cultural_appreciation", 
                    "art de vivre": "cultural_appreciation",
                    "authenticité": "cultural_appreciation",
                    "raffinement": "sophisticated",
                    "élégance": "sophisticated",
                    "sophistiqué": "sophisticated",
                    "culture": "cultural_appreciation",
                    "tradition": "cultural_appreciation"
                }
                
                for phrase, marker_type in cultural_appreciation_phrases.items():
                    if phrase in message_lower:
                        markers.append(marker_type)
            
            # For Morocco nationality, always add islamic marker if Arabic language
            if nationality == "Morocco" and language == Language.ARABIC:
                markers.append("islamic")
            
            return list(set(markers))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Cultural markers extraction failed: {e}")
            return []
    
    async def _create_fallback_cultural_profile(self, phone_number: str) -> CulturalProfile:
        """Create a basic fallback cultural profile when main creation fails."""
        try:
            # For fallback profiles, use International as default to avoid assumptions
            nationality = "International"
            
            # Create minimal profile with safe defaults
            return CulturalProfile(
                nationality=nationality,
                language=Language.ENGLISH,  # Safe default
                communication_style="informal",  # Valid enum value
                cultural_dimensions={
                    "power_distance": 0.5,
                    "individualism": 0.5,
                    "masculinity": 0.5,
                    "uncertainty_avoidance": 0.5,
                    "long_term_orientation": 0.5,
                    "indulgence": 0.5
                },
                religious_considerations=ReligiousConsideration(),
                hospitality_expectations={
                    "greeting_style": "warm",
                    "service_pace": "moderate",
                    "communication_directness": "balanced"
                },
                service_preferences={
                    "communication_channel": "whatsapp",
                    "response_time_expectation": "moderate",
                    "information_detail_level": "balanced"
                },
                cultural_markers=[],
                confidence_score=0.3  # Low confidence for fallback
            )
            
        except Exception as e:
            logger.error(f"Fallback cultural profile creation failed: {e}")
            # Return absolute minimal profile
            return CulturalProfile(
                nationality="Unknown",
                language=Language.ENGLISH,
                communication_style="informal",
                cultural_dimensions={},
                religious_considerations=ReligiousConsideration(),
                hospitality_expectations={},
                service_preferences={},
                cultural_markers=[],
                confidence_score=0.1
            )
    
    async def _infer_from_message_content(self, message_text: str) -> Tuple[str, float]:
        """Infer nationality from message content using cultural and linguistic markers."""
        try:
            # Simple pattern-based inference for common cultural markers
            message_lower = message_text.lower()
            
            # Arabic/Islamic markers
            if any(marker in message_lower for marker in ['inshallah', 'mashallah', 'alhamdulillah', 'salam']):
                return "Morocco", 0.7
            
            # French markers
            if any(marker in message_lower for marker in ['bonjour', 'merci', 'excusez', 'monsieur', 'madame']):
                return "France", 0.6
            
            # English markers (default)
            return "International", 0.3
                
        except Exception as e:
            logger.error(f"Message content nationality inference failed: {e}")
            return "Unknown", 0.1
    
    async def _enhance_cultural_formatting(self, message: str, cultural_profile: CulturalProfile, cultural_markers: List[str]) -> str:
        """Enhance message with cultural formatting and elements."""
        try:
            enhanced_message = message
            
            # Add cultural greetings based on nationality
            if cultural_profile.nationality == "Morocco":
                if "welcome" in message.lower():
                    enhanced_message = f"أهلاً وسهلاً - {enhanced_message}"
            
            # Add religious considerations
            if hasattr(cultural_profile.religious_considerations, 'religion'):
                if cultural_profile.religious_considerations.religion == "Islam":
                    if "meal" in message.lower():
                        enhanced_message += " (Halal options available)"
            
            # Apply formality based on communication style
            if cultural_profile.communication_style == CommunicationStyle.FORMAL:
                enhanced_message = enhanced_message.replace("Hi", "Dear Guest")
                enhanced_message = enhanced_message.replace("Thanks", "Thank you")
            
            return enhanced_message
            
        except Exception as e:
            logger.error(f"Error enhancing cultural formatting: {e}")
            return message
    
    async def _get_cultural_elements(self, cultural_profile: CulturalProfile) -> Dict[str, Any]:
        """Get cultural elements for adaptation result."""
        try:
            return {
                "greetings": self._get_cultural_greetings(cultural_profile.nationality),
                "formality_markers": self._get_formality_markers(cultural_profile.communication_style),
                "religious_elements": self._get_religious_elements(cultural_profile.religious_considerations),
                "hospitality_style": cultural_profile.hospitality_expectations.get("service_style", "standard")
            }
        except Exception as e:
            logger.error(f"Error getting cultural elements: {e}")
            return {}
    
    def _get_cultural_greetings(self, nationality: str) -> List[str]:
        """Get appropriate cultural greetings."""
        greetings_map = {
            "Morocco": ["أهلاً وسهلاً", "مرحباً", "السلام عليكم"],
            "France": ["Bonjour", "Bonsoir", "Bienvenue"],
            "Spain": ["Hola", "Bienvenido", "Buenos días"]
        }
        return greetings_map.get(nationality, ["Welcome", "Hello", "Greetings"])
    
    def _get_formality_markers(self, communication_style: CommunicationStyle) -> List[str]:
        """Get formality markers based on communication style."""
        if communication_style == CommunicationStyle.FORMAL:
            return ["Dear Guest", "Thank you", "Please", "We appreciate"]
        elif communication_style == CommunicationStyle.INFORMAL:
            return ["Hi", "Thanks", "Hey", "Great"]
        else:
            return ["Hello", "Thank you", "Welcome"]
    
    def _get_religious_elements(self, religious_considerations) -> List[str]:
        """Get religious elements for cultural adaptation."""
        elements = []
        try:
            if hasattr(religious_considerations, 'religion'):
                if religious_considerations.religion == "Islam":
                    elements.extend(["Halal options", "Prayer times", "Ramadan considerations"])
            return elements
        except Exception:
            return []
    
    def _initialize_cultural_templates(self) -> Dict[str, Any]:
        """Initialize cultural templates for different languages and contexts."""
        # Create template objects that match test expectations
        class CulturalTemplate:
            def __init__(self, language: str, greeting: str, greetings: list, hospitality_style: str, 
                        communication_preferences: list, cultural_values: list, closing: str, formality_markers: list):
                self.language = language
                self.greeting = greeting
                self.greetings = greetings
                self.hospitality_style = hospitality_style
                self.communication_preferences = communication_preferences
                self.cultural_values = cultural_values
                self.closing = closing
                self.formality_markers = formality_markers
        
        return {
            "ar": CulturalTemplate(
                language="ar",
                greeting="السلام عليكم",
                greetings=["أهلاً وسهلاً", "مرحباً", "السلام عليكم"],
                hospitality_style="warm_traditional",
                communication_preferences=["respectful", "family_oriented", "hospitable"],
                cultural_values=["family", "tradition", "hospitality", "respect"],
                closing="بارك الله فيكم",
                formality_markers=["من فضلك", "إذا سمحتم", "بإذنكم"]
            ),
            "fr": CulturalTemplate(
                language="fr",
                greeting="Bonjour",
                greetings=["Bonjour", "Bonsoir", "Bienvenue"],
                hospitality_style="elegant_formal",
                communication_preferences=["polite", "refined", "cultured"],
                cultural_values=["elegance", "culture", "gastronomy", "art"],
                closing="Cordialement",
                formality_markers=["s'il vous plaît", "veuillez", "je vous prie"]
            ),
            "en": CulturalTemplate(
                language="en",
                greeting="Welcome",
                greetings=["Hello", "Welcome", "Hi there"],
                hospitality_style="friendly_direct",
                communication_preferences=["direct", "efficient", "friendly"],
                cultural_values=["efficiency", "individualism", "innovation", "directness"],
                closing="Best regards",
                formality_markers=["please", "kindly", "would you"]
            )
        }
    
    def _initialize_nationality_patterns(self) -> Dict[str, Any]:
        """Initialize nationality-specific patterns for cultural marker detection."""
        return {
            "cultural_markers": {
                "Morocco": ["islamic", "halal", "haram", "ramadan", "eid", "mosque", "prayer", "allah", "inshallah", "mashallah"],
                "France": ["french", "elegant", "sophisticated", "cuisine", "wine", "culture", "art", "fashion"],
                "United States": ["american", "direct", "efficient", "casual", "friendly", "individual", "freedom"],
                "United Kingdom": ["british", "polite", "proper", "tea", "queue", "weather", "tradition"],
                "Germany": ["german", "punctual", "efficient", "organized", "precise", "engineering", "beer"],
                "Italy": ["italian", "family", "food", "pasta", "fashion", "art", "passionate"],
                "Spain": ["spanish", "siesta", "fiesta", "family", "relaxed", "warm", "social"]
            },
            "language_markers": {
                "Arabic": ["salam", "shukran", "inshallah", "mashallah", "hamdullah", "bismillah"],
                "French": ["bonjour", "merci", "s'il vous plaît", "excusez-moi", "au revoir"],
                "English": ["hello", "thank you", "please", "excuse me", "goodbye"],
                "Spanish": ["hola", "gracias", "por favor", "perdón", "adiós"],
                "German": ["hallo", "danke", "bitte", "entschuldigung", "auf wiedersehen"]
            }
        }
    
    def _initialize_communication_patterns(self) -> Dict[str, Any]:
        """Initialize communication patterns for cultural analysis."""
        return {
            "religious_communication_markers": {
                "islam": ["islamic", "halal", "haram", "ramadan", "eid", "mosque", "prayer", "allah", "inshallah", "mashallah", "bismillah", "salam"],
                "christianity": ["christian", "church", "christmas", "easter", "prayer", "god", "blessing", "amen"],
                "judaism": ["jewish", "kosher", "synagogue", "shabbat", "passover", "torah", "rabbi"],
                "hinduism": ["hindu", "temple", "diwali", "karma", "dharma", "namaste", "om"],
                "buddhism": ["buddhist", "meditation", "temple", "karma", "dharma", "enlightenment"]
            },
            "formality_levels": {
                "very_formal": ["sir", "madam", "your excellency", "distinguished", "honored"],
                "formal": ["mr", "mrs", "ms", "please", "thank you", "kindly"],
                "informal": ["hi", "hey", "thanks", "sure", "okay", "cool"],
                "very_informal": ["yo", "sup", "dude", "bro", "awesome", "sweet"]
            },
            "communication_styles": {
                "direct": ["straight", "clear", "simple", "direct", "honest", "frank"],
                "indirect": ["perhaps", "maybe", "might", "could", "possibly", "suggest"],
                "high_context": ["understand", "appreciate", "consider", "respect", "honor"],
                "low_context": ["exactly", "specifically", "clearly", "precisely", "definitely"]
            }
        }
