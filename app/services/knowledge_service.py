"""
Advanced Knowledge Service with Hybrid CAG-RAG Architecture
Implements 70% static CAG knowledge and 30% dynamic RAG retrieval
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np

import openai
from pinecone import Pinecone
import redis.asyncio as redis
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.models.agent_state import CulturalContext, GuestProfile, RetrievalResult, Language
from app.models.instructor_models import KnowledgeRetrieval, get_instructor_client


class KnowledgeCategory(str, Enum):
    """Knowledge categories for sophisticated retrieval."""
    RIAD_SERVICES = "riad_services"
    LOCAL_ATTRACTIONS = "local_attractions"
    CULTURAL_INFORMATION = "cultural_information"
    DINING_RECOMMENDATIONS = "dining_recommendations"
    TRANSPORTATION = "transportation"
    EMERGENCY_INFO = "emergency_info"
    SEASONAL_ACTIVITIES = "seasonal_activities"
    RELIGIOUS_CONSIDERATIONS = "religious_considerations"
    SHOPPING_GUIDES = "shopping_guides"
    WELLNESS_SPA = "wellness_spa"


class KnowledgeSource(str, Enum):
    """Knowledge sources for provenance tracking."""
    STATIC_CAG = "static_cag"
    DYNAMIC_RAG = "dynamic_rag"
    REAL_TIME_API = "real_time_api"
    GUEST_HISTORY = "guest_history"
    STAFF_INSIGHTS = "staff_insights"
    CULTURAL_EXPERT = "cultural_expert"


class CulturalKnowledgeItem(BaseModel):
    """Culturally enriched knowledge item."""
    content: str
    category: KnowledgeCategory
    source: KnowledgeSource
    cultural_relevance: Dict[str, float]  # Language -> relevance score
    temporal_relevance: Optional[Dict[str, Any]] = None
    guest_personalization: Optional[Dict[str, Any]] = None
    confidence_score: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeService:
    """Sophisticated knowledge service with hybrid CAG-RAG architecture."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # AI clients
        self.openai_client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.instructor_client = get_instructor_client()
        
        # Vector database
        try:
            self.pinecone = Pinecone(api_key=self.settings.pinecone_api_key)
            self.index = self.pinecone.Index(self.settings.pinecone_index_name)
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}")
            self.index = None
        
        # Redis for caching
        self.redis_client: Optional[redis.Redis] = None
        
        # CAG Knowledge Cache (70% of intelligence)
        self.cag_knowledge_cache: Dict[str, List[CulturalKnowledgeItem]] = {}
        self.cultural_knowledge_map: Dict[str, Dict[str, Any]] = {}
        
        # RAG Configuration (30% of intelligence)
        self.rag_config = {
            "embedding_model": self.settings.openai_embedding_model,
            "top_k_results": 5,
            "similarity_threshold": 0.7,
            "cultural_boost_factor": 0.3,
            "temporal_decay_factor": 0.1
        }
        
        # Knowledge freshness tracking
        self.knowledge_freshness: Dict[str, datetime] = {}
        self.cache_ttl = timedelta(hours=6)
        
        # Performance metrics
        self.retrieval_metrics = {
            "cag_hits": 0,
            "rag_queries": 0,
            "cache_hits": 0,
            "avg_response_time": 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize knowledge service with caches and connections."""
        
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                password=self.settings.redis_password,
                decode_responses=True
            )
            await self.redis_client.ping()
            
            # Load CAG knowledge cache
            await self._load_cag_knowledge_cache()
            
            # Initialize cultural knowledge mapping
            await self._initialize_cultural_knowledge_map()
            
            # Warm up embeddings cache
            await self._warm_up_embeddings_cache()
            
            logger.info("✅ Advanced Knowledge Service initialized")
            
        except Exception as e:
            logger.error(f"❌ Knowledge service initialization failed: {e}")
            raise
    
    async def get_hybrid_knowledge(
        self,
        query: str,
        intent: str,
        cultural_context: CulturalContext,
        guest_profile: Optional[GuestProfile] = None,
        category_filter: Optional[List[KnowledgeCategory]] = None
    ) -> Tuple[Dict[str, Any], List[RetrievalResult]]:
        """Get knowledge using hybrid CAG-RAG approach (70% CAG, 30% RAG)."""
        
        start_time = datetime.now()
        
        try:
            # Use Instructor for structured knowledge retrieval planning
            knowledge_plan = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=KnowledgeRetrieval,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_knowledge_planning_prompt()
                    },
                    {
                        "role": "user", 
                        "content": f"Plan knowledge retrieval for: {query}\nIntent: {intent}\nCulture: {cultural_context.nationality}"
                    }
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            # Execute hybrid retrieval strategy
            cag_knowledge = await self._get_cag_knowledge_intelligent(
                query=query,
                intent=intent,
                cultural_context=cultural_context,
                knowledge_plan=knowledge_plan,
                weight=0.7  # 70% CAG
            )
            
            rag_results = await self._get_rag_knowledge_intelligent(
                query=query,
                cultural_context=cultural_context,
                guest_profile=guest_profile,
                knowledge_plan=knowledge_plan,
                weight=0.3  # 30% RAG
            )
            
            # Apply cultural intelligence boost
            cag_knowledge = await self._apply_cultural_intelligence_boost(
                cag_knowledge, cultural_context
            )
            
            rag_results = await self._apply_cultural_relevance_scoring(
                rag_results, cultural_context
            )
            
            # Update metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            await self._update_retrieval_metrics(processing_time)
            
            logger.info(f"Hybrid knowledge retrieved in {processing_time:.3f}s")
            
            return cag_knowledge, rag_results
            
        except Exception as e:
            logger.error(f"Hybrid knowledge retrieval failed: {e}")
            return await self._get_fallback_knowledge(), []
    
    async def _get_cag_knowledge_intelligent(
        self,
        query: str,
        intent: str,
        cultural_context: CulturalContext,
        knowledge_plan: KnowledgeRetrieval,
        weight: float = 0.7
    ) -> Dict[str, Any]:
        """Get CAG knowledge with intelligent cultural adaptation."""
        
        try:
            # Check cache first
            cache_key = f"cag:{hash(query)}:{cultural_context.language.value}:{intent}"
            
            if self.redis_client:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    self.retrieval_metrics["cache_hits"] += 1
                    return json.loads(cached_result)
            
            # Build sophisticated CAG knowledge
            cag_knowledge = {
                "riad_expertise": await self._get_riad_expertise(cultural_context, intent),
                "local_intelligence": await self._get_local_intelligence(cultural_context, query),
                "cultural_insights": await self._get_cultural_insights(cultural_context),
                "seasonal_information": await self._get_seasonal_information(),
                "personalization_data": await self._get_personalization_data(intent)
            }
            
            # Apply knowledge plan filtering
            if knowledge_plan.relevant_knowledge:
                cag_knowledge = self._filter_knowledge_by_plan(
                    cag_knowledge, knowledge_plan
                )
            
            # Apply cultural weighting
            cag_knowledge = await self._apply_cultural_weighting(
                cag_knowledge, cultural_context, weight
            )
            
            # Cache result
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key, 
                    self.settings.cache_ttl, 
                    json.dumps(cag_knowledge, default=str)
                )
            
            self.retrieval_metrics["cag_hits"] += 1
            return cag_knowledge
            
        except Exception as e:
            logger.error(f"CAG knowledge retrieval failed: {e}")
            return await self._get_fallback_cag_knowledge()
    
    async def _get_rag_knowledge_intelligent(
        self,
        query: str,
        cultural_context: CulturalContext,
        guest_profile: Optional[GuestProfile],
        knowledge_plan: KnowledgeRetrieval,
        weight: float = 0.3
    ) -> List[RetrievalResult]:
        """Get RAG knowledge with intelligent personalization."""
        
        try:
            if not self.index:
                logger.warning("Pinecone not available for RAG")
                return []
            
            # Generate culturally-aware embedding
            embedding = await self._generate_cultural_embedding(query, cultural_context)
            
            # Build sophisticated metadata filter
            metadata_filter = await self._build_intelligent_metadata_filter(
                cultural_context, guest_profile, knowledge_plan
            )
            
            # Query with cultural boosting
            results = self.index.query(
                vector=embedding,
                top_k=self.rag_config["top_k_results"],
                include_metadata=True,
                filter=metadata_filter
            )
            
            # Convert to RetrievalResult with cultural scoring
            retrieval_results = []
            for match in results.matches:
                cultural_relevance = await self._calculate_advanced_cultural_relevance(
                    match.metadata, cultural_context, guest_profile
                )
                
                result = RetrievalResult(
                    content=match.metadata.get("content", ""),
                    source=match.metadata.get("source", "vector_db"),
                    relevance_score=float(match.score),
                    metadata=match.metadata,
                    cultural_relevance=cultural_relevance
                )
                retrieval_results.append(result)
            
            # Apply temporal and personalization boosting
            retrieval_results = await self._apply_temporal_boosting(retrieval_results)
            retrieval_results = await self._apply_personalization_boosting(
                retrieval_results, guest_profile
            )
            
            # Sort by combined relevance score
            retrieval_results.sort(
                key=lambda x: (x.relevance_score * 0.6 + x.cultural_relevance * 0.4),
                reverse=True
            )
            
            self.retrieval_metrics["rag_queries"] += 1
            return retrieval_results
            
        except Exception as e:
            logger.error(f"RAG knowledge retrieval failed: {e}")
            return []
    
    async def _get_riad_expertise(
        self, 
        cultural_context: CulturalContext, 
        intent: str
    ) -> Dict[str, Any]:
        """Get sophisticated riad expertise knowledge."""
        
        base_expertise = {
            "property_information": {
                "name": "Riad Dar Al-Andalus",
                "type": "Traditional Moroccan Riad",
                "location": "Medina of Marrakech",
                "architecture": "18th century traditional",
                "rooms": {
                    "total": 12,
                    "types": ["Standard", "Deluxe", "Suite", "Family"],
                    "amenities": ["AC", "WiFi", "Private Bath", "Traditional Decor"]
                }
            },
            "services": {
                "concierge": "24/7 multilingual service",
                "spa": "Traditional hammam and wellness treatments",
                "dining": "Authentic Moroccan cuisine and international options",
                "activities": "Cultural tours, cooking classes, artisan workshops"
            },
            "unique_features": {
                "courtyard": "Central fountain with orange trees",
                "rooftop": "Atlas Mountains view terrace",
                "library": "Collection of Moroccan literature and guides",
                "art": "Local artisan crafts and traditional tilework"
            }
        }
        
        # Culturally adapt the expertise
        if cultural_context.language == Language.ARABIC:
            base_expertise["cultural_elements"] = {
                "islamic_features": "Prayer direction indicators in rooms",
                "halal_dining": "Certified halal kitchen and menu options",
                "cultural_respect": "Traditional Moroccan hospitality values",
                "family_focus": "Family-friendly spaces and activities"
            }
        elif cultural_context.language == Language.FRENCH:
            base_expertise["cultural_elements"] = {
                "french_connection": "Franco-Moroccan cultural bridge",
                "sophistication": "Refined traditional craftsmanship",
                "culinary_excellence": "French-influenced Moroccan cuisine",
                "artistic_heritage": "Blend of Moroccan and French aesthetics"
            }
        
        return base_expertise
    
    async def _get_local_intelligence(
        self, 
        cultural_context: CulturalContext, 
        query: str
    ) -> Dict[str, Any]:
        """Get sophisticated local intelligence."""
        
        local_intelligence = {
            "attractions": {
                "must_visit": [
                    {
                        "name": "Jemaa el-Fnaa",
                        "type": "Cultural Square",
                        "distance": "5 minutes walk",
                        "best_time": "Evening for storytellers and food stalls",
                        "cultural_significance": "Heart of Marrakech medina life"
                    },
                    {
                        "name": "Bahia Palace",
                        "type": "Historical Palace",
                        "distance": "10 minutes walk",
                        "best_time": "Morning to avoid crowds",
                        "cultural_significance": "19th century Moroccan architecture"
                    }
                ],
                "hidden_gems": [
                    {
                        "name": "Maison de la Photographie",
                        "type": "Cultural Museum",
                        "specialty": "Historical Morocco photography",
                        "insider_tip": "Rooftop cafe with medina views"
                    }
                ]
            },
            "dining": {
                "traditional": [
                    {
                        "name": "Dar Yacout",
                        "cuisine": "Traditional Moroccan",
                        "specialty": "Royal Moroccan feast experience",
                        "reservation": "Essential, book 2 days ahead"
                    }
                ],
                "contemporary": [
                    {
                        "name": "Le Jardin",
                        "cuisine": "Modern Moroccan-French",
                        "specialty": "Garden setting with modern twist",
                        "atmosphere": "Romantic courtyard dining"
                    }
                ]
            },
            "shopping": {
                "authentic_crafts": [
                    {
                        "name": "Ensemble Artisanal",
                        "type": "Government Artisan Center",
                        "advantage": "Fixed prices, authentic quality",
                        "specialties": ["Carpets", "Leather", "Ceramics", "Jewelry"]
                    }
                ],
                "souks": [
                    {
                        "name": "Souk Semmarine",
                        "type": "Traditional Market",
                        "tips": "Bargaining expected, start at 30% of asking price",
                        "best_items": ["Textiles", "Spices", "Traditional clothing"]
                    }
                ]
            }
        }
        
        # Add cultural-specific recommendations
        if cultural_context.language == Language.ARABIC:
            local_intelligence["religious_sites"] = [
                {
                    "name": "Kutubiyya Mosque",
                    "significance": "12th century architectural masterpiece",
                    "visitor_info": "Non-Muslims can admire exterior and gardens",
                    "prayer_times": "Five daily prayers, Friday special significance"
                }
            ]
        
        return local_intelligence
    
    def _get_knowledge_planning_prompt(self) -> str:
        """Get system prompt for knowledge retrieval planning."""
        
        return """You are an expert knowledge retrieval planner for a Moroccan riad hospitality system.
        
        Analyze the query and plan the most effective knowledge retrieval strategy:
        
        1. Identify what specific information is needed
        2. Determine the best knowledge sources (static vs dynamic)
        3. Consider cultural context and personalization needs
        4. Identify any gaps that require real-time data
        5. Plan fallback responses if knowledge is insufficient
        
        Focus on providing accurate, culturally appropriate, and personalized information
        that enhances the guest experience while respecting Moroccan hospitality traditions."""
    
    async def cleanup(self) -> None:
        """Cleanup knowledge service resources."""
        
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("✅ Knowledge service cleanup completed")
            
        except Exception as e:
            logger.error(f"Knowledge service cleanup failed: {e}")
    
    # Additional sophisticated methods would continue here...
    # Including cultural intelligence boosting, temporal relevance, personalization, etc.
    
    async def _generate_cultural_embedding(
        self, 
        text: str, 
        cultural_context: CulturalContext
    ) -> List[float]:
        """Generate culturally-aware embedding."""
        
        try:
            # Enhance text with cultural context
            cultural_text = f"{text} [Culture: {cultural_context.nationality}, Language: {cultural_context.language.value}]"
            
            response = await self.openai_client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=cultural_text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Cultural embedding generation failed: {e}")
            return [0.0] * 3072  # Default embedding size
    
    async def _calculate_advanced_cultural_relevance(
        self,
        metadata: Dict[str, Any],
        cultural_context: CulturalContext,
        guest_profile: Optional[GuestProfile]
    ) -> float:
        """Calculate sophisticated cultural relevance score."""
        
        relevance_score = 0.5  # Base score
        
        # Language match boost
        if metadata.get("language") == cultural_context.language.value:
            relevance_score += 0.3
        
        # Cultural context boost
        if cultural_context.nationality and metadata.get("cultural_context"):
            if cultural_context.nationality.lower() in str(metadata["cultural_context"]).lower():
                relevance_score += 0.2
        
        # Guest profile personalization
        if guest_profile:
            # Previous stays boost
            if guest_profile.previous_stays > 0:
                relevance_score += 0.1
            
            # Loyalty tier boost
            if guest_profile.loyalty_tier in ["premium", "vip"]:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
