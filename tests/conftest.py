"""
Pytest configuration and shared fixtures for Riad Concierge AI testing
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.models.agent_state import (
    GuestProfile, CulturalContext, Language, 
    BookingStatus, RoomType, AgentState
)
from app.models.instructor_models import (
    CulturalResponse, IntentClassificationResult, EmotionalAnalysis,
    KnowledgeRetrieval, ServiceAction, UpsellOpportunity
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Test configuration settings."""
    return {
        "environment": "test",
        "debug": True,
        "log_level": "DEBUG",
        "redis_url": "redis://localhost:6379/1",
        "database_url": "postgresql://test:test@localhost:5432/test_riad",
        "whatsapp_access_token": "test_token",
        "whatsapp_phone_number_id": "test_phone_id",
        "openai_api_key": "test_openai_key",
        "pinecone_api_key": "test_pinecone_key",
        "octorate_api_key": "test_octorate_key"
    }


@pytest.fixture
def test_client():
    """FastAPI test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client():
    """Async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.setex.return_value = True
    redis_mock.ping.return_value = True
    redis_mock.close = AsyncMock()  # Add close method for cleanup tests
    return redis_mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    client_mock = AsyncMock()
    
    # Mock embeddings
    client_mock.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    
    # Mock chat completions
    client_mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(
            message=MagicMock(content="Test response")
        )]
    )
    
    return client_mock


@pytest.fixture
def mock_pinecone_index():
    """Mock Pinecone index."""
    index_mock = MagicMock()
    index_mock.query.return_value = MagicMock(
        matches=[
            MagicMock(
                id="test_1",
                score=0.9,
                metadata={
                    "content": "Test knowledge content",
                    "source": "test_source",
                    "language": "en"
                }
            )
        ]
    )
    return index_mock


@pytest.fixture
def sample_guest_profile():
    """Sample guest profile for testing."""
    return GuestProfile(
        guest_id="test_guest_123",
        name="Ahmed Hassan",
        email="ahmed@example.com",
        phone="+212600123456",
        nationality="Morocco",
        language=Language.ARABIC,
        previous_stays=2,
        loyalty_tier="premium",
        preferences={
            "room_type": "suite",
            "dietary": ["halal"],
            "interests": ["culture", "history"]
        },
        special_requests=["late_checkout", "prayer_direction"]
    )


@pytest.fixture
def sample_cultural_context():
    """Sample cultural context for testing."""
    return CulturalContext(
        language=Language.ARABIC,
        nationality="Morocco",
        cultural_markers=["islamic", "traditional", "family_oriented"],
        communication_style="formal",
        religious_considerations=["islam", "prayer_times", "halal", "ramadan", "eid"]
    )


@pytest.fixture
def sample_cultural_profile():
    """Sample cultural profile for testing CulturalService."""
    from app.services.cultural_service import CulturalProfile, ReligiousConsideration, CommunicationStyle, CulturalDimension
    
    religious_considerations = ReligiousConsideration(
        religion="Islam",
        dietary_restrictions=["halal"],
        prayer_requirements=True,
        religious_holidays=["Ramadan", "Eid"],
        cultural_sensitivities=["islamic_customs"]
    )
    
    return CulturalProfile(
        nationality="Morocco",
        language=Language.ARABIC,
        communication_style=CommunicationStyle.FORMAL,
        cultural_dimensions={
            CulturalDimension.POWER_DISTANCE: 0.7,
            CulturalDimension.INDIVIDUALISM: 0.3,
            CulturalDimension.UNCERTAINTY_AVOIDANCE: 0.6
        },
        religious_considerations=religious_considerations,
        hospitality_expectations={
            "warmth_level": "high",
            "service_style": "attentive",
            "cultural_respect": "essential"
        },
        service_preferences={
            "communication_frequency": "regular",
            "response_time": "immediate",
            "cultural_adaptation": "high"
        },
        cultural_markers=["islamic", "traditional", "family_oriented"],
        confidence_score=0.95
    )


@pytest.fixture
def sample_agent_state(sample_guest_profile, sample_cultural_context):
    """Sample agent state for testing."""
    return AgentState(
        conversation_id="test_conv_123",
        guest_id="test_guest_123",
        phone_number="+212600123456",
        messages=[],
        guest_profile=sample_guest_profile,
        cultural_context=sample_cultural_context,
        current_intent="greeting",
        emotional_state={
            "sentiment": "positive",
            "emotions": ["welcoming", "helpful"],
            "confidence": 0.8
        },
        knowledge_context={},
        response_plan={},
        actions_to_execute=[],
        metadata={}
    )


@pytest.fixture
def sample_whatsapp_message():
    """Sample WhatsApp message for testing."""
    return {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "test_entry_id",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "212600123456",
                        "phone_number_id": "test_phone_id"
                    },
                    "messages": [{
                        "from": "212600123456",
                        "id": "test_message_id",
                        "timestamp": "1234567890",
                        "text": {"body": "السلام عليكم، أريد حجز غرفة"},
                        "type": "text"
                    }]
                },
                "field": "messages"
            }]
        }]
    }


@pytest.fixture
def cultural_test_cases():
    """Cultural intelligence test cases."""
    return {
        "arabic_formal": {
            "input": "أريد حجز غرفة",
            "expected_language": Language.ARABIC,
            "expected_formality": "formal",
            "expected_cultural_markers": ["islamic", "traditional"]
        },
        "french_sophisticated": {
            "input": "Bonjour, je souhaiterais réserver une suite",
            "expected_language": Language.FRENCH,
            "expected_formality": "sophisticated",
            "expected_cultural_markers": ["refined", "cultured"]
        },
        "english_casual": {
            "input": "Hi! Looking for a room for tonight",
            "expected_language": Language.ENGLISH,
            "expected_formality": "casual",
            "expected_cultural_markers": ["direct", "informal"]
        }
    }


@pytest.fixture
def revenue_optimization_scenarios():
    """Revenue optimization test scenarios."""
    return {
        "upselling_suite": {
            "guest_segment": "luxury",
            "current_booking": "standard_room",
            "opportunity": "suite_upgrade",
            "expected_revenue": 150.0,
            "cultural_adaptation": True
        },
        "spa_cross_sell": {
            "guest_segment": "wellness",
            "trigger": "couple_booking",
            "opportunity": "spa_package",
            "expected_revenue": 200.0,
            "cultural_considerations": ["relaxation", "privacy"]
        },
        "direct_booking_conversion": {
            "booking_channel": "booking_com",
            "guest_loyalty": "returning",
            "incentive": "loyalty_discount",
            "expected_conversion_rate": 0.4
        }
    }


@pytest.fixture
def performance_benchmarks():
    """Performance testing benchmarks."""
    return {
        "response_time": {
            "target": 2.0,  # seconds
            "percentile": 90
        },
        "cultural_accuracy": {
            "target": 0.95,  # 95%
            "minimum": 0.90
        },
        "knowledge_retrieval": {
            "target_time": 0.5,  # seconds
            "relevance_threshold": 0.7
        },
        "concurrent_users": {
            "target": 100,
            "response_degradation_limit": 0.1  # 10%
        }
    }


@pytest.fixture
def integration_test_data():
    """Integration testing data."""
    return {
        "whatsapp_webhook": {
            "valid_signature": "test_signature",
            "verify_token": "test_verify_token"
        },
        "pms_booking": {
            "booking_id": "TEST_BOOKING_123",
            "guest_data": {
                "name": "Test Guest",
                "email": "test@example.com",
                "phone": "+212600123456"
            },
            "room_details": {
                "type": "suite",
                "check_in": "2024-01-15",
                "check_out": "2024-01-18"
            }
        },
        "vector_search": {
            "query": "traditional Moroccan spa treatments",
            "expected_results": 5,
            "relevance_threshold": 0.7
        }
    }


# Test utilities
class TestHelpers:
    """Helper functions for testing."""
    
    @staticmethod
    def create_mock_instructor_response(response_class, **kwargs):
        """Create mock Instructor response."""
        return response_class(**kwargs)
    
    @staticmethod
    def assert_cultural_appropriateness(response: str, cultural_context: CulturalContext):
        """Assert that response is culturally appropriate."""
        if cultural_context.language == Language.ARABIC:
            # Check for Arabic cultural elements
            assert any(marker in response.lower() for marker in ["respect", "honor", "family"])
        elif cultural_context.language == Language.FRENCH:
            # Check for French sophistication
            assert any(marker in response.lower() for marker in ["sophisticated", "elegant", "refined"])
    
    @staticmethod
    def measure_response_time(func):
        """Decorator to measure function response time."""
        async def wrapper(*args, **kwargs):
            start_time = datetime.now()
            result = await func(*args, **kwargs)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            return result, response_time
        return wrapper


@pytest.fixture
def test_helpers():
    """Test helper utilities."""
    return TestHelpers
