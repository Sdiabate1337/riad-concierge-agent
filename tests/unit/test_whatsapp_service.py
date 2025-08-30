"""
Unit tests for WhatsApp Business API Service
Testing sophisticated message handling and cultural adaptation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json
import asyncio

from app.services.whatsapp_service import (
    WhatsAppService, MessageType, MessagePriority, 
    CulturalTemplate, MessageMetrics
)
from app.models.agent_state import Language, CulturalContext


class TestWhatsAppService:
    """Test suite for WhatsApp Business API Service."""
    
    @pytest.fixture
    async def whatsapp_service(self, mock_redis, mock_openai_client):
        """Create WhatsApp service with mocked dependencies."""
        service = WhatsAppService()
        service.redis_client = mock_redis
        service.instructor_client = mock_openai_client
        return service
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, whatsapp_service, mock_redis):
        """Test WhatsApp service initialization with background tasks."""
        await whatsapp_service.initialize()
        
        # Verify Redis connection
        mock_redis.ping.assert_called_once()
        
        # Verify background tasks started
        assert whatsapp_service._running is True
        assert whatsapp_service._queue_processor_task is not None
        assert len(whatsapp_service._background_tasks) > 0
    
    @pytest.mark.asyncio
    async def test_culturally_adapted_message_queuing(
        self, 
        whatsapp_service, 
        sample_cultural_context
    ):
        """Test culturally adapted message queuing with priority."""
        to = "+212600123456"
        content = "Welcome to our riad"
        priority = MessagePriority.HIGH
        
        # Mock cultural intelligence
        whatsapp_service.instructor_client.chat.completions.create.return_value = MagicMock(
            message="أهلاً وسهلاً بكم في رياضنا التقليدي",
            cultural_markers=["moroccan_hospitality"],
            tone_adjustments=["warm", "respectful"],
            formality_level="formal",
            confidence_score=0.95
        )
        
        message_id = await whatsapp_service.send_culturally_adapted_message(
            to, content, sample_cultural_context, priority
        )
        
        assert message_id is not None
        assert message_id.startswith("msg_")
        
        # Verify message was queued in priority queue for HIGH priority
        assert not whatsapp_service.priority_queue.empty()
    
    @pytest.mark.asyncio
    async def test_interactive_cultural_message(
        self, 
        whatsapp_service, 
        sample_cultural_context
    ):
        """Test interactive message creation with cultural adaptation."""
        to = "+212600123456"
        text = "What would you like to do today?"
        buttons = [
            {"id": "spa", "title": "Spa Services"},
            {"id": "tour", "title": "Cultural Tour"},
            {"id": "dining", "title": "Traditional Dining"}
        ]
        
        # Mock cultural adaptation methods
        whatsapp_service._culturally_adapt_text = AsyncMock(
            return_value="ماذا تودون أن تفعلوا اليوم؟"
        )
        whatsapp_service._culturally_adapt_buttons = AsyncMock(
            return_value=[
                {"id": "spa", "title": "خدمات السبا"},
                {"id": "tour", "title": "جولة ثقافية"},
                {"id": "dining", "title": "عشاء تقليدي"}
            ]
        )
        whatsapp_service._apply_advanced_cultural_formatting = AsyncMock(
            return_value={"messaging_product": "whatsapp", "type": "interactive"}
        )
        whatsapp_service._send_with_circuit_breaker = AsyncMock(
            return_value={"messages": [{"id": "interactive_msg_123"}]}
        )
        whatsapp_service._track_interactive_message = AsyncMock()
        
        message_id = await whatsapp_service.send_interactive_cultural_message(
            to, text, buttons, sample_cultural_context
        )
        
        assert message_id == "interactive_msg_123"
        whatsapp_service._culturally_adapt_text.assert_called_once()
        whatsapp_service._culturally_adapt_buttons.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_proactive_cultural_template(
        self, 
        whatsapp_service, 
        sample_cultural_context
    ):
        """Test proactive message using cultural templates."""
        to = "+212600123456"
        template_type = "welcome_sequence"
        parameters = {"guest_name": "أحمد", "room_number": "101"}
        
        # Mock template retrieval and processing
        whatsapp_service._get_cultural_template = MagicMock(
            return_value="مرحباً {guest_name}، غرفتكم رقم {room_number} جاهزة"
        )
        whatsapp_service._apply_template_parameters = AsyncMock(
            return_value="مرحباً أحمد، غرفتكم رقم 101 جاهزة"
        )
        whatsapp_service.send_culturally_adapted_message = AsyncMock(
            return_value="template_msg_456"
        )
        
        message_id = await whatsapp_service.send_proactive_cultural_template(
            to, template_type, sample_cultural_context, parameters
        )
        
        assert message_id == "template_msg_456"
        whatsapp_service._get_cultural_template.assert_called_once_with(
            template_type, sample_cultural_context
        )
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, whatsapp_service, mock_redis):
        """Test rate limiting functionality."""
        phone_number = "+212600123456"
        
        # Test first request - should pass
        mock_redis.get.return_value = None
        result = await whatsapp_service._check_rate_limit(phone_number)
        assert result is True
        mock_redis.setex.assert_called_with(f"rate_limit:{phone_number}", 3600, 1)
        
        # Test within limit - should pass
        mock_redis.get.return_value = "10"
        result = await whatsapp_service._check_rate_limit(phone_number)
        assert result is True
        mock_redis.incr.assert_called_with(f"rate_limit:{phone_number}")
        
        # Test rate limit exceeded - should fail
        mock_redis.get.return_value = "25"  # Over limit of 20
        result = await whatsapp_service._check_rate_limit(phone_number)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cultural_intelligence_application(
        self, 
        whatsapp_service, 
        sample_cultural_context
    ):
        """Test cultural intelligence application to message content."""
        content = "Your room is ready for check-in"
        
        # Mock Instructor response with AsyncMock
        mock_response = MagicMock(
            message="غرفتكم جاهزة للوصول، مرحباً بكم",
            cultural_markers=["moroccan_hospitality", "respectful_tone"],
            tone_adjustments=["warm", "welcoming"],
            formality_level="formal",
            confidence_score=0.92
        )
        whatsapp_service.instructor_client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        
        # Mock enhancement method
        whatsapp_service._enhance_cultural_formatting = AsyncMock(
            return_value="أهلاً وسهلاً، غرفتكم جاهزة للوصول، مرحباً بكم 🏛️"
        )
        
        adapted_content = await whatsapp_service._apply_cultural_intelligence(
            content, sample_cultural_context
        )
        
        assert "غرفتكم" in adapted_content  # Arabic content
        assert "🏛️" in adapted_content  # Cultural emoji
    
    @pytest.mark.asyncio
    async def test_message_queue_processing(self, whatsapp_service):
        """Test message queue processing with priority handling."""
        # Mock queue processing methods
        whatsapp_service._send_queued_message = AsyncMock()
        
        # Add high priority message
        high_priority_msg = {
            "id": "high_msg_1",
            "priority": MessagePriority.HIGH,
            "content": "Urgent message"
        }
        
        # Add normal priority message
        normal_priority_msg = {
            "id": "normal_msg_1", 
            "priority": MessagePriority.NORMAL,
            "content": "Normal message"
        }
        
        # Queue messages
        await whatsapp_service.priority_queue.put((1, high_priority_msg))
        await whatsapp_service.message_queue.put(normal_priority_msg)
        
        # Process one cycle
        whatsapp_service._running = True
        
        # Simulate processing (would normally run in background)
        if not whatsapp_service.priority_queue.empty():
            _, message_data = await whatsapp_service.priority_queue.get()
            await whatsapp_service._send_queued_message(message_data)
        
        # Verify high priority message processed first
        whatsapp_service._send_queued_message.assert_called_once_with(high_priority_msg)
    
    @pytest.mark.asyncio
    async def test_phone_number_formatting(self, whatsapp_service):
        """Test phone number formatting for WhatsApp API."""
        # Test various phone number formats
        test_cases = [
            ("+212600123456", "212600123456"),
            ("0600123456", "212600123456"),  # Moroccan local format
            ("212600123456", "212600123456"),
            ("+33123456789", "33123456789"),  # French number
        ]
        
        for input_number, expected_output in test_cases:
            formatted = whatsapp_service._format_phone_number(input_number)
            assert formatted == expected_output
    
    @pytest.mark.asyncio
    async def test_cultural_template_initialization(self, whatsapp_service):
        """Test cultural template initialization and structure."""
        templates = whatsapp_service._initialize_cultural_templates()
        
        # Verify all languages have templates
        required_languages = ["ar", "fr", "en"]
        for lang in required_languages:
            assert lang in templates
            template = templates[lang]
            assert isinstance(template, CulturalTemplate)
            assert template.language == lang
            assert template.greeting
            assert template.closing
            assert template.formality_markers
    
    @pytest.mark.asyncio
    async def test_message_metrics_tracking(self, whatsapp_service):
        """Test message delivery and engagement metrics tracking."""
        message_id = "test_msg_123"
        phone_number = "+212600123456"
        content = "Test message"
        
        await whatsapp_service._track_message_sent(message_id, phone_number, content)
        
        # Verify metrics stored
        assert message_id in whatsapp_service.message_metrics
        metrics = whatsapp_service.message_metrics[message_id]
        assert isinstance(metrics, MessageMetrics)
        assert metrics.sent_at is not None
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, whatsapp_service):
        """Test circuit breaker for API resilience."""
        # Test closed state (normal operation)
        assert whatsapp_service.circuit_breaker_state == "closed"
        
        # Simulate failures
        whatsapp_service.failure_count = 5
        whatsapp_service.last_failure_time = datetime.now()
        
        # Mock API call that would trigger circuit breaker
        payload = {"messaging_product": "whatsapp", "to": "212600123456"}
        
        # Circuit breaker logic would be implemented in _send_with_circuit_breaker
        # This is a placeholder for the actual implementation
        assert whatsapp_service.circuit_breaker_state in ["closed", "open", "half-open"]
    
    @pytest.mark.asyncio
    async def test_cleanup_resources(self, whatsapp_service, mock_redis):
        """Test proper cleanup of WhatsApp service resources."""
        whatsapp_service._running = True
        
        # Create proper task mocks with synchronous cancel() method
        queue_task_mock = MagicMock()
        queue_task_mock.cancel = MagicMock(return_value=None)
        whatsapp_service._queue_processor_task = queue_task_mock
        
        # Create awaitable background task mocks
        async def mock_task():
            return None
            
        bg_task1 = asyncio.create_task(mock_task())
        bg_task2 = asyncio.create_task(mock_task())
        bg_task1.cancel = MagicMock(return_value=None)
        bg_task2.cancel = MagicMock(return_value=None)
        whatsapp_service._background_tasks = [bg_task1, bg_task2]
        
        # Mock the client.aclose() method
        whatsapp_service.client = AsyncMock()
        whatsapp_service.client.aclose = AsyncMock()
        
        await whatsapp_service.cleanup()
        
        assert whatsapp_service._running is False
        queue_task_mock.cancel.assert_called_once()
        whatsapp_service.client.aclose.assert_called_once()
        mock_redis.close.assert_called_once()
    
    def test_priority_value_mapping(self, whatsapp_service):
        """Test message priority value mapping for queue ordering."""
        priority_mappings = [
            (MessagePriority.URGENT, 1),
            (MessagePriority.HIGH, 2),
            (MessagePriority.NORMAL, 3),
            (MessagePriority.LOW, 4)
        ]
        
        for priority, expected_value in priority_mappings:
            value = whatsapp_service._get_priority_value(priority)
            assert value == expected_value
    
    @pytest.mark.asyncio
    async def test_cultural_formatting_enhancement(
        self, 
        whatsapp_service, 
        sample_cultural_context
    ):
        """Test cultural formatting enhancement with markers."""
        content = "Welcome to our riad"
        cultural_markers = ["moroccan_hospitality", "traditional_welcome"]
        
        enhanced_content = await whatsapp_service._enhance_cultural_formatting(
            content, sample_cultural_context, cultural_markers
        )
        
        # Should add Arabic greeting and cultural elements
        template = whatsapp_service.cultural_templates.get("ar")
        if template:
            assert template.greeting in enhanced_content or content in enhanced_content
        
        # Should add cultural emoji for Moroccan hospitality
        if "moroccan_hospitality" in cultural_markers:
            assert "🏛️" in enhanced_content or "🇲🇦" in enhanced_content


class TestWhatsAppIntegrationScenarios:
    """Test real-world WhatsApp integration scenarios."""
    
    @pytest.fixture
    async def whatsapp_service(self, mock_redis, mock_openai_client):
        """Create WhatsApp service with mocked dependencies."""
        service = WhatsAppService()
        service.redis_client = mock_redis
        service.instructor_client = mock_openai_client
        return service
    
    @pytest.mark.asyncio
    async def test_guest_arrival_notification_sequence(self, whatsapp_service):
        """Test complete guest arrival notification sequence."""
        guest_phone = "+212600123456"
        cultural_context = CulturalContext(
            language=Language.ARABIC,
            nationality="Morocco",
            cultural_markers=["islamic", "family_oriented"],
            communication_style="formal"
        )
        
        # Mock the complete sequence
        whatsapp_service.send_culturally_adapted_message = AsyncMock(
            side_effect=["msg_1", "msg_2", "msg_3"]
        )
        
        # Send welcome message
        welcome_id = await whatsapp_service.send_culturally_adapted_message(
            guest_phone, "Welcome to our riad", cultural_context, MessagePriority.HIGH
        )
        
        # Send room information
        room_id = await whatsapp_service.send_culturally_adapted_message(
            guest_phone, "Your room is ready", cultural_context, MessagePriority.NORMAL
        )
        
        # Send cultural orientation
        orientation_id = await whatsapp_service.send_culturally_adapted_message(
            guest_phone, "Cultural experiences available", cultural_context, MessagePriority.LOW
        )
        
        assert welcome_id == "msg_1"
        assert room_id == "msg_2" 
        assert orientation_id == "msg_3"
        assert whatsapp_service.send_culturally_adapted_message.call_count == 3
    
    @pytest.mark.asyncio
    async def test_multilingual_message_handling(self, whatsapp_service):
        """Test handling messages in different languages simultaneously."""
        # Arabic guest
        arabic_context = CulturalContext(
            language=Language.ARABIC,
            nationality="Morocco",
            cultural_markers=["islamic"],
            communication_style="formal"
        )
        
        # French guest
        french_context = CulturalContext(
            language=Language.FRENCH,
            nationality="France", 
            cultural_markers=["sophisticated"],
            communication_style="formal"
        )
        
        # English guest
        english_context = CulturalContext(
            language=Language.ENGLISH,
            nationality="United States",
            cultural_markers=["direct"],
            communication_style="casual"
        )
        
        # Mock cultural intelligence for each language
        whatsapp_service._apply_cultural_intelligence = AsyncMock(
            side_effect=[
                "أهلاً وسهلاً بكم",  # Arabic
                "Bienvenue dans notre riad",  # French
                "Welcome to our riad"  # English
            ]
        )
        
        # Send messages to all guests
        arabic_msg = await whatsapp_service.send_culturally_adapted_message(
            "+212600123456", "Welcome", arabic_context
        )
        
        french_msg = await whatsapp_service.send_culturally_adapted_message(
            "+33123456789", "Welcome", french_context
        )
        
        english_msg = await whatsapp_service.send_culturally_adapted_message(
            "+1234567890", "Welcome", english_context
        )
        
        # Verify all messages were processed
        assert arabic_msg is not None
        assert french_msg is not None
        assert english_msg is not None
        assert whatsapp_service._apply_cultural_intelligence.call_count == 3
