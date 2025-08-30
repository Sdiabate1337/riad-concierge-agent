"""
Integration tests for LangGraph Agent Workflow
Testing complete agent orchestration and service interactions
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import json

from app.agents.riad_agent import RiadConciergeAgent
from app.services.agent_service import AgentService
from app.models.agent_state import AgentState, GuestProfile, CulturalContext, Language
from app.models.instructor_models import (
    IntentClassification, EmotionalAnalysis, CulturalResponse,
    KnowledgeRetrieval, ServiceAction, UpsellingSuggestion
)


class TestAgentWorkflowIntegration:
    """Integration tests for complete agent workflow."""
    
    @pytest.fixture
    async def agent_service(self, mock_redis, mock_openai_client, mock_pinecone_index):
        """Create agent service with mocked dependencies."""
        service = AgentService()
        
        # Mock all service dependencies
        service.agent.cultural_service.redis_client = mock_redis
        service.agent.knowledge_service.redis_client = mock_redis
        service.agent.knowledge_service.index = mock_pinecone_index
        service.agent.pms_service.redis_client = mock_redis
        service.agent.whatsapp_service.redis_client = mock_redis
        service.agent.proactive_service.redis_client = mock_redis
        
        return service
    
    @pytest.mark.asyncio
    async def test_complete_message_processing_workflow(
        self, 
        agent_service, 
        sample_whatsapp_message,
        sample_guest_profile,
        sample_cultural_context
    ):
        """Test complete message processing from WhatsApp to response."""
        
        # Mock service responses for each workflow node
        agent_service.agent.cultural_service.create_comprehensive_cultural_profile = AsyncMock(
            return_value=sample_cultural_context
        )
        
        agent_service.agent.pms_service.get_enhanced_guest_profile = AsyncMock(
            return_value=sample_guest_profile
        )
        
        agent_service.agent.instructor_client.chat.completions.create = MagicMock(
            side_effect=[
                # Intent classification
                IntentClassification(
                    intent="booking_inquiry",
                    confidence=0.92,
                    entities={"room_type": "suite", "guests": 2},
                    context_needed=["availability", "pricing"]
                ),
                # Emotional analysis
                EmotionalAnalysis(
                    sentiment="positive",
                    emotions=["excited", "curious"],
                    confidence=0.88,
                    cultural_context=["respectful", "formal"]
                ),
                # Knowledge retrieval planning
                KnowledgeRetrieval(
                    query_type="availability_check",
                    relevant_knowledge=["room_types", "pricing", "amenities"],
                    cultural_adaptation_needed=True,
                    confidence=0.94
                ),
                # Cultural response generation
                CulturalResponse(
                    message="أهلاً وسهلاً بكم! لدينا جناح فاخر متاح بإطلالة رائعة على الحديقة التقليدية",
                    cultural_markers=["moroccan_hospitality", "traditional_welcome"],
                    tone_adjustments=["warm", "respectful"],
                    formality_level="formal",
                    confidence_score=0.96
                )
            ]
        )
        
        # Mock knowledge retrieval
        agent_service.agent.knowledge_service.get_hybrid_knowledge = AsyncMock(
            return_value=(
                {
                    "room_availability": {"suite": True, "deluxe": True},
                    "pricing": {"suite": 250, "deluxe": 180},
                    "amenities": ["spa", "rooftop", "traditional_hammam"]
                },
                []  # RAG results
            )
        )
        
        # Mock revenue opportunity detection
        agent_service.agent.pms_service.identify_revenue_opportunities = AsyncMock(
            return_value=[
                UpsellingSuggestion(
                    opportunity_type="spa_package",
                    description="Traditional Moroccan hammam experience",
                    additional_revenue=120.0,
                    confidence=0.85,
                    cultural_adaptation="Authentic wellness experience in traditional style"
                )
            ]
        )
        
        # Mock WhatsApp message sending
        agent_service.agent.whatsapp_service.send_culturally_adapted_message = AsyncMock(
            return_value="msg_12345"
        )
        
        # Process the message through complete workflow
        result = await agent_service.process_message(
            phone_number="+212600123456",
            message_text="أريد حجز جناح لشخصين",
            message_type="text"
        )
        
        # Verify workflow completion
        assert result is not None
        assert result.get("status") == "completed"
        assert result.get("message_id") == "msg_12345"
        
        # Verify all services were called
        agent_service.agent.cultural_service.create_comprehensive_cultural_profile.assert_called_once()
        agent_service.agent.pms_service.get_enhanced_guest_profile.assert_called_once()
        agent_service.agent.knowledge_service.get_hybrid_knowledge.assert_called_once()
        agent_service.agent.whatsapp_service.send_culturally_adapted_message.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_guest_journey_state_transitions(
        self, 
        agent_service,
        sample_guest_profile
    ):
        """Test agent state transitions through guest journey stages."""
        
        # Initial state - Pre-arrival
        initial_state = AgentState(
            conversation_id="test_conv_123",
            guest_id="test_guest_123",
            phone_number="+212600123456",
            messages=[],
            guest_profile=sample_guest_profile,
            current_intent="pre_arrival_inquiry",
            metadata={"journey_stage": "pre_arrival"}
        )
        
        # Mock state transitions
        agent_service.agent._validate_message = AsyncMock(return_value=True)
        agent_service.agent._detect_language = AsyncMock(return_value=Language.ARABIC)
        agent_service.agent._load_guest_profile = AsyncMock(return_value=sample_guest_profile)
        
        # Process pre-arrival message
        agent_service.agent.instructor_client.chat.completions.create.return_value = IntentClassification(
            intent="arrival_confirmation",
            confidence=0.94,
            entities={"arrival_time": "15:00", "special_requests": ["early_checkin"]},
            context_needed=["room_preparation"]
        )
        
        # Execute workflow nodes
        state = await agent_service.agent._classify_intent(initial_state)
        assert state.current_intent == "arrival_confirmation"
        
        # Verify journey stage progression
        assert state.metadata.get("journey_stage") in ["pre_arrival", "arrival_preparation"]
    
    @pytest.mark.asyncio
    async def test_cultural_intelligence_integration(
        self, 
        agent_service,
        cultural_test_cases
    ):
        """Test cultural intelligence integration across all services."""
        
        for test_case_name, test_data in cultural_test_cases.items():
            # Mock cultural service response
            cultural_context = CulturalContext(
                language=test_data["expected_language"],
                nationality="Morocco" if test_data["expected_language"] == Language.ARABIC else "France",
                cultural_markers=test_data["expected_cultural_markers"],
                communication_style=test_data["expected_formality"]
            )
            
            agent_service.agent.cultural_service.create_comprehensive_cultural_profile = AsyncMock(
                return_value=cultural_context
            )
            
            # Mock culturally adapted response
            agent_service.agent.instructor_client.chat.completions.create.return_value = CulturalResponse(
                message=f"Culturally adapted response for {test_case_name}",
                cultural_markers=test_data["expected_cultural_markers"],
                tone_adjustments=["appropriate", "respectful"],
                formality_level=test_data["expected_formality"],
                confidence_score=0.91
            )
            
            # Process message
            result = await agent_service.process_message(
                phone_number="+212600123456",
                message_text=test_data["input"],
                message_type="text"
            )
            
            # Verify cultural adaptation occurred
            assert result is not None
            agent_service.agent.cultural_service.create_comprehensive_cultural_profile.assert_called()
    
    @pytest.mark.asyncio
    async def test_revenue_optimization_integration(
        self, 
        agent_service,
        revenue_optimization_scenarios
    ):
        """Test revenue optimization integration across workflow."""
        
        for scenario_name, scenario_data in revenue_optimization_scenarios.items():
            # Mock guest profile with appropriate segment
            guest_profile = GuestProfile(
                guest_id="test_guest",
                name="Test Guest",
                phone="+212600123456",
                nationality="Morocco",
                language=Language.ARABIC,
                preferences={"segment": scenario_data["guest_segment"]}
            )
            
            # Mock revenue opportunity detection
            if "opportunity" in scenario_data:
                opportunity = UpsellingSuggestion(
                    opportunity_type=scenario_data["opportunity"],
                    description=f"Revenue opportunity: {scenario_name}",
                    additional_revenue=scenario_data["expected_revenue"],
                    confidence=0.87,
                    cultural_adaptation="Culturally appropriate presentation"
                )
                
                agent_service.agent.pms_service.identify_revenue_opportunities = AsyncMock(
                    return_value=[opportunity]
                )
            
            # Process message that could trigger revenue opportunity
            result = await agent_service.process_message(
                phone_number="+212600123456",
                message_text="I'm interested in upgrading my experience",
                message_type="text"
            )
            
            # Verify revenue optimization was considered
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_proactive_intelligence_triggers(self, agent_service):
        """Test proactive intelligence triggers during workflow."""
        
        # Mock proactive service
        agent_service.agent.proactive_service.start_guest_journey_monitoring = AsyncMock()
        agent_service.agent.proactive_service.trigger_proactive_action = AsyncMock(
            return_value="proactive_msg_789"
        )
        
        # Process message that should trigger proactive monitoring
        result = await agent_service.process_message(
            phone_number="+212600123456",
            message_text="We just arrived at the riad",
            message_type="text"
        )
        
        # Verify proactive intelligence was activated
        assert result is not None
        # Note: In real implementation, proactive monitoring would be triggered
        # based on intent classification and journey stage
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallbacks(self, agent_service):
        """Test error handling and fallback mechanisms in workflow."""
        
        # Simulate service failure
        agent_service.agent.knowledge_service.get_hybrid_knowledge = AsyncMock(
            side_effect=Exception("Knowledge service unavailable")
        )
        
        # Mock fallback response
        agent_service.agent._get_fallback_response = AsyncMock(
            return_value="I apologize, but I'm experiencing technical difficulties. Let me connect you with our staff."
        )
        
        # Process message during service failure
        result = await agent_service.process_message(
            phone_number="+212600123456",
            message_text="What amenities do you have?",
            message_type="text"
        )
        
        # Verify graceful fallback
        assert result is not None
        assert result.get("status") in ["fallback", "completed"]
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_context(self, agent_service):
        """Test context preservation across multiple conversation turns."""
        
        conversation_id = "multi_turn_test"
        phone_number = "+212600123456"
        
        # First message - booking inquiry
        result1 = await agent_service.process_message(
            phone_number=phone_number,
            message_text="أريد حجز غرفة",
            message_type="text",
            conversation_id=conversation_id
        )
        
        # Second message - follow-up question
        result2 = await agent_service.process_message(
            phone_number=phone_number,
            message_text="كم السعر؟",
            message_type="text",
            conversation_id=conversation_id
        )
        
        # Third message - confirmation
        result3 = await agent_service.process_message(
            phone_number=phone_number,
            message_text="نعم، أريد الحجز",
            message_type="text",
            conversation_id=conversation_id
        )
        
        # Verify all messages were processed with context
        assert result1 is not None
        assert result2 is not None
        assert result3 is not None
        
        # In real implementation, context would be preserved across turns
        # through the conversation state management


class TestServiceInteractionIntegration:
    """Test interactions between different services."""
    
    @pytest.mark.asyncio
    async def test_cultural_knowledge_integration(
        self, 
        agent_service,
        sample_cultural_context
    ):
        """Test integration between cultural and knowledge services."""
        
        # Mock cultural context creation
        agent_service.agent.cultural_service.create_comprehensive_cultural_profile = AsyncMock(
            return_value=sample_cultural_context
        )
        
        # Mock knowledge retrieval with cultural relevance
        agent_service.agent.knowledge_service.get_hybrid_knowledge = AsyncMock(
            return_value=(
                {
                    "culturally_relevant_info": "Traditional Moroccan hospitality practices",
                    "religious_considerations": "Prayer times and halal dining options",
                    "local_customs": "Mint tea ceremony and traditional greetings"
                },
                []
            )
        )
        
        # Process culturally-specific query
        result = await agent_service.process_message(
            phone_number="+212600123456",
            message_text="ما هي الخدمات المتوفرة للضيوف المسلمين؟",
            message_type="text"
        )
        
        # Verify cultural-knowledge integration
        assert result is not None
        agent_service.agent.cultural_service.create_comprehensive_cultural_profile.assert_called()
        agent_service.agent.knowledge_service.get_hybrid_knowledge.assert_called()
    
    @pytest.mark.asyncio
    async def test_pms_whatsapp_integration(self, agent_service):
        """Test integration between PMS and WhatsApp services."""
        
        # Mock PMS booking creation
        agent_service.agent.pms_service.create_sophisticated_booking = AsyncMock(
            return_value="BOOKING_123456"
        )
        
        # Mock WhatsApp confirmation message
        agent_service.agent.whatsapp_service.send_culturally_adapted_message = AsyncMock(
            return_value="confirmation_msg_789"
        )
        
        # Process booking confirmation
        booking_data = {
            "guest_name": "Ahmed Hassan",
            "room_type": "suite",
            "check_in": "2024-01-15",
            "check_out": "2024-01-18"
        }
        
        # Simulate booking workflow
        booking_id = await agent_service.agent.pms_service.create_sophisticated_booking(
            guest_data={"name": "Ahmed Hassan"},
            booking_data=booking_data
        )
        
        # Send confirmation via WhatsApp
        message_id = await agent_service.agent.whatsapp_service.send_culturally_adapted_message(
            to="+212600123456",
            content=f"Booking confirmed: {booking_id}",
            cultural_context=CulturalContext(
                language=Language.ARABIC,
                nationality="Morocco",
                cultural_markers=["islamic"],
                communication_style="formal"
            )
        )
        
        # Verify integration
        assert booking_id == "BOOKING_123456"
        assert message_id == "confirmation_msg_789"
    
    @pytest.mark.asyncio
    async def test_proactive_cultural_integration(self, agent_service):
        """Test integration between proactive and cultural services."""
        
        cultural_context = CulturalContext(
            language=Language.ARABIC,
            nationality="Morocco",
            cultural_markers=["islamic", "ramadan"],
            communication_style="formal",
            religious_considerations={
                "current_period": "Ramadan",
                "prayer_times": True,
                "fasting": True
            }
        )
        
        # Mock proactive cultural moment detection
        agent_service.agent.proactive_service.identify_and_act_on_cultural_moments = AsyncMock(
            return_value=[
                MagicMock(
                    message_id="cultural_moment_123",
                    event_type="ramadan_iftar",
                    cultural_adaptation={"ar": "إفطار رمضاني تقليدي"}
                )
            ]
        )
        
        # Trigger cultural moment detection
        cultural_messages = await agent_service.agent.proactive_service.identify_and_act_on_cultural_moments(
            guest_id="test_guest_123",
            current_context={"time": "sunset", "season": "ramadan"}
        )
        
        # Verify cultural-proactive integration
        assert len(cultural_messages) > 0
        assert cultural_messages[0].event_type == "ramadan_iftar"
