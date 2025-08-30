"""
Main Riad Concierge AI Agent using LangGraph + Instructor architecture.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from loguru import logger

from app.core.config import get_settings
from app.models.agent_state import AgentState, Action, ActionType
from app.models.instructor_models import (
    CulturalResponse,
    IntentClassificationResult,
    EmotionalAnalysis,
    ServiceAction,
    UpsellOpportunity,
    DirectBookingConversion,
    KnowledgeRetrieval,
    ResponsePlan,
    get_instructor_client
)
from app.services.cultural_service import CulturalService
from app.services.knowledge_service import KnowledgeService
from app.services.pms_service import PMSService
from app.services.whatsapp_service import WhatsAppService


class RiadConciergeAgent:
    """Main agent orchestrator using LangGraph workflow."""
    
    def __init__(self):
        self.settings = get_settings()
        self.instructor_client = get_instructor_client()
        self.cultural_service = CulturalService()
        self.knowledge_service = KnowledgeService()
        self.pms_service = PMSService()
        self.whatsapp_service = WhatsAppService()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for message processing."""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for each processing step
        workflow.add_node("validate_input", self._validate_input_node)
        workflow.add_node("detect_language", self._detect_language_node)
        workflow.add_node("load_guest_profile", self._load_guest_profile_node)
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("analyze_emotion", self._analyze_emotion_node)
        workflow.add_node("assemble_cag_context", self._assemble_cag_context_node)
        workflow.add_node("retrieve_rag_context", self._retrieve_rag_context_node)
        workflow.add_node("detect_opportunities", self._detect_opportunities_node)
        workflow.add_node("plan_response", self._plan_response_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("execute_actions", self._execute_actions_node)
        workflow.add_node("log_interaction", self._log_interaction_node)
        
        # Define the workflow edges
        workflow.set_entry_point("validate_input")
        
        workflow.add_edge("validate_input", "detect_language")
        workflow.add_edge("detect_language", "load_guest_profile")
        workflow.add_edge("load_guest_profile", "classify_intent")
        workflow.add_edge("classify_intent", "analyze_emotion")
        workflow.add_edge("analyze_emotion", "assemble_cag_context")
        workflow.add_edge("assemble_cag_context", "retrieve_rag_context")
        workflow.add_edge("retrieve_rag_context", "detect_opportunities")
        workflow.add_edge("detect_opportunities", "plan_response")
        workflow.add_edge("plan_response", "generate_response")
        workflow.add_edge("generate_response", "execute_actions")
        workflow.add_edge("execute_actions", "log_interaction")
        workflow.add_edge("log_interaction", END)
        
        return workflow.compile()
    
    async def _validate_input_node(self, state: AgentState) -> AgentState:
        """Validate and sanitize input message."""
        start_time = time.time()
        
        try:
            # Extract message content from raw WhatsApp data
            raw_message = state.raw_message
            
            # Basic validation
            if not raw_message.get("from"):
                state.errors.append("Missing sender information")
                return state
            
            # Create human message from WhatsApp data
            message_text = ""
            if raw_message.get("text"):
                message_text = raw_message["text"].get("body", "")
            elif raw_message.get("interactive"):
                # Handle interactive messages (buttons, lists)
                interactive = raw_message["interactive"]
                if interactive.get("button_reply"):
                    message_text = interactive["button_reply"].get("title", "")
                elif interactive.get("list_reply"):
                    message_text = interactive["list_reply"].get("title", "")
            
            if message_text:
                human_message = HumanMessage(content=message_text)
                state.add_message(human_message)
            
            logger.info(f"Input validated for conversation {state.conversation_id}")
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            state.errors.append(f"Input validation error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Input validation took {processing_time:.3f}s")
        
        return state
    
    async def _detect_language_node(self, state: AgentState) -> AgentState:
        """Detect message language and set cultural context."""
        start_time = time.time()
        
        try:
            latest_message = state.get_latest_human_message()
            if not latest_message:
                return state
            
            # Use cultural service to detect language and cultural context
            cultural_context = await self.cultural_service.detect_cultural_context(
                message=latest_message.content,
                phone_number=state.raw_message.get("from", ""),
                guest_profile=state.guest_profile
            )
            
            state.cultural_context = cultural_context
            logger.info(f"Language detected: {cultural_context.language}")
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            state.errors.append(f"Language detection error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Language detection took {processing_time:.3f}s")
        
        return state
    
    async def _load_guest_profile_node(self, state: AgentState) -> AgentState:
        """Load or create guest profile."""
        start_time = time.time()
        
        try:
            phone_number = state.raw_message.get("from", "")
            
            # Load existing profile or create new one
            guest_profile = await self.pms_service.get_guest_profile(phone_number)
            
            if not guest_profile:
                # Create new guest profile
                guest_profile = await self.pms_service.create_guest_profile(
                    phone_number=phone_number,
                    cultural_context=state.cultural_context
                )
            
            state.guest_profile = guest_profile
            logger.info(f"Guest profile loaded for {phone_number}")
            
        except Exception as e:
            logger.error(f"Guest profile loading failed: {e}")
            state.errors.append(f"Guest profile error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Guest profile loading took {processing_time:.3f}s")
        
        return state
    
    async def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Classify message intent using Instructor."""
        start_time = time.time()
        
        try:
            latest_message = state.get_latest_human_message()
            if not latest_message:
                return state
            
            # Prepare context for intent classification
            context = {
                "message": latest_message.content,
                "conversation_history": [msg.content for msg in state.get_conversation_history()],
                "guest_profile": state.guest_profile.dict() if state.guest_profile else {},
                "cultural_context": state.cultural_context.dict() if state.cultural_context else {}
            }
            
            # Use Instructor to get structured intent classification
            intent_result = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=IntentClassificationResult,
                messages=[
                    SystemMessage(content=self._get_intent_classification_prompt()),
                    HumanMessage(content=f"Classify this message: {latest_message.content}\n\nContext: {context}")
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            state.intent = intent_result
            logger.info(f"Intent classified: {intent_result.primary_intent} (confidence: {intent_result.confidence_score:.2f})")
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            state.errors.append(f"Intent classification error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Intent classification took {processing_time:.3f}s")
        
        return state
    
    async def _analyze_emotion_node(self, state: AgentState) -> AgentState:
        """Analyze emotional state using Instructor."""
        start_time = time.time()
        
        try:
            latest_message = state.get_latest_human_message()
            if not latest_message:
                return state
            
            # Use Instructor for emotional analysis
            emotional_analysis = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=EmotionalAnalysis,
                messages=[
                    SystemMessage(content=self._get_emotional_analysis_prompt()),
                    HumanMessage(content=f"Analyze emotions in: {latest_message.content}\n\nCultural context: {state.cultural_context.dict() if state.cultural_context else {}}")
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            state.emotional_state = emotional_analysis
            logger.info(f"Emotion analyzed: {emotional_analysis.primary_emotion} (intensity: {emotional_analysis.emotion_intensity:.2f})")
            
        except Exception as e:
            logger.error(f"Emotional analysis failed: {e}")
            state.errors.append(f"Emotional analysis error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Emotional analysis took {processing_time:.3f}s")
        
        return state
    
    async def _assemble_cag_context_node(self, state: AgentState) -> AgentState:
        """Assemble static CAG knowledge context."""
        start_time = time.time()
        
        try:
            latest_message = state.get_latest_human_message()
            if not latest_message:
                return state
            
            # Get relevant static knowledge
            cag_knowledge = await self.knowledge_service.get_cag_knowledge(
                query=latest_message.content,
                intent=state.intent.primary_intent if state.intent else "unclear",
                cultural_context=state.cultural_context
            )
            
            state.cag_knowledge = cag_knowledge
            state.cag_retrieval_time = time.time() - start_time
            
            logger.info(f"CAG knowledge assembled: {len(cag_knowledge)} items")
            
        except Exception as e:
            logger.error(f"CAG knowledge assembly failed: {e}")
            state.errors.append(f"CAG knowledge error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"CAG knowledge assembly took {processing_time:.3f}s")
        
        return state
    
    async def _retrieve_rag_context_node(self, state: AgentState) -> AgentState:
        """Retrieve dynamic RAG context."""
        start_time = time.time()
        
        try:
            latest_message = state.get_latest_human_message()
            if not latest_message:
                return state
            
            # Get dynamic knowledge through RAG
            rag_results = await self.knowledge_service.get_rag_knowledge(
                query=latest_message.content,
                guest_profile=state.guest_profile,
                cultural_context=state.cultural_context,
                intent=state.intent.primary_intent if state.intent else "unclear"
            )
            
            state.rag_results = rag_results
            state.rag_retrieval_time = time.time() - start_time
            
            logger.info(f"RAG knowledge retrieved: {len(rag_results)} results")
            
        except Exception as e:
            logger.error(f"RAG knowledge retrieval failed: {e}")
            state.errors.append(f"RAG knowledge error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"RAG knowledge retrieval took {processing_time:.3f}s")
        
        return state
    
    async def _detect_opportunities_node(self, state: AgentState) -> AgentState:
        """Detect upselling and conversion opportunities."""
        start_time = time.time()
        
        try:
            if not state.intent or not state.guest_profile:
                return state
            
            # Detect upselling opportunities
            if state.settings.upselling_enabled:
                upsell_opportunity = await self._detect_upselling_opportunity(state)
                if upsell_opportunity and upsell_opportunity.confidence_score > state.settings.upselling_success_threshold:
                    action = Action(
                        action_type=ActionType.CREATE_UPSELLING_OPPORTUNITY,
                        parameters={"opportunity": upsell_opportunity.dict()},
                        priority=3,
                        revenue_impact=upsell_opportunity.revenue_potential
                    )
                    state.actions.append(action)
            
            # Detect direct booking conversion opportunities
            if state.settings.direct_booking_tracking:
                conversion_opportunity = await self._detect_conversion_opportunity(state)
                if conversion_opportunity and conversion_opportunity.conversion_potential in ["medium", "high"]:
                    action = Action(
                        action_type=ActionType.SEND_PROACTIVE_MESSAGE,
                        parameters={"conversion": conversion_opportunity.dict()},
                        priority=2
                    )
                    state.actions.append(action)
            
            logger.info(f"Opportunities detected: {len(state.actions)} actions planned")
            
        except Exception as e:
            logger.error(f"Opportunity detection failed: {e}")
            state.errors.append(f"Opportunity detection error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Opportunity detection took {processing_time:.3f}s")
        
        return state
    
    async def _plan_response_node(self, state: AgentState) -> AgentState:
        """Plan the response strategy using Instructor."""
        start_time = time.time()
        
        try:
            # Prepare comprehensive context for response planning
            context = {
                "message": state.get_latest_human_message().content if state.get_latest_human_message() else "",
                "intent": state.intent.dict() if state.intent else {},
                "emotional_state": state.emotional_state.dict() if state.emotional_state else {},
                "cultural_context": state.cultural_context.dict() if state.cultural_context else {},
                "guest_profile": state.guest_profile.dict() if state.guest_profile else {},
                "cag_knowledge": state.cag_knowledge,
                "rag_results": [result.dict() for result in state.rag_results],
                "opportunities": [action.dict() for action in state.actions]
            }
            
            # Use Instructor to create response plan
            response_plan = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=ResponsePlan,
                messages=[
                    SystemMessage(content=self._get_response_planning_prompt()),
                    HumanMessage(content=f"Plan response for: {context}")
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            state.response_plan = response_plan
            logger.info(f"Response planned: {response_plan.response_strategy}")
            
        except Exception as e:
            logger.error(f"Response planning failed: {e}")
            state.errors.append(f"Response planning error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Response planning took {processing_time:.3f}s")
        
        return state
    
    async def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate culturally appropriate response using Instructor."""
        start_time = time.time()
        
        try:
            if not state.response_plan:
                return state
            
            # Generate culturally appropriate response
            cultural_response = self.instructor_client.chat.completions.create(
                model=self.settings.openai_model,
                response_model=CulturalResponse,
                messages=[
                    SystemMessage(content=self._get_response_generation_prompt()),
                    HumanMessage(content=f"Generate response based on plan: {state.response_plan.dict()}")
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            # Create AI message
            ai_message = AIMessage(content=cultural_response.message)
            state.add_message(ai_message)
            
            # Add send message action
            send_action = Action(
                action_type=ActionType.SEND_MESSAGE,
                parameters={
                    "message": cultural_response.message,
                    "language": cultural_response.language,
                    "cultural_markers": cultural_response.cultural_markers
                },
                priority=1
            )
            state.actions.insert(0, send_action)  # Highest priority
            
            logger.info(f"Response generated in {cultural_response.language}: {len(cultural_response.message)} chars")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state.errors.append(f"Response generation error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            state.llm_processing_time = processing_time
            logger.debug(f"Response generation took {processing_time:.3f}s")
        
        return state
    
    async def _execute_actions_node(self, state: AgentState) -> AgentState:
        """Execute planned actions."""
        start_time = time.time()
        
        try:
            # Sort actions by priority
            state.actions.sort(key=lambda x: x.priority)
            
            for action in state.actions:
                try:
                    if action.action_type == ActionType.SEND_MESSAGE:
                        await self.whatsapp_service.send_message(
                            to=state.raw_message.get("from", ""),
                            message=action.parameters.get("message", ""),
                            language=action.parameters.get("language", "en")
                        )
                    
                    elif action.action_type == ActionType.UPDATE_GUEST_PROFILE:
                        await self.pms_service.update_guest_profile(
                            phone_number=state.raw_message.get("from", ""),
                            updates=action.parameters
                        )
                    
                    elif action.action_type == ActionType.ESCALATE_TO_HUMAN:
                        await self._escalate_to_human(state, action.parameters)
                    
                    # Add more action types as needed
                    
                    logger.info(f"Executed action: {action.action_type}")
                    
                except Exception as e:
                    logger.error(f"Action execution failed for {action.action_type}: {e}")
                    state.errors.append(f"Action execution error: {str(e)}")
            
        except Exception as e:
            logger.error(f"Actions execution failed: {e}")
            state.errors.append(f"Actions execution error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Actions execution took {processing_time:.3f}s")
        
        return state
    
    async def _log_interaction_node(self, state: AgentState) -> AgentState:
        """Log interaction for analytics and learning."""
        start_time = time.time()
        
        try:
            # Calculate total processing time
            state.processing_time = time.time() - state.timestamp.timestamp()
            
            # Log interaction details
            interaction_data = {
                "conversation_id": state.conversation_id,
                "session_id": state.session_id,
                "guest_phone": state.raw_message.get("from", ""),
                "intent": state.intent.dict() if state.intent else {},
                "cultural_context": state.cultural_context.dict() if state.cultural_context else {},
                "processing_time": state.processing_time,
                "actions_executed": len(state.actions),
                "errors": state.errors,
                "satisfaction_impact": state.response_plan.estimated_satisfaction_impact if state.response_plan else 0.0
            }
            
            # Store interaction for analytics
            await self._store_interaction(interaction_data)
            
            logger.info(f"Interaction logged for conversation {state.conversation_id}")
            
        except Exception as e:
            logger.error(f"Interaction logging failed: {e}")
            state.errors.append(f"Logging error: {str(e)}")
        
        finally:
            processing_time = time.time() - start_time
            logger.debug(f"Interaction logging took {processing_time:.3f}s")
        
        return state
    
    # Helper methods for prompts and utilities
    
    def _get_intent_classification_prompt(self) -> str:
        """Get system prompt for intent classification."""
        return """You are an expert at classifying guest messages in a Moroccan riad hospitality context.
        
        Analyze the message and classify the primary intent, considering:
        - Cultural context and communication style
        - Urgency level and emotional state
        - Business implications (booking, service, complaint, etc.)
        - Need for human escalation
        
        Be especially sensitive to:
        - Religious considerations (prayer times, halal requirements)
        - Cultural nuances in communication
        - Family and group dynamics
        - Local customs and expectations
        """
    
    def _get_emotional_analysis_prompt(self) -> str:
        """Get system prompt for emotional analysis."""
        return """You are an expert at analyzing emotional states in multicultural hospitality contexts.
        
        Analyze the emotional content considering:
        - Cultural expression patterns
        - Religious and social context
        - Communication style variations
        - Implicit vs explicit emotional indicators
        
        Pay special attention to:
        - Moroccan and Arab cultural emotional expressions
        - French sophistication and directness
        - English enthusiasm and informality
        - Family honor and respect dynamics
        """
    
    def _get_response_planning_prompt(self) -> str:
        """Get system prompt for response planning."""
        return """You are a master hospitality concierge planning responses for Moroccan riad guests.
        
        Create a comprehensive response plan that:
        - Addresses the guest's needs appropriately
        - Incorporates cultural intelligence
        - Optimizes for guest satisfaction and revenue
        - Maintains authentic Moroccan hospitality
        
        Consider:
        - Cultural communication preferences
        - Appropriate timing and formality
        - Upselling opportunities (if appropriate)
        - Local recommendations and cultural experiences
        - Religious and cultural sensitivities
        """
    
    def _get_response_generation_prompt(self) -> str:
        """Get system prompt for response generation."""
        return """You are the voice of a luxury Moroccan riad, speaking with warmth, cultural intelligence, and genuine care.
        
        Generate responses that:
        - Reflect authentic Moroccan hospitality (karam)
        - Adapt to the guest's cultural background
        - Use appropriate language and formality
        - Include relevant cultural context
        - Maintain professional warmth
        
        Cultural guidelines:
        - Arabic: Respectful, family-oriented, traditional values
        - French: Sophisticated, culturally appreciative, refined
        - English: Enthusiastic, adventurous, experience-focused
        - Always respect Islamic values and Moroccan customs
        """
    
    async def _detect_upselling_opportunity(self, state: AgentState) -> Optional[UpsellOpportunity]:
        """Detect upselling opportunities using Instructor."""
        # Implementation for upselling detection
        pass
    
    async def _detect_conversion_opportunity(self, state: AgentState) -> Optional[DirectBookingConversion]:
        """Detect direct booking conversion opportunities."""
        # Implementation for conversion detection  
        pass
    
    async def _escalate_to_human(self, state: AgentState, parameters: Dict[str, Any]) -> None:
        """Escalate conversation to human staff."""
        # Implementation for human escalation
        pass
    
    async def _store_interaction(self, interaction_data: Dict[str, Any]) -> None:
        """Store interaction data for analytics."""
        # Implementation for interaction storage
        pass
    
    async def process_message(self, raw_message: Dict[str, Any]) -> AgentState:
        """Process a WhatsApp message through the agent workflow."""
        
        # Create initial state
        state = AgentState(
            raw_message=raw_message,
            conversation_id=f"conv_{raw_message.get('from', '')}_{int(time.time())}",
            session_id=f"session_{int(time.time())}"
        )
        
        try:
            # Run the workflow
            result = await self.workflow.ainvoke(state)
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            state.errors.append(f"Workflow error: {str(e)}")
            state.fallback_triggered = True
            return state
