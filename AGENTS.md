# AGENTS.md

## Project Overview
Production-ready WhatsApp AI agent for Moroccan riad hospitality using LangGraph + Instructor architecture. Transforms OTA-dependent riads into direct-booking powerhouses through culturally intelligent automation and revenue optimization.

## Setup Commands
- Install dependencies: `poetry install`
- Setup environment: `cp .env.example .env && poetry run python setup_env.py`
- Initialize vector database: `poetry run python scripts/init_vectordb.py`
- Start development server: `poetry run python main.py`
- Run workflows: `poetry run python -m langgraph dev`
- Run tests: `poetry run pytest`

## Development Environment
- Python 3.11+ with Poetry dependency management
- LangGraph for agent orchestration and state management
- Instructor for structured LLM outputs and validation
- Pydantic v2 for type safety and data validation
- FastAPI for webhook endpoints and API services
- Redis for caching and session management
- Pinecone for vector storage and semantic search

## Architecture Guidelines

### Core Stack
```python
# Primary dependencies
dependencies = {
    "langgraph": ">=0.2.0",  # Agent workflow orchestration
    "instructor": ">=1.0.0",  # Structured LLM outputs
    "pydantic": ">=2.0.0",   # Data validation
    "fastapi": ">=0.100.0",  # API endpoints
    "redis": ">=5.0.0",      # Caching layer
    "pinecone-client": ">=3.0.0",  # Vector database
    "openai": ">=1.0.0",     # Primary LLM
    "anthropic": ">=0.25.0"  # Backup LLM
}
```

### Agent Graph Structure
```python
# LangGraph state definition
class AgentState(BaseModel):
    messages: List[HumanMessage | AIMessage]
    guest_profile: GuestProfile
    cultural_context: CulturalContext
    cag_knowledge: Dict[str, Any]
    rag_results: List[RetrievalResult]
    intent: IntentClassification
    response_plan: ResponsePlan
    actions: List[Action]
```

### Hybrid CAG-RAG Implementation
- **CAG Static Cache**: Use instructor models for structured knowledge retrieval
- **RAG Dynamic Layer**: Pinecone vector search with metadata filtering
- **Context Assembly**: LangGraph nodes for knowledge injection and personalization
- **Response Generation**: Instructor-validated outputs with cultural appropriateness checks

## Core Agent Workflow

### LangGraph Node Structure
```python
# Main processing graph
workflow = StateGraph(AgentState)
workflow.add_node("validate_input", validate_message_node)
workflow.add_node("detect_language", language_detection_node)
workflow.add_node("load_guest_profile", guest_profile_node)
workflow.add_node("classify_intent", intent_classification_node)
workflow.add_node("assemble_cag_context", cag_knowledge_node)
workflow.add_node("retrieve_rag_context", rag_retrieval_node)
workflow.add_node("generate_response", response_generation_node)
workflow.add_node("execute_actions", action_execution_node)
workflow.add_node("log_interaction", analytics_node)
```

### Instructor Model Definitions
```python
# Structured outputs for agent responses
class CulturalResponse(BaseModel):
    message: str = Field(description="Culturally appropriate response")
    language: str = Field(description="Detected/chosen language")
    cultural_markers: List[str] = Field(description="Cultural elements included")
    tone_adaptation: str = Field(description="Applied communication style")

class ServiceAction(BaseModel):
    action_type: Literal["booking", "information", "escalation", "upselling"]
    priority: Literal["immediate", "scheduled", "proactive"]
    staff_notification: bool
    guest_confirmation_required: bool
    revenue_impact: Optional[float]
```

## Cultural Intelligence Implementation

### Language Detection & Adaptation
```python
# Cultural persona switching
class CulturalPersona(BaseModel):
    nationality: str
    communication_style: str
    service_preferences: List[str]
    upselling_approach: str
    religious_considerations: Optional[List[str]]
```

### Multi-Language Response Generation
- Arabic: Respectful, family-oriented, traditional values
- French: Sophisticated, culturally appreciative, refined
- English: Enthusiastic, adventurous, experience-focused
- Use Instructor to validate cultural appropriateness of all responses

## Revenue Optimization Framework

### Upselling Detection
```python
class UpsellOpportunity(BaseModel):
    trigger_signals: List[str]
    confidence_score: float
    recommended_services: List[str]
    optimal_timing: datetime
    expected_conversion_rate: float
```

### Direct Booking Conversion
```python
class ConversionOpportunity(BaseModel):
    guest_receptivity_score: float
    conversion_tactics: List[str]
    incentive_recommendations: List[str]
    follow_up_schedule: List[datetime]
```

## Testing Strategy

### Unit Testing
```bash
# Test cultural response generation
poetry run pytest tests/test_cultural_intelligence.py -v

# Test revenue optimization logic
poetry run pytest tests/test_revenue_optimization.py -v

# Test LangGraph workflow execution
poetry run pytest tests/test_agent_workflows.py -v
```

### Integration Testing
```bash
# Test WhatsApp webhook integration
poetry run pytest tests/integration/test_whatsapp_flow.py

# Test PMS integration
poetry run pytest tests/integration/test_octorate_api.py

# Test vector database retrieval
poetry run pytest tests/integration/test_rag_system.py
```

### Cultural Validation Testing
- Moroccan hospitality expert review of responses
- Multi-language accuracy validation
- Religious sensitivity compliance testing
- Guest journey simulation testing

## Data Pipeline Management

### Vector Database Setup
```python
# Pinecone index configuration
index_config = {
    "dimension": 3072,  # text-embedding-3-large
    "metric": "cosine",
    "metadata_config": {
        "cultural_context": "string",
        "guest_nationality": "string",
        "service_category": "string",
        "temporal_relevance": "number"
    }
}
```

### Knowledge Base Updates
- Static knowledge updates: `poetry run python scripts/update_cag_cache.py`
- Dynamic embeddings refresh: `poetry run python scripts/refresh_embeddings.py`
- Cultural intelligence updates: Manual expert validation required

## Performance Optimization

### Caching Strategy
- Redis for guest profiles and conversation context
- In-memory caching for frequently accessed CAG knowledge
- Vector similarity caching for common queries
- Response template caching for standard interactions

### Monitoring Requirements
```python
# Performance metrics to track
metrics = {
    "response_time_p90": "<2000ms",
    "accuracy_rate": ">95%",
    "cultural_appropriateness": "100%",
    "conversion_rate": ">40%",
    "guest_satisfaction": ">4.5/5"
}
```

## Security Implementation

### Data Protection
- Encrypt all guest data using cryptography library
- Implement GDPR compliance with automatic data deletion
- WhatsApp webhook signature verification
- Rate limiting and DDoS protection

### Privacy Considerations
```python
class PrivacyConfig(BaseModel):
    data_retention_days: int = 365
    encryption_key_rotation: int = 90
    audit_log_retention: int = 2555  # 7 years
    guest_data_anonymization: bool = True
```

## Deployment Architecture

### Production Setup
- Docker containerization with multi-stage builds
- Kubernetes deployment with auto-scaling
- PostgreSQL primary with Redis caching layer
- Monitoring with Prometheus + Grafana
- Log aggregation with ELK stack

### Environment Configuration
```bash
# Required environment variables
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
WHATSAPP_WEBHOOK_SECRET=your_webhook_secret
PINECONE_API_KEY=your_pinecone_key
OCTORATE_API_KEY=your_pms_key
REDIS_URL=redis://localhost:6379
DATABASE_URL=postgresql://user:pass@localhost/riad_db
```

## Error Handling & Fallbacks
- Graceful LLM fallback from GPT-4o to Claude 3.5 Sonnet
- Static response fallbacks for system outages
- Human escalation triggers for complex issues
- Cultural sensitivity error prevention with validation layers

## Development Workflow
1. Create feature branch with descriptive name
2. Implement using Instructor models for all LLM interactions
3. Add comprehensive tests including cultural validation
4. Run full test suite: `poetry run pytest --cov=app tests/`
5. Cultural expert review for guest-facing features
6. Performance benchmarking against targets
7. Deploy to staging for end-to-end validation

## Code Quality Standards
- Type hints required for all functions
- Instructor models for all LLM interactions
- Pydantic validation for all data structures
- 90%+ test coverage for core agent logic
- Black code formatting with 88-character line limit
- Comprehensive docstrings with cultural context notes