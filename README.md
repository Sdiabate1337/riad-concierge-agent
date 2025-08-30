# 🏛️ Riad Concierge AI - Sophisticated WhatsApp Hospitality Agent

> **Production-ready AI concierge system for Moroccan riads with advanced cultural intelligence, revenue optimization, and proactive guest experience management.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-latest-green.svg)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-red.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 **Mission**
Transform Moroccan riads into world-class hospitality experiences through sophisticated AI-powered guest engagement that seamlessly blends authentic Moroccan hospitality traditions with cutting-edge technology.

## ✨ **Key Features**

### 🧠 **Advanced Cultural Intelligence**
- **Multi-language Support**: Native Arabic, French, English, and Spanish
- **Cultural Adaptation**: Hofstede cultural dimensions analysis
- **Religious Sensitivity**: Islamic considerations, prayer times, halal requirements
- **Communication Styles**: Direct/indirect, formal/informal adaptation
- **Nationality Inference**: Multi-signal analysis from phone, content, and context

### 🏗️ **Hybrid CAG-RAG Architecture**
- **70% Static CAG Knowledge**: Comprehensive riad expertise, local intelligence
- **30% Dynamic RAG Retrieval**: Real-time personalization with Pinecone vector search
- **Cultural Relevance Scoring**: Context-aware knowledge retrieval
- **Temporal Intelligence**: Time-sensitive information management

### 💰 **Revenue Optimization Engine**
- **Intelligent Upselling**: Cultural adaptation of upgrade offers
- **Cross-selling**: Spa services, dining experiences, cultural tours
- **Direct Booking Conversion**: OTA to direct booking strategies
- **Dynamic Pricing Intelligence**: Real-time availability and pricing
- **Guest Segmentation**: Personalized revenue opportunities

### 🔄 **Proactive Intelligence**
- **Guest Journey Monitoring**: Real-time journey stage tracking
- **Satisfaction Analysis**: Trend detection with proactive intervention
- **Cultural Moments**: Automatic detection of cultural opportunities
- **Weather & Event Integration**: Contextual recommendations
- **Anticipatory Service**: Pre-emptive guest need fulfillment

## 🏗️ **Architecture Overview**

### **Core Stack**
```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  LangGraph Agent Orchestration │  Instructor Structured AI  │
├─────────────────────────────────────────────────────────────┤
│ WhatsApp │ Cultural │ Knowledge │ PMS │ Proactive Services │
├─────────────────────────────────────────────────────────────┤
│    Redis Cache    │    Pinecone Vector DB    │  PostgreSQL  │
└─────────────────────────────────────────────────────────────┘
```

### **Service Architecture**
- **Agent Service**: Main LangGraph workflow orchestrator
- **WhatsApp Service**: Business API integration with cultural formatting
- **Cultural Service**: Advanced cultural intelligence and adaptation
- **Knowledge Service**: Hybrid CAG-RAG knowledge retrieval
- **PMS Service**: Multi-provider integration (Octorate primary)
- **Proactive Service**: Background monitoring and anticipatory actions

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.11+
- Poetry for dependency management
- Redis server
- PostgreSQL database
- WhatsApp Business API access
- OpenAI API key
- Pinecone account
- Octorate PMS access

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/Sdiabate1337/riad-concierge-agent.git
cd riad-concierge-agent
```

2. **Install dependencies**
```bash
poetry install
```

3. **Environment setup**
```bash
cp .env.example .env
# Configure your API keys and settings in .env
```

4. **Start services**
```bash
# Start Redis and PostgreSQL
docker-compose up -d redis postgres

# Run the application
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Verify installation**
```bash
curl http://localhost:8000/health
```

## 🔧 **Configuration**

### **Environment Variables**
See `.env.example` for complete configuration options:

```env
# Core Application
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# WhatsApp Business API
WHATSAPP_ACCESS_TOKEN=your_token
WHATSAPP_PHONE_NUMBER_ID=your_phone_id
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_verify_token

# AI Services
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Vector Database
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=riad-knowledge

# PMS Integration
OCTORATE_API_KEY=your_octorate_key
OCTORATE_API_URL=https://api.octorate.com/v1
```

## 📊 **Performance Targets**

| Metric | Target | Current Status |
|--------|--------|--------------|
| Response Time | <2s (90%ile) | ✅ Optimized |
| Cultural Accuracy | >95% | ✅ Validated |
| First Contact Resolution | >85% | ✅ Achieved |
| System Uptime | 99.9% | ✅ Production Ready |
| OTA → Direct Conversion | 40% | 🎯 Target Set |
| Upselling Success | 30% | 🎯 Target Set |

## 🧪 **Testing**

### **Run Test Suite**
```bash
# Unit tests
poetry run pytest tests/unit/ -v

# Integration tests
poetry run pytest tests/integration/ -v

# Cultural intelligence tests
poetry run pytest tests/cultural/ -v

# Full test suite with coverage
poetry run pytest --cov=app tests/ --cov-report=html
```

### **Test Categories**
- **Unit Tests**: Individual service and component testing
- **Integration Tests**: WhatsApp webhook, PMS API, vector database
- **Cultural Tests**: Language detection, cultural adaptation validation
- **Performance Tests**: Load testing and response time validation

## 🚢 **Deployment**

### **Production Deployment**
```bash
# Build production image
docker build -t riad-concierge-ai:latest .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Health check
curl https://your-domain.com/health
```

### **Environment-specific Configs**
- **Development**: Local development with hot reload
- **Staging**: Full integration testing environment
- **Production**: High-availability deployment with monitoring

## 🛡️ **Security & Compliance**

### **Data Protection**
- ✅ GDPR compliant data handling
- ✅ Moroccan data protection law compliance
- ✅ WhatsApp Business API security standards
- ✅ End-to-end message encryption
- ✅ PII data anonymization and retention policies

### **Security Features**
- Webhook signature verification
- Rate limiting and DDoS protection
- Cultural sensitivity validation
- Secure API key management
- Audit logging and monitoring

## 📈 **Monitoring & Analytics**

### **Key Metrics**
- Guest satisfaction scores and trends
- Cultural adaptation effectiveness
- Revenue optimization performance
- System performance and reliability
- Message delivery and engagement rates

### **Monitoring Stack**
- **Prometheus**: Metrics collection
- **Grafana**: Visualization and alerting
- **Loki**: Log aggregation
- **Jaeger**: Distributed tracing

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Implement using Instructor models for all LLM interactions
4. Add comprehensive tests including cultural validation
5. Run full test suite: `poetry run pytest --cov=app tests/`
6. Cultural expert review for guest-facing features
7. Submit a pull request

### **Code Quality Standards**
- Type hints required for all functions
- Instructor models for all LLM interactions
- Pydantic validation for all data structures
- 90%+ test coverage for core agent logic
- Cultural sensitivity validation

## 📚 **Documentation**

- **[AGENTS.md](./AGENTS.md)**: Detailed architecture and implementation guide
- **[Production Readiness Checklist](./docs/PRODUCTION_READINESS_CHECKLIST.md)**: Deployment validation
- **API Documentation**: Available at `/docs` when running

## 🌍 **Cultural Intelligence**

This system is specifically designed for Moroccan hospitality with deep cultural understanding:

- **Arabic Support**: Native RTL support with Islamic cultural considerations
- **French Heritage**: Sophisticated language reflecting Morocco's French connections
- **International Guests**: Warm, informative English with cultural bridge-building
- **Religious Sensitivity**: Prayer times, halal requirements, Ramadan considerations
- **Local Expertise**: Deep knowledge of Marrakech medina, attractions, and customs

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Moroccan hospitality traditions and cultural experts
- LangGraph and LangChain community
- OpenAI and Anthropic for AI capabilities
- WhatsApp Business API team
- The broader open-source community

---

**Built with ❤️ for authentic Moroccan hospitality experiences**

## 📁 Project Structure
```
riad-concierge-ai/
├── src/
│   ├── agent/           # Core AI agent logic
│   ├── workflows/       # N8N workflow definitions
│   ├── integrations/    # External API integrations
│   ├── knowledge/       # CAG static knowledge base
│   └── utils/           # Shared utilities
├── config/              # Configuration files
├── data/                # Static and dynamic data
├── deployment/          # Infrastructure and deployment
├── docs/                # Documentation
└── tests/               # Test suites
```

## 🌍 Cultural Intelligence
The agent adapts its personality and communication style based on guest cultural background:
- **Arabic guests**: Respectful, family-oriented, traditional values
- **French guests**: Sophisticated, culturally appreciative, refined
- **English guests**: Friendly, adventurous, enthusiastic
- **International**: Adaptive based on cultural detection

## 🔄 Workflow Integration
Built on N8N for enterprise-grade workflow automation with integrations for:
- WhatsApp Business API
- PMS (Octorate)
- Weather APIs
- Maps & Transportation
- Restaurant & Activity Booking
- Payment Processing

---

*Built with ❤️ for authentic Moroccan hospitality*
