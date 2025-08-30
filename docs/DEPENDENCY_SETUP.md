# Dependency Setup Guide

## Overview

This guide provides step-by-step instructions for setting up the development environment and installing all required dependencies for the Riad Concierge AI system.

## Prerequisites

- Python 3.11+ installed
- pip or Poetry package manager
- Git for version control

## Installation Options

### Option 1: Using Poetry (Recommended)

Poetry provides better dependency management and virtual environment isolation.

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell

# Run tests
poetry run python scripts/run_tests.py --all
```

### Option 2: Using pip

If Poetry is not available, you can install dependencies using pip:

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install core dependencies
pip install -r requirements.txt

# Or install manually:
pip install langgraph instructor pydantic fastapi uvicorn redis pinecone-client openai anthropic httpx loguru python-dotenv langdetect phonenumbers pytz schedule websockets pycountry

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy pre-commit

# Run tests
python scripts/run_tests.py --all
```

## Required Dependencies

### Core Application Dependencies

```toml
# From pyproject.toml [tool.poetry.dependencies]
python = "^3.11"
langgraph = "^0.2.0"
instructor = "^1.0.0"
pydantic = "^2.0.0"
fastapi = "^0.100.0"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
redis = "^5.0.0"
pinecone-client = "^3.0.0"
openai = "^1.0.0"
anthropic = "^0.25.0"
httpx = "^0.25.0"
loguru = "^0.7.0"
python-dotenv = "^1.0.0"
langdetect = "^1.0.9"
phonenumbers = "^8.13.0"
pytz = "^2023.3"
schedule = "^1.2.0"
websockets = "^12.0"
```

### Development Dependencies

```toml
# From pyproject.toml [tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
pytest-asyncio = "^0.21.0"
pytest-cov = "^4.1.0"
black = "^23.9.0"
isort = "^5.12.0"
flake8 = "^6.1.0"
mypy = "^1.6.0"
pre-commit = "^3.5.0"
```

## Environment Configuration

### 1. Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your API keys and configuration
nano .env  # or use your preferred editor
```

### 2. Required Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Anthropic Configuration (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# WhatsApp Business API
WHATSAPP_ACCESS_TOKEN=your_whatsapp_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_here
WHATSAPP_WEBHOOK_VERIFY_TOKEN=your_webhook_verify_token_here

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here
PINECONE_INDEX_NAME=riad-concierge-knowledge

# PMS Integration (Octorate)
OCTORATE_API_KEY=your_octorate_api_key_here
OCTORATE_HOTEL_ID=your_hotel_id_here

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
DEBUG=true

# Security
SECRET_KEY=your_secret_key_here
WEBHOOK_SECRET=your_webhook_secret_here
```

## External Services Setup

### 1. Redis Setup

**Option A: Using Docker**
```bash
docker run -d --name redis -p 6379:6379 redis:alpine
```

**Option B: Local Installation**
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
```

### 2. Pinecone Setup

1. Sign up at [Pinecone.io](https://pinecone.io)
2. Create a new index:
   - Name: `riad-concierge-knowledge`
   - Dimensions: `1536` (for OpenAI embeddings)
   - Metric: `cosine`
3. Get your API key from the dashboard

### 3. OpenAI Setup

1. Sign up at [OpenAI](https://openai.com)
2. Generate an API key
3. Ensure you have access to GPT-4 (recommended)

### 4. WhatsApp Business API Setup

1. Set up a Meta Business Account
2. Create a WhatsApp Business App
3. Get your access token and phone number ID
4. Set up webhook endpoints

## Verification

### 1. Run Simplified Tests

```bash
# Test core system structure
./scripts/simple_test_runner.py
```

Expected output:
```
🎉 All simplified tests passed! Core system structure is valid.
📝 Next step: Install dependencies and run full test suite.
```

### 2. Run Full Test Suite

```bash
# Run all tests
./scripts/run_tests.py --all --verbose

# Run specific test categories
./scripts/run_tests.py --unit
./scripts/run_tests.py --integration
./scripts/run_tests.py --performance
```

### 3. Start the Application

```bash
# Start the FastAPI server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the startup script
python app/main.py
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Redis Connection Error**: Ensure Redis is running
   ```bash
   redis-cli ping  # Should return PONG
   ```

3. **OpenAI API Error**: Check your API key and quota
   ```bash
   curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
   ```

4. **Import Errors**: Check Python path and virtual environment
   ```bash
   which python
   python -c "import sys; print(sys.path)"
   ```

### Dependency Conflicts

If you encounter dependency conflicts:

1. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

2. **Use fresh virtual environment**:
   ```bash
   rm -rf venv
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install dependencies one by one**:
   ```bash
   pip install langgraph
   pip install instructor
   # ... continue with other dependencies
   ```

## Development Workflow

### 1. Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 2. Code Formatting

```bash
# Format code with black
black app/ tests/ scripts/

# Sort imports with isort
isort app/ tests/ scripts/

# Lint with flake8
flake8 app/ tests/ scripts/
```

### 3. Type Checking

```bash
# Run mypy type checking
mypy app/
```

## Production Deployment

For production deployment, additional considerations:

1. **Environment Variables**: Use secure secret management
2. **Database**: Set up persistent Redis or use Redis Cloud
3. **Monitoring**: Configure logging and metrics collection
4. **Security**: Enable HTTPS and proper authentication
5. **Scaling**: Consider load balancing and horizontal scaling

## Support

If you encounter issues:

1. Check the [Testing Strategy](TESTING_STRATEGY.md) documentation
2. Review the [Production Readiness Checklist](PRODUCTION_READINESS_CHECKLIST.md)
3. Examine log files for detailed error messages
4. Ensure all environment variables are properly configured

## Next Steps

After successful dependency setup:

1. Run the full test suite to validate functionality
2. Configure external service integrations
3. Perform integration testing with real APIs
4. Set up monitoring and logging
5. Prepare for production deployment
