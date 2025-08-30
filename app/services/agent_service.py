"""
Agent Service - Main orchestrator for the Riad Concierge AI system.
"""

import asyncio
from typing import Any, Dict, Optional

from loguru import logger

from app.agents.riad_agent import RiadConciergeAgent
from app.core.config import get_settings
from app.models.agent_state import AgentState


class AgentService:
    """Main agent service orchestrator."""
    
    def __init__(self):
        self.settings = get_settings()
        self.agent: Optional[RiadConciergeAgent] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the agent service."""
        try:
            logger.info("Initializing Agent Service...")
            
            # Initialize the main agent
            self.agent = RiadConciergeAgent()
            
            # Perform any additional initialization
            await self._setup_services()
            
            self._initialized = True
            logger.info("✅ Agent Service initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Agent Service initialization failed: {e}")
            raise
    
    async def _setup_services(self) -> None:
        """Setup supporting services."""
        # Initialize any additional services needed
        pass
    
    async def process_message(self, message_data: Dict[str, Any]) -> AgentState:
        """Process a message through the agent workflow."""
        if not self._initialized or not self.agent:
            raise RuntimeError("Agent service not initialized")
        
        try:
            result = await self.agent.process_message(message_data)
            return result
            
        except Exception as e:
            logger.error(f"Message processing failed: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup agent service resources."""
        try:
            logger.info("Cleaning up Agent Service...")
            
            if self.agent:
                # Perform any cleanup needed
                pass
            
            self._initialized = False
            logger.info("✅ Agent Service cleanup completed")
            
        except Exception as e:
            logger.error(f"Agent Service cleanup failed: {e}")
    
    @property
    def is_initialized(self) -> bool:
        """Check if agent service is initialized."""
        return self._initialized
