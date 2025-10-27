"""
Pydantic models for the gym lead qualification chatbot simulator.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


class Intent(str, Enum):
    """Possible fitness intents/goals."""
    WEIGHT_LOSS = "weight_loss"
    STRESS_RELIEF = "stress_relief_mental_health"
    BOXING_TECHNIQUE = "learn_boxing_technique"
    GENERAL_FITNESS = "general_fitness"
    SOCIAL_COMMUNITY = "social_community"
    JUST_FREE_CLASS = "just_wants_free_class"

class IntentDetection(BaseModel):
    """Sales LLM's detected intent output."""
    detected_intent: Optional[Intent] = None
    confidence_level: Optional[float] = Field(None, ge=0.0, le=1.0)
    reasoning: Optional[str] = None
    best_time_to_visit: Optional[str] = None  # morning/evening/weekend

class salesResponse(BaseModel):
    """Sales LLM's response structure."""
    message: str = Field(..., description="The sales agent's message to the prospect.") 
    intent_detection: Optional[IntentDetection] = Field(..., description="The detected intent from the prospect's response.")

class OutcomeChoices(str, Enum):
    """Possible conversation outcomes."""
    AGREED_FREE_CLASS= "agreed_to_free_class"
    NOT_INTERESTED  = "not_interested"
    CONTINUE ="continue"
    #REACHED_MESSAGE_LIMIT = "reached_message_limit"  #remove this

class ConversationOutcome(BaseModel):
    """Result of conversation. and whether or not to proceed"""
    outcome: OutcomeChoices = Field(..., description="The determined outcome of the conversation.")
    reasoning: str = Field(..., description="Explanation for the determined outcome.")
    should_end: bool = Field(..., description="Indicates if the conversation should end.")

class SimulationConfig(BaseModel):
    """Configuration for running simulations."""
    # Model settings
    prospect_model: str = "gpt-4o-mini"
    sales_model: str = "anthropic/claude-3.5-haiku"
    prospect_temperature: float = 0.85
    sales_temperature: float = 0.6
    prospect_max_tokens: int = 100
    sales_max_tokens: int = 250
    
    # Conversation limits
    max_message_exchanges: int = 3
    
    # Simulation settings
    num_simulations: int = 100
    
    # API settings
    api_provider: Literal["openai", "groq", "together"] = "openai"
    api_key: Optional[str] = None