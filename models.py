"""
Pydantic models for the gym lead qualification chatbot simulator.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


class BigFiveTraits(BaseModel):
    """Big Five personality traits on a 1-10 scale."""
    openness: int = Field(ge=1, le=10)
    conscientiousness: int = Field(ge=1, le=10)
    extraversion: int = Field(ge=1, le=10)
    agreeableness: int = Field(ge=1, le=10)
    neuroticism: int = Field(ge=1, le=10)


class Intent(str, Enum):
    """Possible fitness intents/goals."""
    WEIGHT_LOSS = "weight_loss"
    STRESS_RELIEF = "stress_relief_mental_health"
    BOXING_TECHNIQUE = "learn_boxing_technique"
    GENERAL_FITNESS = "general_fitness"
    SOCIAL_COMMUNITY = "social_community"
    JUST_FREE_CLASS = "just_wants_free_class"


class ObjectionType(str, Enum):
    """Common objections prospects may have."""
    PRICE = "price"
    TIME_COMMITMENT = "time_commitment"
    INJURY_CONCERNS = "injury_concerns"
    INTIMIDATION = "intimidation_factor"
    LOCATION_PARKING = "location_parking"
    JUST_LOOKING = "just_looking"


class ReadinessLevel(str, Enum):
    """Lead readiness levels."""
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class ProspectProfile(BaseModel):
    """Complete prospect profile for simulation."""
    big_five: BigFiveTraits
    true_intent: Intent
    objection_type: Optional[ObjectionType]
    readiness_level: ReadinessLevel
    age_range: str  # e.g., "25-35"
    fitness_background: str  # e.g., "beginner", "intermediate", "couch_to_5k"


class ConversationMessage(BaseModel):
    """Individual message in a conversation."""
    role: Literal["prospect", "sales"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    tokens_used: Optional[int] = None


class IntentDetection(BaseModel):
    """Sales LLM's detected intent output."""
    detected_intent: Intent
    confidence_level: float = Field(ge=0.0, le=1.0)
    reasoning: str
    best_time_to_visit: Optional[str] = None  # morning/evening/weekend


class ConversationOutcome(str, Enum):
    """Possible conversation outcomes."""
    AGREED_FREE_CLASS = "agreed_to_free_class"
    NOT_INTERESTED = "not_interested"
    REACHED_MESSAGE_LIMIT = "reached_message_limit"


class ConversationLog(BaseModel):
    """Complete log of a single simulated conversation."""
    conversation_id: str
    prospect_profile: ProspectProfile
    messages: List[ConversationMessage]
    intent_detection: Optional[IntentDetection]
    outcome: ConversationOutcome
    total_tokens_used: int
    conversation_length: int  # number of back-and-forth exchanges
    intent_match: bool  # True if detected_intent == true_intent
    timestamp: datetime = Field(default_factory=datetime.now)


class SimulationConfig(BaseModel):
    """Configuration for running simulations."""
    # Model settings
    prospect_model: str = "gpt-4o-mini"
    sales_model: str = "gpt-4o-mini"
    prospect_temperature: float = 0.85
    sales_temperature: float = 0.3
    prospect_max_tokens: int = 100
    sales_max_tokens: int = 120
    
    # Conversation limits
    max_message_exchanges: int = 3
    
    # Simulation settings
    num_simulations: int = 100
    
    # API settings
    api_provider: Literal["openai", "groq", "together"] = "openai"
    api_key: Optional[str] = None
