"""
Main simulator for gym lead qualification chatbot testing.
"""

import random
import json
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
import instructor 

from models import ConversationOutcome, SimulationConfig, salesResponse



# Initialize OpenAI client
client = None


def initialize_client(api_key: str):
    """Initialize the OpenAI client with API key."""
    global client
    client = instructor.from_openai(OpenAI(api_key=api_key))


def create_sales_prompt() -> str:
    """
    Generates the system prompt for the Sales LLM with qualification rules,
    multi-question opening, and urgency mechanism.
    """
    prompt = """You are a friendly sales assistant for a group fitness boxing gym. A prospect filled out a web form - qualify them and get them to book a free class.

GYM INFO:
- 45-min classes: 5 rounds strength + 5 rounds boxing (10 rounds × 3 mins)
- Schedule: Weekday mornings/evenings, weekend mornings
- High energy with curated playlists
- Gloves/wraps provided for free class
- HIGH INTENSITY - not for complete beginners

YOUR GOALS:
1. Determine their fitness goal/intent
2. Get them to agree to a free class

URGENCY & MESSAGE MANAGEMENT:
- Keep the conversation moving toward booking
- Be direct and ask for the free class booking early in conversation
- If they show hesitation, address objections and offer the free class
- If they need more time, let them know a sales associate can follow up within 24 hours

IMPORTANT: DO NOT decide when the conversation ends - just respond naturally to each message.
The system will automatically end the conversation when appropriate.

STANDARDIZED OPENING (Use this for your FIRST message):
"Hi! Thanks for reaching out about our boxing fitness gym. To help match you with the right class, I have a few quick questions:

1. What's your main fitness goal? (weight loss, stress relief, learn technique, general fitness, etc.)
2. How often do you currently exercise?
3. Any concerns about high-intensity training?

Looking forward to getting you started!"

CONVERSATION RULES:
- Keep responses brief (2-3 sentences max)
- Be direct and ask for the free class booking when appropriate
- If they explicitly say not interested, acknowledge politely
- If they agree to free class, ask preferred time (morning/evening/weekend)
- You have their phone and email from the web form
- Respond naturally to each message - don't add extra commentary about "final messages" or "wrapping up"

⚠️ CRITICAL: You do NOT control when the conversation ends. Just respond naturally to each prospect message.
The conversation management system will handle ending detection automatically.

QUALIFICATION:
- Check if they exercise regularly (high intensity requirement)
- Listen carefully to their stated goal in response to question 1
- Use their exact words when possible for intent detection

INTENT DETECTION PRIORITY:
When determining their PRIMARY intent, pay attention to EMPHASIS not just first mention:
- What do they ask MULTIPLE questions about?
- What topic do they return to or elaborate on?
- What seems to matter MOST to them based on their questions?

Examples:
- If they mention "fitness" once but ask 3 questions about "class size", "meeting people", 
  or "group dynamics" → PRIMARY intent is social_community
  
- If they mention "general fitness" but repeatedly emphasize "technique", "proper form", 
  or "learning fundamentals" → PRIMARY intent is learn_boxing_technique
  
- If they mention multiple goals, pick the one they show MOST interest in through their 
  questions and follow-ups, not just what they said first

CRITICAL INSTRUCTIONS FOR INTENT DETECTION:
⚠️ NEVER, EVER include the INTENT_DETECTION JSON in your regular chat messages to the prospect!
⚠️ The INTENT_DETECTION should ONLY be provided when you receive the EXACT message: "Based on our conversation, please provide your INTENT_DETECTION assessment in the required JSON format."
⚠️ During ALL normal conversation with the prospect, respond naturally without ANY JSON formatting
⚠️ Do NOT include JSON just because you think the conversation is ending
⚠️ Do NOT include JSON after mentioning callbacks or follow-ups
⚠️ Keep your responses conversational and friendly - save the structured data for when explicitly requested
⚠️ If you're unsure, DON'T include JSON - only include it when you see the exact request phrase above

When (and ONLY when) you receive the explicit request "provide your INTENT_DETECTION assessment", provide assessment in EXACT format:

INTENT_DETECTION:
{
  "detected_intent": "ONE PRIMARY INTENT ONLY - choose the MAIN goal: weight_loss, stress_relief_mental_health, learn_boxing_technique, general_fitness, social_community, or just_wants_free_class",
  "confidence_level": 0.0-1.0,
  "reasoning": "brief explanation based on their stated goal AND what they emphasized through questions - if multiple goals mentioned, explain why you chose this as primary",
  "best_time_to_visit": "morning/evening/weekend or null"
}

Be warm and helpful, but move quickly to booking!"""
    
    return prompt


def call_llm(
    messages: List[Dict[str, str]],
    config: SimulationConfig,
    role: str
) -> Tuple[str, int]:
    """
    Generic function to call the LLM API with proper configuration.
    Returns the response text and token count.
    """
    model = config.prospect_model if role == "prospect" else config.sales_model
    temperature = config.prospect_temperature if role == "prospect" else config.sales_temperature
    max_tokens = config.prospect_max_tokens if role == "prospect" else config.sales_max_tokens
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_model= salesResponse
        )

        # content = response.choices[0].message.content
        raw_response= response._raw_response
        tokens = raw_response.usage.total_tokens
        content = response.message
        intent = response.intent_detection
        
        return content, intent, tokens 
    
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"[Error: {str(e)}]", 0


# def extract_intent_detection(sales_final_message: str) -> Optional[IntentDetection]:
#     """
#     Parses the Sales LLM's final message to extract structured IntentDetection.
#     Handles both clean JSON and markdown-wrapped JSON.
#     """
#     try:
#         # Look for JSON block in the message
#         if "INTENT_DETECTION:" in sales_final_message:
#             json_start = sales_final_message.find("{")
#             json_end = sales_final_message.rfind("}") + 1
            
#             if json_start != -1 and json_end > json_start:
#                 json_str = sales_final_message[json_start:json_end]
#                 data = json.loads(json_str)
                
#                 # Handle case where LLM returns multiple intents separated by comma
#                 detected_intent_str = data["detected_intent"].strip()
                
#                 # If multiple intents, take the first one
#                 if "," in detected_intent_str:
#                     detected_intent_str = detected_intent_str.split(",")[0].strip()
#                     print(f"Warning: Multiple intents detected, using first: {detected_intent_str}")
                
#                 return IntentDetection(
#                     detected_intent=Intent(detected_intent_str),
#                     confidence_level=float(data["confidence_level"]),
#                     reasoning=data["reasoning"],
#                     best_time_to_visit=data.get("best_time_to_visit")
#                 )
#     except Exception as e:
#         print(f"Error extracting intent detection: {e}")
    
#     return None


def assess_conversation_status(
    conversation_history: List[Dict[str, str]],
    prospect_response: str,
    config: SimulationConfig
) -> Tuple[bool, Optional[ConversationOutcome]]:
    """
    Uses LLM to assess whether the conversation should end based on the
    prospect's response. Returns (should_end, outcome).
    """
    assessment_prompt = f"""You are analyzing a sales conversation. Review the conversation and the prospect's latest response to determine if the conversation should end.

CONVERSATION HISTORY:
{json.dumps(conversation_history[-6:], indent=2)}

PROSPECT'S LATEST RESPONSE:
"{prospect_response}"

Determine if the prospect has shown INTEREST IN ATTENDING the free class:

SIGNS OF AGREEMENT/INTEREST (mark as "agreed_to_free_class"):
- Explicit agreement ("yes", "sure", "sounds good", "I'd like to", "let's do it", "I'm in", "sign me up")
- Discussing specific times or days ("weekend works", "Tuesday evening", "mornings are best", "I can do 6pm")
- Asking about scheduling ("what times?", "when do classes start?", "what days are available?", "when's the next class?")
- Expressing time preferences ("I'd prefer evening", "weekend morning would work", "I'm free Tuesday")
- Providing availability information ("I'm available weekdays", "mornings work for me")
- Asking logistical questions about attending ("where's it located?", "what should I bring?", "should I wear anything specific?", "do I need to arrive early?")
- Responding positively to booking offers ("that works", "sounds perfect", "let's try it")
- Any indication they're planning to attend or moving toward booking
- Discussing with sales rep about scheduling ("let me check my calendar", "what works for you?")

🚨 CRITICAL RULE: If the prospect is discussing WHEN, WHERE, or HOW to attend → they have AGREED!
Don't wait for magic words like "yes, book me now". In real sales, talking logistics = commitment.

EXAMPLES THAT ARE AGREEMENT:
✅ "Tuesday works for me" → AGREED (discussing when)
✅ "I can do mornings" → AGREED (stating availability)
✅ "What time is the next class?" → AGREED (asking about scheduling)
✅ "Should I bring anything?" → AGREED (logistics question)
✅ "Where's it located?" → AGREED (planning to attend)
✅ "That sounds good" after booking offer → AGREED (positive response)

SIGNS OF DECLINE (mark as "not_interested"):
- Explicit rejection ("no thanks", "not interested", "I'll pass", "not for me", "maybe later")
- Clear backing out after initial interest
- Strong hesitation with no forward movement ("I need to think about it", "let me get back to you")
- Saying they're just browsing/looking

OTHERWISE (mark as "continue"):
- Still asking questions about the gym/classes (not booking-related)
- Hasn't engaged with booking yet
- Needs more information before deciding
- General conversation without commitment signals
- Sales bot mentioned follow-up/callback but prospect hasn't explicitly declined

⚠️ IMPORTANT: Don't be too conservative! In real sales, discussing scheduling = commitment.
If they're talking about WHEN/WHERE/HOW to attend, mark as "agreed_to_free_class" immediately.
Don't require explicit "yes, I want to book" - that's unrealistic!

CRITICAL: Set "should_end" based on outcome:
- If outcome is "agreed_to_free_class" → should_end = TRUE
- If outcome is "not_interested" → should_end = TRUE
- If outcome is "continue" → should_end = FALSE

Return ONLY valid JSON in this exact format:
{{
  "should_end": true or false,
  "outcome": "agreed_to_free_class" or "not_interested" or "continue",
  "reasoning": "brief explanation of your decision"
}}"""


    try:
        # Create assessment messages
        assessment_messages = [
            {"role": "system", "content": "You are a conversation analyzer. Return only valid JSON."},
            {"role": "user", "content": assessment_prompt}
        ]
        
        # Call LLM for assessment (use lower temperature for consistency)
        response = client.chat.completions.create(
            model=config.sales_model,
            messages=assessment_messages,
            max_tokens=150,
            temperature=0.1,  # Very low for consistent assessment\
            response_model= ConversationOutcome
        )
        
        should_end = response.should_end
        outcome = response.outcome.value
        
        # response_text = response.choices[0].message.content.strip()
        
        # # Parse JSON response
        # # Handle potential markdown code blocks
        # if "```json" in response_text:
        #     response_text = response_text.split("```json")[1].split("```")[0].strip()
        # elif "```" in response_text:
        #     response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # assessment = json.loads(response_text)
        
        # should_end = assessment.get("should_end", False)
        # outcome_str = assessment.get("outcome", "continue")
        
        # # Map to ConversationOutcome
        # outcome = None
        # if outcome_str == "agreed_to_free_class":
        #     outcome = ConversationOutcome.AGREED_FREE_CLASS
        # elif outcome_str == "not_interested":
        #     outcome = ConversationOutcome.NOT_INTERESTED
        
        return should_end, outcome
        
    except Exception as e:
        print(f"Error in conversation assessment: {e}")
        # Default to continue if assessment fails
        return False, None


