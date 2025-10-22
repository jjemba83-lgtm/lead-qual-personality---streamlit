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

from models import (
    ProspectProfile, BigFiveTraits, Intent, ObjectionType, ReadinessLevel,
    ConversationMessage, IntentDetection, ConversationLog, ConversationOutcome,
    SimulationConfig
)


# Initialize OpenAI client
client = None


def initialize_client(api_key: str):
    """Initialize the OpenAI client with API key."""
    global client
    client = OpenAI(api_key=api_key)


def generate_prospect_profile() -> ProspectProfile:
    """
    Randomly generates a complete prospect profile with Big Five traits,
    intent, objection, readiness level, and demographics.
    """
    big_five = BigFiveTraits(
        openness=random.randint(1, 10),
        conscientiousness=random.randint(1, 10),
        extraversion=random.randint(1, 10),
        agreeableness=random.randint(1, 10),
        neuroticism=random.randint(1, 10)
    )
    
    true_intent = random.choice(list(Intent))
    objection_type = random.choice([None] + list(ObjectionType))
    readiness_level = random.choice(list(ReadinessLevel))
    
    age_ranges = ["18-25", "25-35", "35-45", "45-55", "55+"]
    fitness_backgrounds = ["beginner", "intermediate", "advanced", "couch_to_5k", "former_athlete"]
    
    return ProspectProfile(
        big_five=big_five,
        true_intent=true_intent,
        objection_type=objection_type,
        readiness_level=readiness_level,
        age_range=random.choice(age_ranges),
        fitness_background=random.choice(fitness_backgrounds)
    )


def create_prospect_prompt(profile: ProspectProfile) -> str:
    """
    Converts a ProspectProfile into a system prompt for the Prospect LLM.
    Instructs the LLM to embody the personality and intent naturally.
    """
    # Helper function to get trait level and description
    def get_trait_description(trait_name: str, score: int) -> tuple:
        """Returns (level, brief description) for a given trait and score."""
        trait_descriptions = {
            "openness": {
                "low": "Prefer routines, skeptical of trends",
                "medium": "Open to reasonable new things",
                "high": "Eager to try new approaches"
            },
            "conscientiousness": {
                "low": "Spontaneous, go with flow",
                "medium": "Somewhat organized when motivated",
                "high": "Disciplined, goal-oriented"
            },
            "extraversion": {
                "low": "Introverted, prefer small groups",
                "medium": "Moderately social",
                "high": "Outgoing, love group activities"
            },
            "agreeableness": {
                "low": "Skeptical, challenge claims",
                "medium": "Polite but questioning",
                "high": "Friendly, trusting, cooperative"
            },
            "neuroticism": {
                "low": "Calm, don't worry much",
                "medium": "Some anxiety, manageable",
                "high": "Anxious, worry about risks"
            }
        }
        
        if score <= 3:
            level = "LOW"
            desc = trait_descriptions[trait_name]["low"]
        elif score <= 7:
            level = "MEDIUM"
            desc = trait_descriptions[trait_name]["medium"]
        else:
            level = "HIGH"
            desc = trait_descriptions[trait_name]["high"]
        
        return level, desc
    
    # Get descriptions for each trait
    openness_level, openness_desc = get_trait_description("openness", profile.big_five.openness)
    consc_level, consc_desc = get_trait_description("conscientiousness", profile.big_five.conscientiousness)
    extra_level, extra_desc = get_trait_description("extraversion", profile.big_five.extraversion)
    agree_level, agree_desc = get_trait_description("agreeableness", profile.big_five.agreeableness)
    neuro_level, neuro_desc = get_trait_description("neuroticism", profile.big_five.neuroticism)
    
    # Get intent-specific behavioral cues
    behavioral_cues = {
        Intent.WEIGHT_LOSS: "Mention concerns about weight, clothes fitting, or wanting to 'slim down' or 'drop pounds'. Reference how you used to look or upcoming events.",
        Intent.STRESS_RELIEF: "Talk about feeling stressed, overwhelmed, or needing an outlet. Mention work pressure, anxiety, or needing to 'blow off steam'.",
        Intent.BOXING_TECHNIQUE: "Ask about proper form, technique training, or learning fundamentals. Show interest in the technical/skill aspects of boxing.",
        Intent.GENERAL_FITNESS: "Focus on overall health, staying active, or 'getting in shape'. Keep goals broad rather than specific.",
        Intent.SOCIAL_COMMUNITY: "Ask about class sizes, group dynamics, or making friends. Show interest in the PEOPLE and community more than just the workout.",
        Intent.JUST_FREE_CLASS: "Be vague about commitment. Deflect when asked about long-term goals. Say 'just curious' or 'wanted to try it once'. Show hesitation about ongoing membership."
    }
    
    intent_descriptions = {
        Intent.WEIGHT_LOSS: "You want to lose weight and get in better shape",
        Intent.STRESS_RELIEF: "You're looking for stress relief and mental health benefits through exercise",
        Intent.BOXING_TECHNIQUE: "You want to learn proper boxing technique and skills",
        Intent.GENERAL_FITNESS: "You want to improve your overall fitness level",
        Intent.SOCIAL_COMMUNITY: "You're looking for a social community and group fitness experience",
        Intent.JUST_FREE_CLASS: "You just want the free class and have no real intention of joining"
    }
    
    objection_descriptions = {
        ObjectionType.PRICE: "You're concerned about the cost",
        ObjectionType.TIME_COMMITMENT: "You're worried about having enough time",
        ObjectionType.INJURY_CONCERNS: "You're concerned about getting injured",
        ObjectionType.INTIMIDATION: "You feel intimidated by boxing or group fitness",
        ObjectionType.LOCATION_PARKING: "You have concerns about the location or parking",
        ObjectionType.JUST_LOOKING: "You're just browsing and not ready to commit"
    }
    
    readiness_descriptions = {
        ReadinessLevel.HOT: "You're very interested and ready to take action soon",
        ReadinessLevel.WARM: "You're interested but want to learn more before deciding",
        ReadinessLevel.COLD: "You're just exploring options and not in a hurry"
    }
    
    prompt = f"""You are a potential gym member who filled out a web form to learn more about a boxing fitness gym.

PERSONALITY PROFILE (Big Five Traits):
- Openness: {profile.big_five.openness}/10 ({openness_level}) - {openness_desc}
- Conscientiousness: {profile.big_five.conscientiousness}/10 ({consc_level}) - {consc_desc}
- Extraversion: {profile.big_five.extraversion}/10 ({extra_level}) - {extra_desc}
- Agreeableness: {profile.big_five.agreeableness}/10 ({agree_level}) - {agree_desc}
- Neuroticism: {profile.big_five.neuroticism}/10 ({neuro_level}) - {neuro_desc}

YOUR TRUE INTENT: {intent_descriptions[profile.true_intent]}

HOW TO REVEAL YOUR INTENT NATURALLY:
{behavioral_cues[profile.true_intent]}

YOUR READINESS: {readiness_descriptions[profile.readiness_level]}

DEMOGRAPHICS:
- Age range: {profile.age_range}
- Fitness background: {profile.fitness_background}

{f"YOUR CONCERN: {objection_descriptions[profile.objection_type]}" if profile.objection_type else "You have no major objections."}

INSTRUCTIONS:
- You're texting/chatting - keep responses to 1-2 sentences maximum
- Respond naturally based on your personality traits
- Let your intent emerge through conversation using the behavioral cues above - don't explicitly say your intent
- Be realistic - show interest or skepticism based on your traits
- If you have concerns, let them surface naturally
- DO NOT mention your personality scores or state your intent directly"""
    
    return prompt


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
- You have a MAXIMUM of 3 message exchanges (6 total messages)
- Message 1: Use the standardized opening below
- Message 2: Address their response, handle objections, and OFFER THE FREE CLASS
- Message 3: Final attempt - if they haven't agreed, tell them a sales associate will call within 24 hours

STANDARDIZED OPENING (Use this for your FIRST message):
"Hi! Thanks for reaching out about our boxing fitness gym. To help match you with the right class, I have a few quick questions:

1. What's your main fitness goal? (weight loss, stress relief, learn technique, general fitness, etc.)
2. How often do you currently exercise?
3. Any concerns about high-intensity training?

Looking forward to getting you started!"

RULES:
- Keep responses brief (2-3 sentences max)
- Be direct and ask for the free class booking by message 2
- If they explicitly say not interested, end politely
- If they agree to free class, ask preferred time (morning/evening/weekend) then end
- You have their phone and email from the web form

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
⚠️ NEVER include the INTENT_DETECTION JSON in your regular chat messages to the prospect!
⚠️ The INTENT_DETECTION should ONLY be provided when you are explicitly asked: "provide your INTENT_DETECTION assessment"
⚠️ During normal conversation, respond naturally without ANY JSON formatting
⚠️ Keep your responses conversational and friendly - save the structured data for when specifically requested

When explicitly asked for intent detection, provide assessment in EXACT format:

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
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        
        return content, tokens
    
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return f"[Error: {str(e)}]", 0


def extract_intent_detection(sales_final_message: str) -> Optional[IntentDetection]:
    """
    Parses the Sales LLM's final message to extract structured IntentDetection.
    Handles both clean JSON and markdown-wrapped JSON.
    """
    try:
        # Look for JSON block in the message
        if "INTENT_DETECTION:" in sales_final_message:
            json_start = sales_final_message.find("{")
            json_end = sales_final_message.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = sales_final_message[json_start:json_end]
                data = json.loads(json_str)
                
                # Handle case where LLM returns multiple intents separated by comma
                detected_intent_str = data["detected_intent"].strip()
                
                # If multiple intents, take the first one
                if "," in detected_intent_str:
                    detected_intent_str = detected_intent_str.split(",")[0].strip()
                    print(f"Warning: Multiple intents detected, using first: {detected_intent_str}")
                
                return IntentDetection(
                    detected_intent=Intent(detected_intent_str),
                    confidence_level=float(data["confidence_level"]),
                    reasoning=data["reasoning"],
                    best_time_to_visit=data.get("best_time_to_visit")
                )
    except Exception as e:
        print(f"Error extracting intent detection: {e}")
    
    return None


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
- Explicit rejection ("no thanks", "not interested", "I'll pass", "not for me")
- Clear backing out after initial interest
- Strong hesitation with no forward movement

OTHERWISE (mark as "continue"):
- Still asking questions about the gym/classes (not booking-related)
- Hasn't engaged with booking yet
- Needs more information before deciding
- General conversation without commitment signals

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
            temperature=0.1  # Very low for consistent assessment
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        assessment = json.loads(response_text)
        
        should_end = assessment.get("should_end", False)
        outcome_str = assessment.get("outcome", "continue")
        
        # Map to ConversationOutcome
        outcome = None
        if outcome_str == "agreed_to_free_class":
            outcome = ConversationOutcome.AGREED_FREE_CLASS
        elif outcome_str == "not_interested":
            outcome = ConversationOutcome.NOT_INTERESTED
        
        return should_end, outcome
        
    except Exception as e:
        print(f"Error in conversation assessment: {e}")
        # Default to continue if assessment fails
        return False, None


def run_single_conversation(
    config: SimulationConfig,
    conversation_id: str
) -> ConversationLog:
    """
    Orchestrates one complete conversation between Prospect and Sales LLMs.
    Alternates between LLMs, enforces rules, and returns complete log.
    """
    # Generate prospect profile
    profile = generate_prospect_profile()
    
    # Initialize conversation
    messages: List[ConversationMessage] = []
    total_tokens = 0
    
    # Create system prompts
    prospect_system = create_prospect_prompt(profile)
    sales_system = create_sales_prompt()
    
    # Prospect conversation history
    prospect_history = [{"role": "system", "content": prospect_system}]
    
    # Sales conversation history
    sales_history = [{"role": "system", "content": sales_system}]
    
    # Sales LLM initiates with standardized opening
    sales_opening = """Hi! Thanks for reaching out about our boxing fitness gym. I have a few questions to help us learn more about you:

1. What's your main fitness goal? (weight loss, stress relief, learn technique, general fitness, etc.)
2. How often do you currently exercise?
3. Any concerns about high-intensity training?

Looking forward to getting you started!"""
    
    sales_msg = ConversationMessage(
        role="sales",
        content=sales_opening,
        tokens_used=0
    )
    messages.append(sales_msg)
    
    # Add to histories
    sales_history.append({"role": "assistant", "content": sales_opening})
    prospect_history.append({"role": "user", "content": sales_opening})
    
    # Conversation loop - SALES FIRST, PROSPECT LAST
    outcome = None
    intent_detection = None
    
    for exchange in range(config.max_message_exchanges):
        # Sales responds (or uses opening for first exchange)
        if exchange == 0:
            # First exchange already has sales opening
            pass
        else:
            sales_response, sales_tokens = call_llm(sales_history, config, "sales")
            total_tokens += sales_tokens
            
            sales_msg = ConversationMessage(
                role="sales",
                content=sales_response,
                tokens_used=sales_tokens
            )
            messages.append(sales_msg)
            
            # Update histories
            sales_history.append({"role": "assistant", "content": sales_response})
            prospect_history.append({"role": "user", "content": sales_response})
        
        # Prospect responds (always after sales)
        prospect_response, prospect_tokens = call_llm(prospect_history, config, "prospect")
        total_tokens += prospect_tokens
        
        prospect_msg = ConversationMessage(
            role="prospect",
            content=prospect_response,
            tokens_used=prospect_tokens
        )
        messages.append(prospect_msg)
        
        # Update histories
        prospect_history.append({"role": "assistant", "content": prospect_response})
        sales_history.append({"role": "user", "content": prospect_response})
        
        # Use LLM to assess if conversation should end after prospect response
        should_end, assessed_outcome = assess_conversation_status(
            sales_history,  # Full conversation context
            prospect_response,
            config
        )
        
        if should_end and assessed_outcome:
            outcome = assessed_outcome
            break
        
        # Check if we've reached message limit
        if exchange >= config.max_message_exchanges - 1:
            # This is the last exchange - prospect has responded
            # Need to check one final time if they agreed
            should_end, assessed_outcome = assess_conversation_status(
                sales_history,
                prospect_response,
                config
            )
            if should_end and assessed_outcome:
                outcome = assessed_outcome
            else:
                outcome = ConversationOutcome.REACHED_MESSAGE_LIMIT
            break
    
    # If no outcome determined, it hit the limit
    if outcome is None:
        outcome = ConversationOutcome.REACHED_MESSAGE_LIMIT
    
    # Extract intent detection - need to get it from sales bot
    # Make a final call to sales bot to get intent assessment
    intent_request = sales_history + [{
        "role": "user",
        "content": "Based on our conversation, please provide your INTENT_DETECTION assessment in the required JSON format."
    }]
    intent_response, intent_tokens = call_llm(intent_request, config, "sales")
    total_tokens += intent_tokens
    intent_detection = extract_intent_detection(intent_response)
    
    # Determine if intent matches
    intent_match = False
    if intent_detection:
        intent_match = intent_detection.detected_intent == profile.true_intent
    
    # Create conversation log
    conversation_log = ConversationLog(
        conversation_id=conversation_id,
        prospect_profile=profile,
        messages=messages,
        intent_detection=intent_detection,
        outcome=outcome,
        total_tokens_used=total_tokens,
        conversation_length=len([m for m in messages if m.role == "sales"]),
        intent_match=intent_match
    )
    
    return conversation_log


def run_simulation_batch(config: SimulationConfig) -> List[ConversationLog]:
    """
    Executes N conversations with progress tracking.
    Returns list of all conversation logs.
    """
    logs: List[ConversationLog] = []
    
    print(f"\n🥊 Starting {config.num_simulations} simulated conversations...\n")
    
    for i in tqdm(range(config.num_simulations), desc="Running simulations"):
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:04d}"
        
        try:
            log = run_single_conversation(config, conversation_id)
            logs.append(log)
        except Exception as e:
            print(f"\nError in conversation {conversation_id}: {e}")
            continue
    
    print(f"\n✅ Completed {len(logs)}/{config.num_simulations} conversations\n")
    
    return logs


def save_logs(logs: List[ConversationLog], output_dir: str = "/home/claude"):
    """
    Saves conversation logs to JSON file for transcript preservation.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_transcripts_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert to JSON-serializable format
    logs_data = [log.model_dump(mode='json') for log in logs]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(logs_data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"💾 Transcripts saved to: {filepath}")
    
    return filepath
