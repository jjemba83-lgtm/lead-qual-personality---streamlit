"""
Streamlit app for human testing of gym lead qualification chatbot.
Allows humans to chat with the sales bot and download conversation logs.
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env file

import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Optional
import os
from openai import OpenAI

from models import (
    ConversationMessage, IntentDetection, ConversationLog, 
    ConversationOutcome, Intent, SimulationConfig
)
from simulator import (
    create_sales_prompt, extract_intent_detection, call_llm,
    assess_conversation_status, initialize_client
)


# Page configuration
st.set_page_config(
    page_title="Gym Sales Bot Tester",
    page_icon="🥊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        max-width: 800px;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-container {
        height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .intent-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin-top: 1rem;
    }
    .stats-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    h1 {
        color: #1e3a8a;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sales_history' not in st.session_state:
        st.session_state.sales_history = []
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    if 'conversation_ended' not in st.session_state:
        st.session_state.conversation_ended = False
    if 'intent_detection' not in st.session_state:
        st.session_state.intent_detection = None
    if 'conversation_outcome' not in st.session_state:
        st.session_state.conversation_outcome = None
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'config' not in st.session_state:
        st.session_state.config = SimulationConfig(
            sales_model="gpt-4o-mini",
            sales_temperature=0.3,
            sales_max_tokens=120,
            max_message_exchanges=10
        )


def start_conversation():
    """Initialize conversation with sales bot opening message."""
    sales_system = create_sales_prompt()
    st.session_state.sales_history = [{"role": "system", "content": sales_system}]
    
    # Sales bot opening message
    sales_opening = """Hi! Thanks for reaching out about our boxing fitness gym. I have a few questions to help us learn more about you:

1. What's your main fitness goal? (weight loss, stress relief, learn technique, general fitness, etc.)
2. How often do you currently exercise?
3. Any concerns about high-intensity training?

Looking forward to getting you started!"""
    
    st.session_state.messages.append({
        "role": "sales",
        "content": sales_opening,
        "timestamp": datetime.now()
    })
    
    st.session_state.sales_history.append({"role": "assistant", "content": sales_opening})
    st.session_state.conversation_started = True


def send_message(user_input: str):
    """Process user message and get sales bot response."""
    if not user_input.strip():
        return
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    st.session_state.sales_history.append({"role": "user", "content": user_input})
    
    # Check if conversation should end
    should_end, outcome = assess_conversation_status(
        st.session_state.sales_history,
        user_input,
        st.session_state.config
    )
    
    if should_end and outcome:
        st.session_state.conversation_ended = True
        st.session_state.conversation_outcome = outcome
        detect_intent()
        return
    
    # Check message limit
    num_exchanges = len([m for m in st.session_state.messages if m["role"] == "user"])
    if num_exchanges >= st.session_state.config.max_message_exchanges:
        st.session_state.conversation_ended = True
        st.session_state.conversation_outcome = ConversationOutcome.REACHED_MESSAGE_LIMIT
        detect_intent()
        return
    
    # Get sales bot response
    try:
        sales_response, tokens = call_llm(
            st.session_state.sales_history,
            st.session_state.config,
            "sales"
        )
        st.session_state.total_tokens += tokens
        
        st.session_state.messages.append({
            "role": "sales",
            "content": sales_response,
            "timestamp": datetime.now()
        })
        st.session_state.sales_history.append({"role": "assistant", "content": sales_response})
        
    except Exception as e:
        st.error(f"Error getting response: {str(e)}")


def detect_intent():
    """Get intent detection from sales bot."""
    try:
        intent_request = st.session_state.sales_history + [{
            "role": "user",
            "content": "Based on our conversation, please provide your INTENT_DETECTION assessment in the required JSON format."
        }]
        
        intent_response, tokens = call_llm(
            intent_request,
            st.session_state.config,
            "sales"
        )
        st.session_state.total_tokens += tokens
        
        intent_detection = extract_intent_detection(intent_response)
        st.session_state.intent_detection = intent_detection
        
    except Exception as e:
        st.error(f"Error detecting intent: {str(e)}")


def create_download_json() -> str:
    """Create JSON export of conversation."""
    conversation_data = {
        "conversation_id": f"human_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "messages": st.session_state.messages,
        "intent_detection": st.session_state.intent_detection.model_dump() if st.session_state.intent_detection else None,
        "outcome": st.session_state.conversation_outcome.value if st.session_state.conversation_outcome else None,
        "total_tokens_used": st.session_state.total_tokens,
        "conversation_length": len([m for m in st.session_state.messages if m["role"] == "sales"])
    }
    return json.dumps(conversation_data, indent=2, default=str)


def create_download_pdf() -> bytes:
    """Create PDF export of conversation."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from io import BytesIO
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e3a8a'),
            spaceAfter=30
        )
        story.append(Paragraph("🥊 Gym Sales Bot Conversation", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Metadata
        metadata_style = ParagraphStyle('Metadata', parent=styles['Normal'], fontSize=10, textColor=colors.grey)
        story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", metadata_style))
        story.append(Paragraph(f"<b>Conversation ID:</b> human_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}", metadata_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Conversation
        story.append(Paragraph("<b>Conversation Transcript</b>", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        for msg in st.session_state.messages:
            role_name = "Sales Bot" if msg["role"] == "sales" else "You"
            role_style = ParagraphStyle(
                'Role',
                parent=styles['Normal'],
                fontSize=11,
                textColor=colors.HexColor('#1e3a8a') if msg["role"] == "sales" else colors.HexColor('#059669'),
                fontName='Helvetica-Bold'
            )
            content_style = ParagraphStyle('Content', parent=styles['Normal'], fontSize=10, leftIndent=20)
            
            story.append(Paragraph(f"{role_name}:", role_style))
            story.append(Paragraph(msg["content"], content_style))
            story.append(Spacer(1, 0.15*inch))
        
        # Intent Detection
        if st.session_state.intent_detection:
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph("<b>Intent Detection Results</b>", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            intent_data = [
                ["Detected Intent", st.session_state.intent_detection.detected_intent.value.replace('_', ' ').title()],
                ["Confidence", f"{st.session_state.intent_detection.confidence_level:.1%}"],
                ["Reasoning", st.session_state.intent_detection.reasoning]
            ]
            
            if st.session_state.intent_detection.best_time_to_visit:
                intent_data.append(["Best Time to Visit", st.session_state.intent_detection.best_time_to_visit])
            
            intent_table = Table(intent_data, colWidths=[2*inch, 4*inch])
            intent_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f8ff')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(intent_table)
        
        # Outcome
        if st.session_state.conversation_outcome:
            story.append(Spacer(1, 0.2*inch))
            outcome_style = ParagraphStyle('Outcome', parent=styles['Normal'], fontSize=11)
            outcome_text = st.session_state.conversation_outcome.value.replace('_', ' ').title()
            story.append(Paragraph(f"<b>Outcome:</b> {outcome_text}", outcome_style))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        st.error("PDF export requires reportlab. Install with: pip install reportlab")
        return None


def reset_conversation():
    """Reset all session state to start a new conversation."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()


def main():
    """Main Streamlit app."""
    initialize_session_state()
    
    # Header
    st.title("🥊 Gym Sales Bot Tester")
    st.markdown("Chat with the sales bot to test lead qualification before production deployment.")
    
    # API Key input in sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        
        if api_key:
            initialize_client(api_key)
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter your OpenAI API Key")
        
        st.divider()
        st.markdown(f"**Max Exchanges:** {st.session_state.config.max_message_exchanges}")
        st.markdown(f"**Model:** {st.session_state.config.sales_model}")
        
        if st.session_state.total_tokens > 0:
            st.divider()
            st.markdown(f"**Tokens Used:** {st.session_state.total_tokens:,}")
    
    # Main content
    if not st.session_state.conversation_started:
        st.info("👋 Click 'Start Conversation' to begin chatting with the sales bot.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Conversation", type="primary", disabled=not api_key):
                start_conversation()
                st.rerun()
    
    else:
        # Chat display
        st.markdown("### 💬 Chat")
        
        # Display messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="🥊" if msg["role"] == "sales" else "👤"):
                st.markdown(msg["content"])
        
        # Input area
        if not st.session_state.conversation_ended:
            user_input = st.chat_input("Type your message...", key="chat_input")
            if user_input:
                send_message(user_input)
                st.rerun()
        else:
            st.info("🎯 Conversation ended. View the results below.")
        
        # Intent Detection Results
        if st.session_state.conversation_ended and st.session_state.intent_detection:
            st.markdown("---")
            st.markdown("### 🎯 Intent Detection Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Detected Intent", 
                    st.session_state.intent_detection.detected_intent.value.replace('_', ' ').title()
                )
            with col2:
                st.metric(
                    "Confidence",
                    f"{st.session_state.intent_detection.confidence_level:.0%}"
                )
            with col3:
                if st.session_state.intent_detection.best_time_to_visit:
                    st.metric(
                        "Best Time",
                        st.session_state.intent_detection.best_time_to_visit.title()
                    )
            
            with st.expander("📝 Reasoning", expanded=True):
                st.markdown(st.session_state.intent_detection.reasoning)
            
            # Outcome
            if st.session_state.conversation_outcome:
                outcome_text = st.session_state.conversation_outcome.value.replace('_', ' ').title()
                outcome_emoji = "✅" if st.session_state.conversation_outcome == ConversationOutcome.AGREED_FREE_CLASS else "❌"
                st.markdown(f"**Outcome:** {outcome_emoji} {outcome_text}")
        
        # Download options
        if st.session_state.conversation_ended:
            st.markdown("---")
            st.markdown("### 📥 Download Conversation")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                json_data = create_download_json()
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                try:
                    pdf_data = create_download_pdf()
                    if pdf_data:
                        st.download_button(
                            label="📑 Download PDF",
                            data=pdf_data,
                            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                except Exception as e:
                    st.button("📑 Download PDF", disabled=True, use_container_width=True)
                    st.caption("Install reportlab for PDF export")
            
            with col3:
                if st.button("🔄 New Conversation", type="primary", use_container_width=True):
                    reset_conversation()
                    st.rerun()


if __name__ == "__main__":
    main()
