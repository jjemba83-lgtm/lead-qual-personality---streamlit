# 🥊 Gym Sales Bot Tester - Human Testing Interface

A Streamlit application for testing the gym lead qualification chatbot with real humans before production deployment.

## 📊 Current Status
- **AI Testing Accuracy**: 82%
- **Purpose**: Human validation before production
- **Ready for**: Real user testing

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Your OpenAI API Key

**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

**Option B: Enter in App**
- Launch the app and enter your API key in the sidebar

### 3. Run the App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 File Structure

Make sure these files are in the same directory:
```
project/
├── app.py              # Main Streamlit application
├── simulator.py        # Core chatbot logic (your existing file)
├── models.py          # Pydantic models (your existing file)
└── requirements.txt   # Python dependencies
```

## 🎯 Features

### 1. **Interactive Chat Interface**
- Clean, simple UI for natural conversation
- Real-time message display
- Chat-like experience similar to popular messaging apps

### 2. **Intent Detection**
- Automatic detection when conversation ends
- Shows:
  - Detected fitness intent
  - Confidence level
  - Reasoning behind detection
  - Recommended visit time

### 3. **Download Options**
- **JSON**: Complete conversation log with metadata
- **PDF**: Formatted transcript with intent results
- Both include timestamps and all conversation details

### 4. **Smart Conversation Management**
- Auto-detects when prospect agrees to free class
- Auto-detects when prospect is not interested
- Enforces message limits (default: 10 exchanges)
- Tracks token usage

## 💡 Usage Guide

### Starting a Conversation
1. Enter your OpenAI API key (if not set via environment)
2. Click "🚀 Start Conversation"
3. The sales bot will send its opening message

### During the Chat
- Type naturally as if you're a real gym prospect
- Test different personas:
  - Someone genuinely interested in fitness
  - Someone just looking for a free class
  - Someone with objections (price, time, intimidation)
  - Different fitness goals (weight loss, stress relief, etc.)

### Conversation End
The conversation automatically ends when:
- You agree to book a free class ✅
- You clearly indicate you're not interested ❌
- The message limit is reached (10 exchanges)

### After Conversation
1. Review the intent detection results
2. Check if the bot correctly identified your goal
3. Download the conversation log for analysis
4. Start a new conversation to test different scenarios

## 🧪 Testing Scenarios

### Recommended Test Cases:

1. **Hot Lead - Weight Loss**
   - Express clear interest in losing weight
   - Show enthusiasm
   - Agree to book quickly

2. **Warm Lead - Stress Relief**
   - Mention work stress
   - Ask questions about the gym
   - Show moderate interest

3. **Cold Lead - Just Looking**
   - Be vague about commitment
   - Only interested in the free class
   - Deflect long-term questions

4. **Objection Handling - Price**
   - Express interest but concern about cost
   - Ask about pricing
   - Test how bot handles objections

5. **Objection Handling - Intimidation**
   - Mention being nervous about boxing
   - Ask if it's beginner-friendly
   - Express concerns about fitness level

6. **Social Seeker**
   - Focus on community aspect
   - Ask about class sizes and atmosphere
   - Less focus on workout itself

## 📊 What to Look For

### Good Bot Performance:
- ✅ Correctly identifies your fitness goal
- ✅ Responds naturally to your messages
- ✅ Handles objections smoothly
- ✅ Doesn't push too hard if not interested
- ✅ Asks relevant follow-up questions

### Potential Issues:
- ❌ Misidentifies your intent
- ❌ Repetitive responses
- ❌ Doesn't pick up on objections
- ❌ Too pushy or not persuasive enough
- ❌ Awkward conversation flow

## 🔧 Configuration

### Adjust Settings in `app.py`:

```python
SimulationConfig(
    sales_model="gpt-4o-mini",        # LLM model
    sales_temperature=0.3,             # Response creativity (0-1)
    sales_max_tokens=120,              # Max response length
    max_message_exchanges=10           # Message limit
)
```

### Change Model:
- `gpt-4o-mini`: Fast, cost-effective (current)
- `gpt-4o`: More capable, higher quality
- `gpt-4-turbo`: Balance of speed and quality

## 📈 Analyzing Results

### JSON Download Contains:
- Complete message history
- Intent detection results
- Conversation outcome
- Token usage
- Timestamps

### PDF Download Contains:
- Formatted transcript
- Intent analysis
- Visual separation of messages
- Professional layout for sharing

## 🐛 Troubleshooting

### "Please enter your OpenAI API Key"
- Add API key in sidebar OR set `OPENAI_API_KEY` environment variable

### "Error getting response"
- Check API key is valid
- Ensure internet connection
- Verify OpenAI API status

### PDF Export Not Working
- Install reportlab: `pip install reportlab`
- JSON export always works as fallback

### Conversation Not Ending
- Manually trigger end by reaching message limit
- Be more explicit about agreement/disagreement

## 📝 Feedback Collection

When testing, document:
1. **Intent Detection Accuracy**: Did it identify your goal correctly?
2. **Conversation Quality**: Natural vs. robotic?
3. **Objection Handling**: Smooth vs. awkward?
4. **Outcome Accuracy**: Correct agreement/rejection detection?
5. **Overall Experience**: Would a real prospect engage?

## 🎓 Next Steps

1. **Gather Data**: Run 20-30 test conversations
2. **Analyze Patterns**: Where does the bot excel/struggle?
3. **Compare to AI Testing**: Human vs. AI prospect differences?
4. **Iterate**: Adjust prompts based on findings
5. **Production Ready**: Deploy when confidence is high

## 💻 Technical Notes

- **Session State**: Conversation persists during session
- **Token Tracking**: Monitor API usage costs
- **Error Handling**: Graceful failures with user feedback
- **Responsive Design**: Works on desktop and mobile

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Verify all files are present
3. Ensure dependencies are installed
4. Check OpenAI API key and credits

---

**Built for testing gym lead qualification accuracy with real human prospects** 🥊
