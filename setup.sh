#!/bin/bash

# Setup script for Gym Sales Bot Tester

echo "🥊 Setting up Gym Sales Bot Tester..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "✓ Python version: $python_version"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check for API key
echo ""
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  OpenAI API key not found in environment"
    echo "You can either:"
    echo "  1. Set it now: export OPENAI_API_KEY='your-key-here'"
    echo "  2. Enter it in the app sidebar when you launch"
else
    echo "✓ OpenAI API key found in environment"
fi

# Create a launch script
cat > run.sh << 'EOL'
#!/bin/bash
streamlit run app.py
EOL

chmod +x run.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "To launch the app, run:"
echo "  ./run.sh"
echo "  or"
echo "  streamlit run app.py"
echo ""
echo "The app will open in your browser at http://localhost:8501"
