#!/usr/bin/env python3
"""
Installation verification script for Gym Sales Bot Tester.
Run this to check if everything is set up correctly.
"""

import sys
import os

def print_header(text):
    print("\n" + "="*50)
    print(f"  {text}")
    print("="*50)

def check_python_version():
    print("\n1️⃣ Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def check_dependencies():
    print("\n2️⃣ Checking dependencies...")
    required = {
        'streamlit': 'streamlit',
        'openai': 'openai',
        'pydantic': 'pydantic',
        'reportlab': 'reportlab (optional)',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, display_name in required.items():
        try:
            __import__(module)
            print(f"   ✅ {display_name} - installed")
        except ImportError:
            if module == 'reportlab':
                print(f"   ⚠️  {display_name} - not installed (PDF export won't work)")
            else:
                print(f"   ❌ {display_name} - NOT installed")
                all_ok = False
    
    return all_ok

def check_files():
    print("\n3️⃣ Checking required files...")
    required_files = [
        'app.py',
        'models.py',
        'simulator.py',
        'requirements.txt'
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✅ {file} - found")
        else:
            print(f"   ❌ {file} - MISSING")
            all_ok = False
    
    return all_ok

def check_api_key():
    print("\n4️⃣ Checking OpenAI API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    
    if api_key:
        masked_key = api_key[:7] + "..." + api_key[-4:]
        print(f"   ✅ API key found: {masked_key}")
        return True
    else:
        print("   ⚠️  No API key in environment")
        print("      You can enter it in the app sidebar")
        return True  # Not critical, can enter in app

def main():
    print_header("🥊 Gym Sales Bot Tester - Installation Check")
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_files(),
        check_api_key()
    ]
    
    print("\n" + "="*50)
    if all(checks[:3]):  # API key is optional
        print("✅ All checks passed!")
        print("\nYou're ready to run the app:")
        print("   streamlit run app.py")
    else:
        print("❌ Some checks failed!")
        print("\nTo fix:")
        print("   pip install -r requirements.txt")
        print("\nMake sure simulator.py and models.py are in the same folder")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
