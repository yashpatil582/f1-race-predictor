"""
Streamlit Cloud entry point.
"""
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
from dashboard.app import main

# Run the app
main()
