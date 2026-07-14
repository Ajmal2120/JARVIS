"""
JARVIS Background Launcher - Runs without CMD window
Double-click this file to start JARVIS in background.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run the embedded launcher
if __name__ == "__main__":
    import jarvis_embedded
    jarvis_embedded.main()
