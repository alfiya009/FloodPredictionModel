from flask import Flask
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing Flask app
from app import app

# This is what Vercel will use