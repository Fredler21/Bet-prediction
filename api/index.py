import sys
import os

# Ensure project root is on the path so `src` package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.web import app
