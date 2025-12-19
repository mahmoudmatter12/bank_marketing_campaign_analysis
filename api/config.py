"""
Configuration settings for the Flask application
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Dataset path (CSV file is in the api directory)
DATASET_PATH = Path(__file__).parent / 'bank-additional-full.csv'

# Flask configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 5000))

# CORS configuration
CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

# Data processing settings
TARGET_COLUMN = 'y'
LOG_DURATION_THRESHOLD = 5  # For conditional probability calculations

