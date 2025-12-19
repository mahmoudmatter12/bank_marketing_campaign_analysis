"""
Error Handler - Centralized error handling utilities
"""
from flask import jsonify
import traceback


def handle_error(error: Exception, status_code: int = 500) -> tuple:
    """Handle errors and return JSON response"""
    error_message = str(error)
    error_type = type(error).__name__
    
    # Determine status code based on error type
    if isinstance(error, ValueError):
        status_code = 400
    elif isinstance(error, FileNotFoundError):
        status_code = 404
    elif isinstance(error, KeyError):
        status_code = 400
    
    response = {
        'error': True,
        'message': error_message,
        'type': error_type,
        'status_code': status_code
    }
    
    # In development, include traceback
    import os
    if os.getenv('FLASK_ENV') == 'development':
        response['traceback'] = traceback.format_exc()
    
    return jsonify(response), status_code

