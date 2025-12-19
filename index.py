"""
Main Flask Application Entry Point
Works for both local development and Vercel deployment
"""
from api import create_app
from api.config import DEBUG, HOST, PORT


# Create the Flask app instance
app = create_app()

# For local development
if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG)

# Export app for Vercel (Vercel will automatically detect this)
__all__ = ['app']
