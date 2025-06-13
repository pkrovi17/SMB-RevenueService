#!/usr/bin/env python3
"""
WSGI configuration for production deployment
"""
import os
from dotenv import load_dotenv
from app import app

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Production settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    ) 