#!/usr/bin/env python3
"""
WSGI configuration for production deployment
"""
import os
from dotenv import load_dotenv
from app import app
from werkzeug.serving import WSGIRequestHandler

# Load environment variables
load_dotenv()

# Suppress only GET /performance logs
class SilentPerformanceRequestHandler(WSGIRequestHandler):
    def log_request(self, code='-', size='-'):
        if self.path == '/performance' and self.command == 'GET':
            return
        super().log_request(code, size)

if __name__ == "__main__":
    # Production settings
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True,
        request_handler=SilentPerformanceRequestHandler
    ) 