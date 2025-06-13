# SMB Revenue Cloud - Financial Analysis Dashboard

A Flask-based web application for analyzing financial data from small-to-medium businesses using AI-powered insights and forecasting.

## Features

- 📊 **Financial Data Analysis**: Upload CSV, Excel files or Google Sheets
- 🤖 **AI-Powered Insights**: Uses LLaMA models for data extraction and dashboard generation
- 📈 **Interactive Charts**: Plotly-based visualizations for revenue, profit, and cost analysis
- 🔮 **Forecasting**: Prophet-based time series forecasting
- 🔒 **Security**: CSRF protection, rate limiting, file validation
- 🚀 **Production Ready**: Health checks, monitoring, and deployment guides

## Security Features

- ✅ CSRF token protection
- ✅ Rate limiting (10 requests/minute per IP)
- ✅ File type validation
- ✅ File size limits (16MB max)
- ✅ Input sanitization
- ✅ Session management with automatic cleanup
- ✅ Environment-based configuration

## Prerequisites

- Python 3.9+
- Ollama with LLaMA models installed

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SMB-Revenue-Cloud
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install and start Ollama**
   ```bash
   # Install Ollama (https://ollama.ai)
   ollama pull llama3
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask secret key | `dev-secret-key-change-in-production` |
| `FLASK_DEBUG` | Debug mode | `False` |
| `RATE_LIMIT` | Requests per minute | `10` |
| `RATE_LIMIT_WINDOW` | Rate limit window (seconds) | `60` |
| `MAX_CONTENT_LENGTH` | Max file size (bytes) | `16777216` (16MB) |
| `OLLAMA_TIMEOUT` | Ollama request timeout | `90` |
| `OLLAMA_MAX_RETRIES` | Max retry attempts | `3` |
| `SESSION_TIMEOUT_HOURS` | Session timeout | `24` |

### Production Deployment

1. **Set secure environment variables**
   ```bash
   export SECRET_KEY="your-super-secure-secret-key"
   export FLASK_DEBUG=False
   export FLASK_ENV=production
   ```

2. **Run with Flask's built-in server**
   ```bash
   python wsgi.py
   ```

3. **Set up reverse proxy (Nginx)**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

4. **Use systemd for service management**
   ```ini
   # /etc/systemd/system/smb-revenue.service
   [Unit]
   Description=SMB Revenue Cloud
   After=network.target

   [Service]
   User=www-data
   WorkingDirectory=/path/to/SMB-Revenue-Cloud
   Environment=PATH=/path/to/venv/bin
   ExecStart=/path/to/venv/bin/python wsgi.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

## API Endpoints

- `GET /` - Main application page
- `POST /upload` - Upload financial data
- `GET /dashboard/<session_id>` - View analysis dashboard
- `GET /api/data/<session_id>` - Get session data (JSON)
- `GET /health` - Health check endpoint

## Security Considerations

### File Upload Security
- Only CSV, XLSX, and XLS files allowed
- Maximum file size: 16MB
- Files are processed in memory and immediately deleted
- No virus scanning (consider adding for production)

### Rate Limiting
- 10 requests per minute per IP address
- Configurable via environment variables
- Returns HTTP 429 when exceeded

### Session Management
- Sessions expire after 24 hours
- Automatic cleanup of expired sessions
- Thread-safe session storage

### CSRF Protection
- All POST requests require valid CSRF token
- Tokens are automatically generated and validated

## Monitoring and Health Checks

### Health Check Endpoint
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "active_sessions": 5,
  "cleanup_count": 2,
  "timestamp": "2024-01-15T10:30:00"
}
```

### Logging
- Application logs to stdout/stderr
- Structured logging with different levels
- Error tracking for debugging

## Troubleshooting

### Common Issues

1. **Ollama not found**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3
   ```

2. **Permission denied on uploads**
   ```bash
   # Fix directory permissions
   chmod 755 uploads/
   ```

3. **Memory issues with large files**
   - Reduce `MAX_CONTENT_LENGTH` in environment
   - Monitor system memory usage

4. **Rate limiting issues**
   - Increase `RATE_LIMIT` in environment
   - Check for multiple requests from same IP

### Performance Optimization

1. **Use Redis for session storage**
   ```python
   # Replace in-memory storage with Redis
   from flask_session import Session
   app.config['SESSION_TYPE'] = 'redis'
   Session(app)
   ```

2. **Add caching layer**
   ```python
   from flask_caching import Cache
   cache = Cache(app, config={'CACHE_TYPE': 'redis'})
   ```

3. **Enable threading for concurrent requests**
   ```python
   app.run(host='0.0.0.0', port=5000, threaded=True)
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the logs for error details