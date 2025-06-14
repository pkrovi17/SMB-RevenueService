# SMB Revenue Cloud - Financial Analysis Dashboard

A Flask-based web application for analyzing financial data from small-to-medium businesses using AI-powered insights and forecasting.

## Features

-  **Financial Data Analysis**: Upload CSV, Excel files or Google Sheets
-  **AI-Powered Insights**: Uses LLaMA models for data extraction and dashboard generation
-  **Interactive Charts**: Plotly-based visualizations for revenue, profit, and cost analysis
-  **Forecasting**: Prophet-based time series forecasting
-  **Security**: CSRF protection, rate limiting, file validation
-  **Production Ready**: Health checks, monitoring, and deployment guides

## Security Features

-  CSRF token protection
-  Rate limiting (10 requests/minute per IP)
-  File type validation
-  File size limits (16MB max)
-  Input sanitization
-  Session management with automatic cleanup
-  Environment-based configuration

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

## Direct Deployment (No Nginx)

### AWS EC2 Setup

1. **Launch EC2 Instance**
   - Instance Type: `t3.medium` or `t3.large`
   - OS: Ubuntu 22.04 LTS
   - Security Group: Allow SSH (22) and Custom TCP (5000)

2. **Connect and Install Dependencies**
   ```bash
   # Connect to your EC2 instance
   ssh -i your-key.pem ubuntu@your-ec2-ip

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Python and dependencies
   sudo apt install python3 python3-pip python3-venv git curl -y

   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull llama3
   ```

3. **Deploy Application**
   ```bash
   # Clone your repository
   git clone <your-repo-url>
   cd SMB-Revenue-Cloud

   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Set up environment variables
   cp .env.example .env
   nano .env
   ```

4. **Configure Environment**
   ```bash
   # Edit .env file
   SECRET_KEY=your-super-secret-key-for-aws
   FLASK_DEBUG=False
   FLASK_ENV=production
   RATE_LIMIT=10
   RATE_LIMIT_WINDOW=60
   MAX_CONTENT_LENGTH=16777216
   OLLAMA_TIMEOUT=90
   OLLAMA_MAX_RETRIES=3
   SESSION_TIMEOUT_HOURS=24
   ```

5. **Run with systemd Service**
   ```bash
   # Create systemd service
   sudo nano /etc/systemd/system/smb-revenue.service
   ```

   Add this content:
   ```ini
   [Unit]
   Description=SMB Revenue Cloud
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/SMB-Revenue-Cloud
   Environment=PATH=/home/ubuntu/SMB-Revenue-Cloud/venv/bin
   ExecStart=/home/ubuntu/SMB-Revenue-Cloud/venv/bin/python wsgi.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   # Enable and start the service
   sudo systemctl daemon-reload
   sudo systemctl enable smb-revenue
   sudo systemctl start smb-revenue

   # Check status
   sudo systemctl status smb-revenue
   ```

6. **Access Your Application**
   - Direct URL: `http://your-ec2-public-ip:5000`
   - Health check: `http://your-ec2-public-ip:5000/health`

7. **Monitoring and Maintenance**
   ```bash
   # Check application logs
   sudo journalctl -u smb-revenue -f

   # Monitor system resources
   htop
   df -h
   free -h

   # Health check
   curl http://localhost:5000/health
   ```

8. **Security Setup**
   ```bash
   # Set up firewall (UFW)
   sudo ufw enable
   sudo ufw allow ssh
   sudo ufw allow 5000

   # Update regularly
   sudo apt update && sudo apt upgrade -y
   ```

### Alternative: Screen/Tmux Method

If you prefer not to use systemd:

```bash
# Install screen
sudo apt install screen -y

# Start application in screen session
screen -S smb-app
python wsgi.py

# Detach from screen (Ctrl+A, then D)
# Reattach later with: screen -r smb-app
```

### Troubleshooting Direct Deployment

```bash
# Check if Ollama is running
ollama list

# Restart Ollama if needed
sudo systemctl restart ollama

# Check application status
sudo systemctl status smb-revenue

# View recent logs
sudo journalctl -u smb-revenue --since "1 hour ago"

# Check if port 5000 is listening
sudo netstat -tlnp | grep :5000
```

### Python Version Compatibility Issues

If you encounter numpy installation errors on Python 3.12:

```bash
# Check Python version
python3 --version

# If using Python 3.12, you may need to upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install numpy separately first
pip install numpy==1.26.2

# Then install the rest of the requirements
pip install -r requirements.txt
```

Alternative approach for Python 3.12:
```bash
# Use conda instead of pip (if available)
conda install numpy pandas plotly prophet flask werkzeug openpyxl python-dotenv

# Or install system packages first
sudo apt install python3-numpy python3-pandas python3-scipy
pip install -r requirements.txt --no-deps
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

### Python Version Compatibility Issues

If you encounter numpy installation errors on Python 3.12:

```bash
# Check Python version
python3 --version

# If using Python 3.12, you may need to upgrade pip and setuptools first
pip install --upgrade pip setuptools wheel

# Install numpy separately first
pip install numpy==1.26.2

# Then install the rest of the requirements
pip install -r requirements.txt
```

Alternative approach for Python 3.12:
```bash
# Use conda instead of pip (if available)
conda install numpy pandas plotly prophet flask werkzeug openpyxl python-dotenv

# Or install system packages first
sudo apt install python3-numpy python3-pandas python3-scipy
pip install -r requirements.txt --no-deps
```

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