from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pandas as pd
import json
import os
import subprocess
import plotly.graph_objs as go
import plotly.utils
from prophet import Prophet
import numpy as np
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime, timedelta
import logging
from functools import wraps
import threading
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Security configurations
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
GOOGLE_SHEETS_PATTERN = r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)'

# Rate limiting
request_counts = {}
RATE_LIMIT = 10  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store session data in memory (for demo - use Redis/DB in production)
session_data = {}
session_lock = threading.Lock()

# Security and utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_google_sheets_url(url):
    """Validate Google Sheets URL format"""
    if not url or not isinstance(url, str):
        return False
    return bool(re.match(GOOGLE_SHEETS_PATTERN, url.strip()))

def rate_limit_exceeded(ip):
    """Check if rate limit is exceeded for an IP"""
    global request_counts
    current_time = time.time()
    
    # Clean old entries
    request_counts = {k: v for k, v in request_counts.items() 
                     if current_time - v['timestamp'] < RATE_LIMIT_WINDOW}
    
    if ip not in request_counts:
        request_counts[ip] = {'count': 1, 'timestamp': current_time}
        return False
    
    if current_time - request_counts[ip]['timestamp'] > RATE_LIMIT_WINDOW:
        request_counts[ip] = {'count': 1, 'timestamp': current_time}
        return False
    
    request_counts[ip]['count'] += 1
    return request_counts[ip]['count'] > RATE_LIMIT

def cleanup_old_sessions():
    """Remove sessions older than 24 hours"""
    current_time = datetime.now()
    expired_sessions = []
    
    with session_lock:
        for session_id, data in session_data.items():
            if (current_time - data['timestamp']) > timedelta(hours=24):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del session_data[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    return len(expired_sessions)

def generate_csrf_token():
    """Generate CSRF token"""
    if 'csrf_token' not in session:
        session['csrf_token'] = str(uuid.uuid4())
    return session['csrf_token']

def validate_csrf_token():
    """Validate CSRF token"""
    token = request.form.get('csrf_token')
    return token and token == session.get('csrf_token')

# Decorators
def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        ip = request.remote_addr
        if rate_limit_exceeded(ip):
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        return f(*args, **kwargs)
    return decorated_function

def require_csrf(f):
    """CSRF protection decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.method == 'POST':
            if not validate_csrf_token():
                logger.warning(f"CSRF token validation failed for IP: {request.remote_addr}")
                flash('Invalid request. Please try again.', 'error')
                return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Background cleanup task
def background_cleanup():
    """Background task to clean up old sessions"""
    while True:
        try:
            cleanup_old_sessions()
            time.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in background cleanup: {e}")
            time.sleep(3600)

# Start background cleanup thread
cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
cleanup_thread.start()

def run_llama_dashboard_with_retry(data_str, max_attempts=5):
    prompt = get_dashboard_prompt(data_str)
    last_error = ""
    charts = []

    for attempt in range(1, max_attempts + 1):
        print(f"🔁 Attempt {attempt} to get LLaMA dashboard config...")
        if attempt > 1:
            prompt = get_dashboard_prompt(data_str, error_message=last_error)

        try:
            llama_output = run_ollama_prompt(prompt, model='llama3')
            start = llama_output.find("[")
            end = llama_output.rfind("]") + 1
            json_block = llama_output[start:end].strip()

            charts = json.loads(json_block)

            if isinstance(charts, dict):
                charts = [charts]
            if not isinstance(charts, list):
                raise ValueError("LLaMA returned a non-list structure")

            return charts

        except Exception as e:
            last_error = str(e)
            print(f"❌ JSON parse error: {last_error}")
            with open(f"llama_dashboard_attempt_{attempt}.txt", "w", encoding="utf-8") as f:
                f.write(llama_output)

    print("⚠️ All attempts failed. Returning empty chart list.")
    return []


# Utility functions from your original code
def get_nested_value(data, path):
    if isinstance(path, list):
        for p in path:
            val = get_nested_value(data, p)
            if val is not None:
                return val
        return None

    keys = path.split('.')
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, None)
        else:
            return None

    if isinstance(data, (int, float)):
        return data
    elif isinstance(data, str):
        clean = data.replace(",", "").strip()
        if clean.startswith("(") and clean.endswith(")"):
            try:
                return -float(clean[1:-1])
            except ValueError:
                return None
        try:
            return float(clean)
        except ValueError:
            return None
    else:
        return None

def clean_price(price_str):
    if isinstance(price_str, str):
        return float(price_str.replace("€", "").replace(",", ".").strip())
    return float(price_str or 0)

# Data reading functions
def read_data(file_path_or_url):
    data = {}
    try:
        if file_path_or_url.startswith("http"):
            if not validate_google_sheets_url(file_path_or_url):
                raise ValueError("Invalid Google Sheets URL format")
            
            sheet_id = re.match(GOOGLE_SHEETS_PATTERN, file_path_or_url.strip()).group(1)
            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            
            # Add timeout and error handling for network requests
            df = pd.read_csv(export_url, timeout=30)
            if df.empty:
                raise ValueError("Google Sheet is empty or inaccessible")
            
            data["Google Sheet"] = df.fillna('')
            
        elif file_path_or_url.endswith(".csv"):
            if not os.path.exists(file_path_or_url):
                raise FileNotFoundError("CSV file not found")
            
            df = pd.read_csv(file_path_or_url)
            if df.empty:
                raise ValueError("CSV file is empty")
            
            data["CSV File"] = df.fillna('')
            
        elif file_path_or_url.endswith((".xlsx", ".xls")):
            if not os.path.exists(file_path_or_url):
                raise FileNotFoundError("Excel file not found")
            
            xls = pd.ExcelFile(file_path_or_url, engine='openpyxl')
            if not xls.sheet_names:
                raise ValueError("Excel file has no sheets")
            
            data = {sheet: xls.parse(sheet).fillna('') for sheet in xls.sheet_names}
            
        else:
            raise ValueError("Unsupported file type or URL format")
            
    except pd.errors.EmptyDataError:
        logger.error("File is empty")
        raise ValueError("The uploaded file is empty")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing file: {e}")
        raise ValueError(f"Error parsing file: {str(e)}")
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        raise
    
    return data

def format_for_prompt(data_dict):
    formatted = ""
    for sheet, df in data_dict.items():
        formatted += f"\n### Sheet: {sheet}\n"
        # Limit data size to prevent prompt overflow
        if len(df) > 1000:
            df = df.head(1000)
            formatted += "# Note: Data truncated to first 1000 rows\n"
        formatted += df.to_csv(index=False)
    return formatted

def run_ollama_prompt(prompt, model='llama3', max_retries=3):
    # Validate model parameter to prevent command injection
    allowed_models = ['llama3', 'llama2', 'mistral', 'codellama']
    if model not in allowed_models:
        logger.error(f"Invalid model: {model}")
        return f"Error: Invalid model '{model}'"
    
    # Validate prompt length
    if len(prompt) > 100000:  # 100KB limit
        logger.error("Prompt too large")
        return "Error: Input data too large"
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['ollama', 'run', model],
                input=prompt.encode('utf-8'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=180,
                check=True
            )
            return result.stdout.decode('utf-8')
            
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return "Error: Request timed out after multiple attempts"
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama process failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return f"Error running ollama: {e}"
            time.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"Unexpected error running ollama (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return f"Error running ollama: {e}"
            time.sleep(2 ** attempt)
    
    return "Error: All retry attempts failed"

def extract_json_from_response(response):
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end == 0:
            logger.warning("No JSON found in response")
            return {"raw_response": response}
        return json.loads(response[start:end])
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {"raw_response": response}
    except Exception as e:
        logger.error(f"Unexpected error parsing JSON: {e}")
        return {"raw_response": response}

# Prompt functions (simplified versions of your prompts.py)
def get_extraction_prompt(prompt_data, error_message=None):
    error_section = f"\nNote: The previous attempt failed with this parsing error:\n{error_message}\nTry to fix the JSON formatting.\n" if error_message else ""
    suggested_structure = '''
{
  "revenue_analysis": {
    "revenue": 0,
    "revenue_by_month": {}
  },
  "profit_margin_analysis": {
    "revenue": 0,
    "cost_of_goods_sold": 0,
    "gross_profit": 0,
    "net_income": 0
  },
  "cost_optimization_analysis": {
    "operating_expenses": 0,
    "inventory_costs": 0,
    "logistics_costs": 0
  }
}
'''
    return f"""
You are a financial analyst AI. Given the following spreadsheet data from a small-to-medium retail business,
convert it into a structured JSON format needed to perform:

1. Revenue analysis
2. Profit margin analysis  
3. Cost optimization analysis

Extract only the necessary data for these analyses. Only extract what's available from the data.
IMPORTANT: Any value labeled with words like "loss", "negative", "deficit", or parentheses like (1234) should be interpreted as a negative number.
Please follow this suggested JSON structure exactly as a guide. Output only valid JSON — no commentary or explanation.

{error_section}

Suggested structure:
{suggested_structure}

Spreadsheet data:
{prompt_data}
"""

def get_dashboard_prompt(json_str, error_message=None):
    error_section = f"\nNote: The previous attempt failed with this JSON parsing error:\n{error_message}\nPlease output valid JSON.\n" if error_message else ""
    
    return f"""
You are a financial dashboard AI assistant.
Given this JSON data for a small business, generate dashboards for these 3 fixed sections:
1. Revenue Analysis
2. Profit Margin Analysis  
3. Cost Optimization Analysis

Your entire response must be a valid JSON list: [{{"..."}}, {{"..."}}, ...]

Each object should have:
- "title": short chart name
- "description": what it shows
- "chart_type": one of: line, bar, pie, scatter
- "data_points": label → JSON path (e.g., "Revenue": "revenue_analysis.revenue")
- "insight": a recommendation (2–3 sentences)

{error_section}

Financial data:
{json_str}
"""

# Chart generation functions
def generate_figure(dash_config, financial_data):
    title = dash_config["title"]
    chart_type = dash_config["chart_type"].lower()
    data_points = dash_config.get("data_points", {})

    labels = list(data_points.keys())
    values = [get_nested_value(financial_data, path) or 0 for path in data_points.values()]
    
    fig = go.Figure()

    if chart_type in ["line", "time series"]:
        fig.add_trace(go.Scatter(x=labels, y=values, mode='lines+markers'))
    elif chart_type == "pie":
        fig.add_trace(go.Pie(labels=labels, values=values))
    elif chart_type == "scatter":
        fig.add_trace(go.Scatter(x=labels, y=values, mode='markers'))
    else:  # default to bar
        fig.add_trace(go.Bar(x=labels, y=values, marker_color="#f5c147"))

    fig.update_layout(
        title=title, 
        template="plotly_dark", 
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# Forecasting functions (simplified)
def prepare_prophet_input(financial_data):
    if "revenue_analysis" in financial_data and "revenue_by_month" in financial_data["revenue_analysis"]:
        monthly_data = financial_data["revenue_analysis"]["revenue_by_month"]
        return [{"ds": k, "y": float(v)} for k, v in monthly_data.items() if v]
    return []

def forecast_timeseries(data, periods=12):
    if not data or len(data) < 2:
        return None, pd.DataFrame()
    
    df = pd.DataFrame(data)
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna(subset=['y'])
    
    if len(df) < 2:
        return None, pd.DataFrame()
    
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='ME')
    forecast = model.predict(future)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                             mode='lines+markers', name='Forecast'))
    fig.add_trace(go.Scatter(x=df["ds"], y=df["y"],
                             mode='markers', name='Historical'))
    fig.update_layout(
        title="📈 Revenue Forecast",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig, forecast.tail(periods)

# Flask routes
@app.route('/')
def index():
    csrf_token = generate_csrf_token()
    return render_template('index.html', csrf_token=csrf_token)

@app.route('/upload', methods=['POST'])
@rate_limit
@require_csrf
def upload_file():
    try:
        session_id = str(uuid.uuid4())
        
        # Validate file upload
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            # Check if file is empty
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('index'))
            
            # Validate file extension
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload CSV, XLSX, or XLS files only.', 'error')
                return redirect(url_for('index'))
            
            # Check file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                flash('File too large. Maximum size is 16MB.', 'error')
                return redirect(url_for('index'))
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                data = read_data(filepath)
            finally:
                # Clean up file
                if os.path.exists(filepath):
                    os.remove(filepath)
            
        elif 'google_url' in request.form and request.form['google_url'].strip():
            url = request.form['google_url'].strip()
            
            # Validate URL
            if not validate_google_sheets_url(url):
                flash('Invalid Google Sheets URL format', 'error')
                return redirect(url_for('index'))
            
            data = read_data(url)
        else:
            flash('Please upload a file or provide a Google Sheets URL', 'error')
            return redirect(url_for('index'))
        
        if not data:
            flash('Could not read data from the provided source', 'error')
            return redirect(url_for('index'))
        
        # Extract financial data using LLaMA
        prompt_data = format_for_prompt(data)
        
        # Process with LLaMA
        prompt = get_extraction_prompt(prompt_data)
        response = run_ollama_prompt(prompt)
        financial_data = extract_json_from_response(response)
        
        # Store in session with thread safety
        with session_lock:
            session_data[session_id] = {
                'financial_data': financial_data,
                'timestamp': datetime.now()
            }
        
        logger.info(f"Created new session: {session_id}")
        return redirect(url_for('dashboard', session_id=session_id))
        
    except ValueError as e:
        flash(f'Error processing data: {str(e)}', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {e}")
        flash('An unexpected error occurred. Please try again.', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard/<session_id>')
@rate_limit
def dashboard(session_id):
    # Validate session_id format
    try:
        uuid.UUID(session_id)
    except ValueError:
        flash('Invalid session ID', 'error')
        return redirect(url_for('index'))
    
    with session_lock:
        if session_id not in session_data:
            flash('Session expired or invalid', 'error')
            return redirect(url_for('index'))
        
        financial_data = session_data[session_id]['financial_data']
    
    # Generate dashboard suggestions
    json_str = json.dumps(financial_data, indent=2)
    dashboard_prompt = get_dashboard_prompt(json_str)
    dashboard_response = run_ollama_prompt(dashboard_prompt)
    
    try:
        start = dashboard_response.find('[')
        end = dashboard_response.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in response")
        json_block = dashboard_response[start:end].strip()
        dashboards = json.loads(json_block)
        if isinstance(dashboards, dict):
            dashboards = [dashboards]
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse dashboard response: {e}")
        dashboards = [
            {
                "title": "Revenue Overview",
                "description": "Basic revenue analysis",
                "chart_type": "bar",
                "data_points": {"Revenue": "revenue_analysis.revenue"},
                "insight": "Revenue data processed successfully."
            }
        ]
    
    # Generate charts
    charts = []
    for dash_config in dashboards:
        try:
            fig = generate_figure(dash_config, financial_data)
            chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            charts.append({
                'config': dash_config,
                'chart': chart_json
            })
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            continue
    
    # Generate forecast if possible
    forecast_chart = None
    try:
        prophet_input = prepare_prophet_input(financial_data)
        if prophet_input:
            forecast_fig, _ = forecast_timeseries(prophet_input)
            if forecast_fig:
                forecast_chart = json.dumps(forecast_fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
    
    return render_template('dashboard.html', 
                         charts=charts, 
                         forecast_chart=forecast_chart,
                         financial_data=financial_data)

@app.route('/api/data/<session_id>')
@rate_limit
def get_data(session_id):
    # Validate session_id format
    try:
        uuid.UUID(session_id)
    except ValueError:
        return jsonify({'error': 'Invalid session ID'}), 400
    
    with session_lock:
        if session_id not in session_data:
            return jsonify({'error': 'Session not found'}), 404
        return jsonify(session_data[session_id]['financial_data'])

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if background cleanup is running
        cleanup_count = cleanup_old_sessions()
        return jsonify({
            'status': 'healthy',
            'active_sessions': len(session_data),
            'cleanup_count': cleanup_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

if __name__ == '__main__':
    # For AWS deployment, use:
    # app.run(host='0.0.0.0', port=5000, debug=False)
    # For local testing:
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)