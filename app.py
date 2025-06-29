from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pandas as pd
import json
import os
import subprocess
import plotly.graph_objs as go
import plotly.utils
import numpy as np
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime, timedelta
import logging
from functools import wraps
import threading
import time
import re
from prompts import get_extraction_prompt, get_dashboard_prompt

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

# Configure TBB to prevent "already found in load path" error
os.environ['TBB_DISABLE_PREVIEW'] = '1'
os.environ['TBB_PREVIEW_WAIT_FOR_OLD_PREVIEW'] = '0'

# Import Prophet after TBB configuration
try:
    from prophet import Prophet
except ImportError as e:
    print(f"Warning: Prophet not available - {e}")
    Prophet = None

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
            logger.info(f"Read DataFrame from Google Sheet: {df.head()}")
            
        elif file_path_or_url.endswith(".csv"):
            if not os.path.exists(file_path_or_url):
                raise FileNotFoundError("CSV file not found")
            
            df = pd.read_csv(file_path_or_url)
            if df.empty:
                raise ValueError("CSV file is empty")
            
            data["CSV File"] = df.fillna('')
            logger.info(f"Read DataFrame from CSV: {df.head()}")
            
        elif file_path_or_url.endswith((".xlsx", ".xls")):
            if not os.path.exists(file_path_or_url):
                raise FileNotFoundError("Excel file not found")
            with pd.ExcelFile(file_path_or_url, engine='openpyxl') as xls:
                if not xls.sheet_names:
                    raise ValueError("Excel file has no sheets")
                data = {sheet: xls.parse(sheet).fillna('') for sheet in xls.sheet_names}
                for sheet, df in data.items():
                    logger.info(f"Read DataFrame from Excel sheet '{sheet}': {df.head()}")
            
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
    allowed_models = ['llama3', 'llama2', 'mistral', 'codellama', 'phi', 'phi3', 'tinyllama']
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
                timeout=600,
                check=True
            )
            output = result.stdout.decode('utf-8')
            if not output.strip():
                logger.warning(f"Ollama returned empty output (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return "Error: Ollama returned empty response"
                continue
            return output
            
        except subprocess.TimeoutExpired:
            logger.error(f"Ollama request timed out (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return "Error: Request timed out after multiple attempts"
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.decode('utf-8') if e.stderr else "No stderr output"
            logger.error(f"Ollama process failed (attempt {attempt + 1}/{max_retries}): {e}")
            logger.error(f"Ollama stderr: {stderr_output}")
            if attempt == max_retries - 1:
                return f"Error running ollama: {e}. Stderr: {stderr_output}"
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

def forecast_timeseries(data, periods=12, title="📈 Revenue Forecast"):
    if data.empty or len(data) < 2:
        return None, pd.DataFrame()
    
    df = pd.DataFrame(data)
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.dropna(subset=['y'])
    
    if len(df) < 2:
        return None, pd.DataFrame()
    
    # Check if Prophet is available
    if Prophet is None:
        logger.warning("Prophet not available, using simple trend analysis")
        return create_simple_revenue_forecast(df, title)
    
    try:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                 mode='lines+markers', name='Forecast'))
        fig.add_trace(go.Scatter(x=df["ds"], y=df["y"],
                                 mode='markers', name='Historical'))
        fig.update_layout(
            title=title, 
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig, forecast.tail(periods)
        
    except Exception as e:
        logger.error(f"Prophet forecast failed: {e}")
        return create_simple_revenue_forecast(df, title)

def create_simple_revenue_forecast(df, title):
    """Create simple revenue forecast when Prophet is not available"""
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=df["ds"], 
        y=df["y"],
        mode='markers+lines', 
        name='Historical',
        marker=dict(color='#ffd700', size=8),
        line=dict(color='#f5c147', width=2)
    ))
    
    # Add simple trend line
    if len(df) > 1:
        z = np.polyfit(range(len(df)), df['y'], 1)
        p = np.poly1d(z)
        
        # Extend trend line for future months
        future_months = pd.date_range(
            start=df['ds'].iloc[-1], 
            periods=13, 
            freq='M'
        )[1:]  # Skip the last historical month
        
        future_values = p(range(len(df), len(df) + len(future_months)))
        
        fig.add_trace(go.Scatter(
            x=future_months,
            y=future_values,
            mode='lines',
            name='Simple Trend Forecast',
            line=dict(color='#f5c147', width=3, dash='dot')
        ))
    
    fig.update_layout(
        title=f"{title} (Simple Analysis)",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig, pd.DataFrame()

def get_chart_insight_prompt(chart_title, chart_data_csv):
    return f"""
You are a business analyst. Given the following chart titled '{chart_title}' and its data:

{chart_data_csv}

Write a concise, actionable business insight (2-3 sentences) for a small business owner. Do not repeat the chart title.
"""

def get_chatbot_prompt(question, financial_data_str, chart_data_str):
    return f"""
You are a helpful AI business analyst assistant. A user is asking questions about their financial data and dashboard insights.

Financial Data Summary:
{financial_data_str}

Available Charts and Insights:
{chart_data_str}

SKU Performance Scoring System:
- Performance scores (0-100) are calculated using a weighted formula:
  * 40% Frequency Score: Based on number of transactions (normalized, capped at 10 transactions)
  * 40% Price Score: Based on average transaction amount (normalized, capped at $1000)
  * 20% Consistency Score: Based on price stability (lower volatility = higher score)
- Scores are color-coded: Green (70+), Orange (40-69), Red (0-39)
- SKUs are ranked by performance score (best performing first)
- Each SKU has two charts: Performance Score Forecast and Transaction History

User Question: {question}

Please provide a helpful, professional response that:
1. Answers the user's question based on the available data
2. References specific insights or trends from the charts when relevant
3. Explains the performance scoring system when discussing SKU analysis
4. References specific SKU performance scores and rankings when applicable
5. Provides actionable business advice when appropriate
6. Keeps the response concise but informative (2-4 sentences)
7. Uses a friendly, professional tone

Response:"""

def calculate_sku_score(transactions):
    """Calculate weighted score for SKU based on frequency and price"""
    if not transactions or len(transactions) == 0:
        return 0
    
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    if df.empty:
        return 0
    
    # Calculate metrics
    total_transactions = len(df)
    avg_price = df['amount'].mean()
    price_volatility = df['amount'].std() / avg_price if avg_price > 0 else 0
    
    # Weighted scoring formula
    # Higher frequency = higher score
    # Higher average price = higher score  
    # Lower volatility = higher score (more consistent pricing)
    frequency_score = min(total_transactions / 10, 1.0)  # Normalize to 0-1, cap at 10 transactions
    price_score = min(avg_price / 1000, 1.0)  # Normalize to 0-1, cap at $1000
    consistency_score = max(1 - price_volatility, 0)  # Lower volatility = higher score
    
    # Weighted average (can adjust weights)
    weighted_score = (0.4 * frequency_score + 0.4 * price_score + 0.2 * consistency_score) * 100
    
    return round(weighted_score, 2)

def prepare_sku_forecast_data(transactions):
    """Prepare monthly aggregated data with weighted scores for forecasting"""
    if not transactions or len(transactions) == 0:
        return pd.DataFrame()
    
    df = pd.DataFrame(transactions)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    
    if df.empty:
        return pd.DataFrame()
    
    # Group by month and calculate metrics
    df['month'] = df['date'].dt.to_period('M')
    monthly_data = df.groupby('month').agg({
        'amount': ['sum', 'count', 'mean', 'std']
    }).reset_index()
    
    monthly_data.columns = ['month', 'total_amount', 'transaction_count', 'avg_price', 'price_std']
    
    # Calculate weighted score for each month
    monthly_data['score'] = monthly_data.apply(
        lambda row: calculate_monthly_score(row['transaction_count'], row['avg_price'], row['price_std']), 
        axis=1
    )
    
    # Convert period to datetime for Prophet
    monthly_data['ds'] = monthly_data['month'].dt.to_timestamp()
    monthly_data['y'] = monthly_data['score']
    
    return monthly_data[['ds', 'y']]

def calculate_monthly_score(transaction_count, avg_price, price_std):
    """Calculate weighted score for a specific month"""
    if transaction_count == 0 or avg_price == 0:
        return 0
    
    # Normalize metrics
    frequency_score = min(transaction_count / 5, 1.0)  # Cap at 5 transactions per month
    price_score = min(avg_price / 500, 1.0)  # Cap at $500 average price
    consistency_score = max(1 - (price_std / avg_price if avg_price > 0 else 0), 0)
    
    # Weighted average
    weighted_score = (0.4 * frequency_score + 0.4 * price_score + 0.2 * consistency_score) * 100
    
    return round(weighted_score, 2)

def forecast_sku_performance(transactions, sku_name, periods=12):
    """Forecast SKU performance using weighted scoring system"""
    if not transactions or len(transactions) == 0:
        return None, None, None
    
    # Check if Prophet is available
    if Prophet is None:
        logger.warning("Prophet not available, using simple trend analysis")
        return create_simple_forecast(transactions, sku_name)
    
    # Prepare forecast data
    forecast_data = prepare_sku_forecast_data(transactions)
    
    if forecast_data.empty or len(forecast_data) < 2:
        return create_simple_forecast(transactions, sku_name)
    
    try:
        # Generate forecast using Prophet
        model = Prophet()
        model.fit(forecast_data)
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)
        
        # Create forecast chart
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast["ds"], 
            y=forecast["yhat"],
            mode='lines+markers', 
            name='Forecasted Score',
            line=dict(color='#f5c147', width=3)
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data["ds"], 
            y=forecast_data["y"],
            mode='markers', 
            name='Historical Score',
            marker=dict(color='#ffd700', size=8)
        ))
        
        fig_forecast.update_layout(
            title=f"📊 Performance Score Forecast - {sku_name}",
            xaxis_title="Date",
            yaxis_title="Performance Score",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
    except Exception as e:
        logger.error(f"Prophet forecast failed for {sku_name}: {e}")
        return create_simple_forecast(transactions, sku_name)
    
    # Create transaction detail chart
    df_transactions = pd.DataFrame(transactions)
    df_transactions['date'] = pd.to_datetime(df_transactions['date'], errors='coerce')
    df_transactions.dropna(subset=['date'], inplace=True)
    df_transactions = df_transactions.sort_values('date')
    
    fig_transactions = go.Figure()
    fig_transactions.add_trace(go.Scatter(
        x=df_transactions['date'],
        y=df_transactions['amount'],
        mode='markers+lines',
        name='Individual Transactions',
        marker=dict(color='#ffd700', size=6),
        line=dict(color='#f5c147', width=1)
    ))
    
    # Add trend line
    if len(df_transactions) > 1:
        z = np.polyfit(range(len(df_transactions)), df_transactions['amount'], 1)
        p = np.poly1d(z)
        fig_transactions.add_trace(go.Scatter(
            x=df_transactions['date'],
            y=p(range(len(df_transactions))),
            mode='lines',
            name='Trend Line',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
    
    fig_transactions.update_layout(
        title=f"💰 Transaction History - {sku_name}",
        xaxis_title="Date",
        yaxis_title="Transaction Amount ($)",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Calculate overall SKU score
    overall_score = calculate_sku_score(transactions)
    
    return fig_forecast, fig_transactions, overall_score

def create_simple_forecast(transactions, sku_name):
    """Create simple forecast when Prophet is not available"""
    if not transactions or len(transactions) == 0:
        return None, None, None
    
    # Prepare data
    forecast_data = prepare_sku_forecast_data(transactions)
    
    if forecast_data.empty or len(forecast_data) < 2:
        return None, None, None
    
    # Create simple forecast chart (no Prophet)
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data["ds"], 
        y=forecast_data["y"],
        mode='markers+lines', 
        name='Historical Score',
        marker=dict(color='#ffd700', size=8),
        line=dict(color='#f5c147', width=2)
    ))
    
    # Add simple trend line
    if len(forecast_data) > 1:
        z = np.polyfit(range(len(forecast_data)), forecast_data['y'], 1)
        p = np.poly1d(z)
        
        # Extend trend line for future months
        future_months = pd.date_range(
            start=forecast_data['ds'].iloc[-1], 
            periods=13, 
            freq='M'
        )[1:]  # Skip the last historical month
        
        future_scores = p(range(len(forecast_data), len(forecast_data) + len(future_months)))
        
        fig_forecast.add_trace(go.Scatter(
            x=future_months,
            y=future_scores,
            mode='lines',
            name='Simple Trend Forecast',
            line=dict(color='#f5c147', width=3, dash='dot')
        ))
    
    fig_forecast.update_layout(
        title=f"📊 Performance Score Trend - {sku_name} (Simple Analysis)",
        xaxis_title="Date",
        yaxis_title="Performance Score",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create transaction detail chart
    df_transactions = pd.DataFrame(transactions)
    df_transactions['date'] = pd.to_datetime(df_transactions['date'], errors='coerce')
    df_transactions.dropna(subset=['date'], inplace=True)
    df_transactions = df_transactions.sort_values('date')
    
    fig_transactions = go.Figure()
    fig_transactions.add_trace(go.Scatter(
        x=df_transactions['date'],
        y=df_transactions['amount'],
        mode='markers+lines',
        name='Individual Transactions',
        marker=dict(color='#ffd700', size=6),
        line=dict(color='#f5c147', width=1)
    ))
    
    # Add trend line
    if len(df_transactions) > 1:
        z = np.polyfit(range(len(df_transactions)), df_transactions['amount'], 1)
        p = np.poly1d(z)
        fig_transactions.add_trace(go.Scatter(
            x=df_transactions['date'],
            y=p(range(len(df_transactions))),
            mode='lines',
            name='Trend Line',
            line=dict(color='#ff6b6b', width=2, dash='dash')
        ))
    
    fig_transactions.update_layout(
        title=f"💰 Transaction History - {sku_name}",
        xaxis_title="Date",
        yaxis_title="Transaction Amount ($)",
        template="plotly_dark",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Calculate overall SKU score
    overall_score = calculate_sku_score(transactions)
    
    return fig_forecast, fig_transactions, overall_score

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
        # Select prompt based on user mode
        mode = request.form.get('mode', 'summary')
        if mode == 'forecast':
            from prompts import get_forecast_extraction_prompt
            prompt_func = get_forecast_extraction_prompt
        else:
            from prompts import get_summary_extraction_prompt
            prompt_func = get_summary_extraction_prompt
        last_error = ""
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                extraction_prompt = prompt_func(prompt_data, error_message=last_error)
            else:
                extraction_prompt = prompt_func(prompt_data)
            response = run_ollama_prompt(extraction_prompt, model='llama3')
            logger.info(f"Raw LLM extraction output (attempt {attempt}): {response}")
            try:
                financial_data = extract_json_from_response(response)
                if isinstance(financial_data, dict) and 'raw_response' not in financial_data:
                    break
                else:
                    raise ValueError("Malformed JSON")
            except Exception as e:
                last_error = str(e)
                with open(f"llama_extraction_attempt_{attempt}.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                if attempt == max_attempts:
                    flash('Failed to extract financial data from your file. Please try again.', 'error')
                    return redirect(url_for('index'))
        
        # Store in session with thread safety
        with session_lock:
            session_data[session_id] = {
                'financial_data': financial_data,
                'timestamp': datetime.now(),
                'mode': mode
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
        
        session_info = session_data[session_id]
        financial_data = session_info['financial_data']
        mode = session_info.get('mode', 'summary')
    
    charts = []
    forecast_chart = None
    sku_forecast_charts = []
    
    if mode == 'forecast':
        if 'sku_transactions' in financial_data:
            sku_analysis = []
            for sku, transactions in financial_data['sku_transactions'].items():
                if not transactions or not isinstance(transactions, list):
                    continue
                try:
                    # Generate both forecast and transaction charts
                    forecast_fig, transaction_fig, overall_score = forecast_sku_performance(transactions, sku)
                    
                    if forecast_fig and transaction_fig:
                        sku_analysis.append({
                            'sku_name': sku,
                            'forecast_chart': json.dumps(forecast_fig, cls=plotly.utils.PlotlyJSONEncoder),
                            'transaction_chart': json.dumps(transaction_fig, cls=plotly.utils.PlotlyJSONEncoder),
                            'overall_score': overall_score,
                            'transaction_count': len(transactions),
                            'total_revenue': sum(t['amount'] for t in transactions if isinstance(t, dict) and 'amount' in t),
                            'avg_price': sum(t['amount'] for t in transactions if isinstance(t, dict) and 'amount' in t) / len(transactions) if transactions else 0
                        })
                except Exception as e:
                    logger.error(f"Error generating forecast for SKU '{sku}': {e}")
                    continue
            
            # Sort SKUs by overall score (best performing first)
            sku_analysis.sort(key=lambda x: x['overall_score'], reverse=True)
            sku_forecast_charts = sku_analysis
    else:
        # Predetermined summary charts
        summary_charts = [
            {
                "title": "Revenue Trends",
                "description": "Monthly revenue trends",
                "chart_type": "line",
                "data_points": {},  # Will be filled below
                "insight": ""
            },
            {
                "title": "Profit Margin Breakdown",
                "description": "Breakdown of profit margin components",
                "chart_type": "pie",
                "data_points": {
                    "Revenue": "profit_margin_analysis.revenue",
                    "Cost of Goods Sold": "profit_margin_analysis.cost_of_goods_sold",
                    "Gross Profit": "profit_margin_analysis.gross_profit",
                    "Net Income": "profit_margin_analysis.net_income"
                },
                "insight": ""
            },
            {
                "title": "Cost Categories",
                "description": "Breakdown of company costs",
                "chart_type": "bar",
                "data_points": {
                    "Operating Expenses": "cost_optimization_analysis.operating_expenses",
                    "Inventory Costs": "cost_optimization_analysis.inventory_costs",
                    "Logistics Costs": "cost_optimization_analysis.logistics_costs"
                },
                "insight": ""
            }
        ]
        # Fill in monthly revenue for Revenue Trends
        monthly = None
        if (
            "revenue_analysis" in financial_data and
            "revenue_by_month" in financial_data["revenue_analysis"] and
            isinstance(financial_data["revenue_analysis"]["revenue_by_month"], dict)
        ):
            monthly = financial_data["revenue_analysis"]["revenue_by_month"]
        if monthly and len(monthly) > 0:
            summary_charts[0]["data_points"] = {k: f"revenue_analysis.revenue_by_month.{k}" for k in monthly.keys()}
        else:
            summary_charts[0]["data_points"] = {"Revenue": "revenue_analysis.revenue"}
        # Generate charts with LLM insights
        for dash_config in summary_charts:
            try:
                # Prepare chart data as CSV for the LLM
                chart_data = {}
                for label, path in dash_config["data_points"].items():
                    value = get_nested_value(financial_data, path)
                    chart_data[label] = value
                chart_df = pd.DataFrame(list(chart_data.items()), columns=["Label", "Value"])
                chart_data_csv = chart_df.to_csv(index=False)
                # Get LLM insight
                insight_prompt = get_chart_insight_prompt(dash_config["title"], chart_data_csv)
                insight = run_ollama_prompt(insight_prompt, model='llama3')
                dash_config["insight"] = insight.strip()
                fig = generate_figure(dash_config, financial_data)
                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                charts.append({
                    'config': dash_config,
                    'chart': chart_json
                })
            except Exception as e:
                logger.error(f"Error generating chart: {e}")
                continue
    
    # Generate overall forecast for summary mode if data is available
    if mode == 'summary':
        try:
            prophet_input = prepare_prophet_input(financial_data)
            if prophet_input:
                forecast_fig, _ = forecast_timeseries(prophet_input)
                if forecast_fig:
                    forecast_chart = json.dumps(forecast_fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
    
    logger.info(f"SKU Forecast Charts generated: {len(sku_forecast_charts)}")
    
    # Store charts data in session for chatbot access
    with session_lock:
        if session_id in session_data:
            session_data[session_id]['charts'] = charts
            session_data[session_id]['forecast_chart'] = forecast_chart
            session_data[session_id]['sku_forecast_charts'] = sku_forecast_charts
    
    return render_template('dashboard.html', 
                         charts=charts, 
                         forecast_chart=forecast_chart,
                         sku_forecast_charts=sku_forecast_charts,
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

@app.route('/api/chat/<session_id>', methods=['POST'])
@rate_limit
def chat_with_data(session_id):
    """Handle chatbot messages about the financial data"""
    try:
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        # Get session data
        with session_lock:
            if session_id not in session_data:
                return jsonify({'error': 'Session not found'}), 404
            
            session_info = session_data[session_id]
            financial_data = session_info['financial_data']
        
        # Get user message
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Prepare data for the prompt
        financial_data_str = json.dumps(financial_data, indent=2)
        
        # Get chart data from the session
        chart_data = []
        if 'charts' in session_info:
            chart_data = session_info['charts']
        
        # Get SKU analysis data
        sku_analysis_data = []
        if 'sku_forecast_charts' in session_info:
            sku_analysis_data = session_info['sku_forecast_charts']
        
        chart_data_str = ""
        
        # Add regular charts
        for i, chart in enumerate(chart_data):
            chart_data_str += f"\nChart {i+1}: {chart['config']['title']}\n"
            chart_data_str += f"Description: {chart['config']['description']}\n"
            chart_data_str += f"Insight: {chart['config']['insight']}\n"
        
        # Add SKU analysis data
        if sku_analysis_data:
            chart_data_str += f"\n📊 SKU Performance Analysis ({len(sku_analysis_data)} SKUs):\n"
            for i, sku in enumerate(sku_analysis_data):
                chart_data_str += f"\nSKU {i+1}: {sku['sku_name']}\n"
                chart_data_str += f"  Performance Score: {sku['overall_score']}/100\n"
                chart_data_str += f"  Transaction Count: {sku['transaction_count']}\n"
                chart_data_str += f"  Total Revenue: ${sku['total_revenue']:.2f}\n"
                chart_data_str += f"  Average Price: ${sku['avg_price']:.2f}\n"
                chart_data_str += f"  Charts: Performance Score Forecast + Transaction History\n"
        
        # Add mode information
        mode = session_info.get('mode', 'summary')
        chart_data_str += f"\nAnalysis Mode: {mode.upper()}\n"
        
        # Generate response using LLaMA
        prompt = get_chatbot_prompt(user_message, financial_data_str, chart_data_str)
        response = run_ollama_prompt(prompt, model='llama3')
        
        # Clean up the response
        response = response.strip()
        if response.startswith("Response:"):
            response = response[9:].strip()
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat_with_data: {e}")
        return jsonify({'error': 'An error occurred while processing your message'}), 500

if __name__ == '__main__':
    # For AWS deployment, use:
    # app.run(host='0.0.0.0', port=5000, debug=False)
    # For local testing:
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)