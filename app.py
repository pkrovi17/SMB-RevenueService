from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import json
import os
import subprocess
import plotly.graph_objs as go
import plotly.utils
from prophet import Prophet
import numpy as np
from werkzeug.utils import secure_filename
import tempfile
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store session data in memory (for demo - use Redis/DB in production)
session_data = {}

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
            except:
                return None
        try:
            return float(clean)
        except:
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
            if "docs.google.com" in file_path_or_url:
                sheet_id = file_path_or_url.split("/d/")[1].split("/")[0]
                export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                df = pd.read_csv(export_url)
                data["Google Sheet"] = df.fillna('')
            else:
                raise ValueError("Only Google Sheets URLs are supported for now.")
        elif file_path_or_url.endswith(".csv"):
            df = pd.read_csv(file_path_or_url)
            data["CSV File"] = df.fillna('')
        elif file_path_or_url.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(file_path_or_url, engine='openpyxl')
            data = {sheet: xls.parse(sheet).fillna('') for sheet in xls.sheet_names}
        else:
            raise ValueError("Unsupported file type or URL format.")
    except Exception as e:
        print(f"Error reading file: {e}")
    return data

def format_for_prompt(data_dict):
    formatted = ""
    for sheet, df in data_dict.items():
        formatted += f"\n### Sheet: {sheet}\n"
        formatted += df.to_csv(index=False)
    return formatted

def run_ollama_prompt(prompt, model='llama3'):
    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt.encode('utf-8'),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90
        )
        return result.stdout.decode('utf-8')
    except Exception as e:
        return f"Error running ollama: {e}"

def extract_json_from_response(response):
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        return json.loads(response[start:end])
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
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
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        session_id = str(uuid.uuid4())
        
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            data = read_data(filepath)
            os.remove(filepath)  # Clean up
            
        elif 'google_url' in request.form and request.form['google_url'].strip():
            url = request.form['google_url'].strip()
            data = read_data(url)
        else:
            flash('Please upload a file or provide a Google Sheets URL', 'error')
            return redirect(url_for('index'))
        
        if not data:
            flash('Could not read data from the provided source', 'error')
            return redirect(url_for('index'))
        
        # Extract financial data using LLaMA
        prompt_data = format_for_prompt(data)
        mode = request.form.get('mode', 'summary')
        
        # Process with LLaMA
        prompt = get_extraction_prompt(prompt_data)
        response = run_ollama_prompt(prompt)
        financial_data = extract_json_from_response(response)
        
        # Store in session
        session_data[session_id] = {
            'financial_data': financial_data,
            'timestamp': datetime.now()
        }
        
        return redirect(url_for('dashboard', session_id=session_id))
        
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard/<session_id>')
def dashboard(session_id):
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
        json_block = dashboard_response[start:end].strip()
        dashboards = json.loads(json_block)
        if isinstance(dashboards, dict):
            dashboards = [dashboards]
    except:
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
        fig = generate_figure(dash_config, financial_data)
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        charts.append({
            'config': dash_config,
            'chart': chart_json
        })
    
    # Generate forecast if possible
    forecast_chart = None
    prophet_input = prepare_prophet_input(financial_data)
    if prophet_input:
        forecast_fig, _ = forecast_timeseries(prophet_input)
        if forecast_fig:
            forecast_chart = json.dumps(forecast_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('dashboard.html', 
                         charts=charts, 
                         forecast_chart=forecast_chart,
                         financial_data=financial_data)

@app.route('/api/data/<session_id>')
def get_data(session_id):
    if session_id not in session_data:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify(session_data[session_id]['financial_data'])

if __name__ == '__main__':
    # For AWS deployment, use:
    # app.run(host='0.0.0.0', port=5000, debug=False)
    # For local testing:
    app.run(host='0.0.0.0', port=5000, debug=True)