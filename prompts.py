# prompts.py

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

If the spreadsheet contains a column with months (e.g., 'Month', 'Date', or similar) and a column with revenue (e.g., 'Revenue', 'Sales'), extract them as 'revenue_by_month' in the format {{"YYYY-MM": value, ...}}. Use the first day of the month if only the month and year are given. If no such data exists, leave 'revenue_by_month' as an empty object.

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

Respond with only a valid JSON array. Do not include any text before or after the JSON. Do not include markdown, comments, or explanations.

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