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
    example_section = '''
Example spreadsheet:
date,transaction_amount
07-03-2022,260
8/23/2022,300
11/20/2022,20

Should produce:
"revenue_by_month": {
  "2022-07": 260,
  "2022-08": 300,
  "2022-11": 20
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

If the spreadsheet contains a column with dates (e.g., 'Month', 'Date', or similar) and a column with revenue (e.g., 'Revenue', 'Sales', 'Amount', 'Total', 'transaction_amount'), extract them as 'revenue_by_month' in the format {{"YYYY-MM": value, ...}}. Sum all transaction amounts for each month. Accept date formats like 'MM-DD-YYYY', 'M/D/YYYY', or 'YYYY-MM-DD'. If no such data exists, leave 'revenue_by_month' as an empty object.

{example_section}

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

def get_summary_extraction_prompt(prompt_data, error_message=None):
    error_section = f"\nNote: The previous attempt failed with this parsing error:\n{error_message}\nTry to fix the JSON formatting.\n" if error_message else ""
    suggested_structure = '''
{
  "revenue_analysis": {
    "revenue": 0
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
    example_section = '''
Example spreadsheet:
date,transaction_amount
07-03-2022,260
8/23/2022,300
11/20/2022,20

Should produce:
"revenue_analysis": {
  "revenue": 580
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

If the spreadsheet contains a column with revenue (e.g., 'Revenue', 'Sales', 'Amount', 'Total', 'transaction_amount'), sum all values in that column for the total revenue. If no such data exists, set revenue to 0.

{example_section}

Please follow this suggested JSON structure exactly as a guide. Output only valid JSON — no commentary or explanation.

{error_section}

Suggested structure:
{suggested_structure}

Spreadsheet data:
{prompt_data}
"""

def get_forecast_extraction_prompt(prompt_data, error_message=None):
    error_section = f"\nNote: The previous attempt failed with this parsing error:\n{error_message}\nTry to fix the JSON formatting.\n" if error_message else ""
    suggested_structure = '''
{
  "sku_transactions": {
    "SKU_NAME_1": [
      { "date": "YYYY-MM-DD", "amount": 123.45 },
      { "date": "YYYY-MM-DD", "amount": 67.89 }
    ],
    "SKU_NAME_2": [
      { "date": "YYYY-MM-DD", "amount": 11.22 }
    ]
  }
}
'''
    example_section = '''
Example spreadsheet:
date,item_name,transaction_amount
07-03-2022,Aalopuri,260
08-23-2022,Vadapav,300
07-12-2022,Vadapav,160

Should produce:
"sku_transactions": {
  "Aalopuri": [ { "date": "07-03-2022", "amount": 260 } ],
  "Vadapav": [ { "date": "08-23-2022", "amount": 300 }, { "date": "07-12-2022", "amount": 160 } ]
}
'''
    return f"""
You are a financial analyst AI. Given the following spreadsheet data, extract all individual transactions for each SKU (item_name).

The output JSON should have a single root key: "sku_transactions".
The value should be an object where each key is a unique 'item_name' (SKU) from the data.
The value for each SKU should be a LIST of objects, where each object represents a single transaction with "date" and "amount" (from the 'transaction_amount' column).
Keep the original date string. DO NOT aggregate the data.

{example_section}

Please follow this suggested JSON structure exactly. Output only valid JSON — no commentary or explanation.

{error_section}

Suggested structure:
{suggested_structure}

Spreadsheet data:
{prompt_data}
""" 