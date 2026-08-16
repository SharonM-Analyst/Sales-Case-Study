# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Sales Analytics EDA
# MAGIC
# MAGIC Reads the daily sales dataset and computes:
# MAGIC   1. Daily sales price per unit
# MAGIC   2. Average unit sales price
# MAGIC   3. Daily % gross profit
# MAGIC   4. Daily % gross profit per unit
# MAGIC   5. Promotion detection + Price Elasticity of Demand for 3 promo periods
# MAGIC   6. Extra insights: rolling averages, day-of-week, seasonality, monthly/yearly trends, YoY growth, negative-GP-day analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Secton 1 : Libraries & Data Overview

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Install Library

# COMMAND ----------

pip install openpyxl numpy pandas matplotlib seaborn scikit-learn

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Import Libraries

# COMMAND ----------

import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
from   plotly.offline	import	iplot
import plotly.graph_objects as go
from   plotly.subplots	import	make_subplots
import plotly.figure_factory as ff

# filter warning
import warnings


# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Data Loading and Overview

# COMMAND ----------

df = pd.read_excel('/Workspace/Repos/sshanay92@gmail.com/Sales-Case-Study/1. Project Description & Raw Data/Sales_Case_Study_2021_Raw Dataset.xlsx')


# Data Overview
display(df.sample(5, random_state=42))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 2: Data Inspection & Understanding

# COMMAND ----------

rows, cols = df.shape
print(f"Rows    : {rows:,}")      # print(f"Rows    : {rows:,}")  also creates an alias
print(f"Columns : {cols}")
print(f"Shape   : {df.shape}")

# COMMAND ----------

# Using Python Loop To enumerate Columns names
print("Columns in the dataset:")                  # alias/Heading
for i, col in enumerate(df.columns, 1):           # loop
    print(f"  {i:>2}. {col}")

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC - No nulls
# MAGIC - No incorrect data type

# COMMAND ----------

df.describe().T

# COMMAND ----------

# MAGIC %md
# MAGIC ## Section 3 Data Quality Check

# COMMAND ----------

# MAGIC %md
# MAGIC - There are no nulls

# COMMAND ----------

# Are there duplicates
dup_count = df.duplicated().sum()
print(f"Exact duplicate rows : {dup_count}")                # Embedded variable{dup_count} in   f-string 


# COMMAND ----------

print("\n=== Global Check (Any Column) ===")

# Checks every column in the entire table for leading whitespace
total_rows_leading = df.map(lambda x: str(x).startswith(' ')).any(axis=1).sum()
print(f"Total rows in table containing leading spaces: {total_rows_leading}")


# COMMAND ----------

# MAGIC %md
# MAGIC - Checks unusual numeric values in Sales, Quantity Sold and Sales Cost. Further checks where Cost of sales was higher than sales.
# MAGIC

# COMMAND ----------


print("=== Sales ===")
print(f"Min: R{df['Sales'].min():,.2f}  |  Max: R{df['Sales'].max():,.2f}")
neg_sales = (df['Sales'] < 0).sum()
print(f"Negative sales values: {neg_sales} ({neg_sales/len(df)*100:.2f}%)")



print("\n=== Quantity Sold ===")
print(f"Min: {df['Quantity Sold'].min():,}  |  Max: {df['Quantity Sold'].max():,}")
neg_qty = (df['Quantity Sold'] < 0).sum()
zero_qty_sales = ((df['Quantity Sold'] == 0) & (df['Sales'] > 0)).sum()
print(f"Negative quantities  : {neg_qty} ({neg_qty/len(df)*100:.2f}%)")
print(f"Zero qty with sales  : {zero_qty_sales} ({zero_qty_sales/len(df)*100:.2f}%)")



print("\n=== Cost of Sales ===")
print(f"Min: R{df['Cost Of Sales'].min():,.2f}  |  Max: R{df['Cost Of Sales'].max():,.2f}")
neg_cos = (df['Cost Of Sales'] < 0).sum()
high_cos = (df['Cost Of Sales'] > df['Sales']).sum()
print(f"Negative cost values : {neg_cos} ({neg_cos/len(df)*100:.2f}%)")
print(f"COS exceeds Sales    : {high_cos} ({high_cos/len(df)*100:.2f}%)")



print("\n=== Sample of Unusual Rows (First 5) ===")
# Quickly display rows where Cost exceeds Sales or numbers are negative
unusual_mask = (df['Sales'] < 0) | (df['Quantity Sold'] < 0) | (df['Cost Of Sales'] > df['Sales'])
print(df[unusual_mask][['Sales', 'Quantity Sold', 'Cost Of Sales']].head(5))


# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Quality Check Summary
# MAGIC - No nulls
# MAGIC - No doplicates
# MAGIC - No Unusual Numeric Values
# MAGIC - D.types arecorrect.
# MAGIC
# MAGIC #### No Need To Complete Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 4 Feature Engineering

# COMMAND ----------

df_clean = df.copy()


# COMMAND ----------

# Date-based Features
df_clean['Year of Sales']         = df_clean['Date'].dt.year
df_clean['Sales Quarter']         = df_clean['Date'].dt.quarter
df_clean['Sales Month Name']      = df_clean['Date'].dt.month_name()
df_clean['Weekly Sales']          = df_clean['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
df_clean['Sales Day Name']        = df_clean['Date'].dt.day_name()



# COMMAND ----------

# Seasonal Feature (Southern Hemisphere: Summer=Dec-Feb, Autumn=Mar-May, Winter=Jun-Aug, Spring=Sep-Nov)
def get_season(month):
    if month in ["December", "January", "February"]:
        return 'Summer'
    elif month in ["March", "April", "May"]:
        return 'Autumn'
    elif month in ["June", "July", "August"]:
        return 'Winter'
    else:
        return 'Spring'

df_clean['Sales Season'] = df_clean['Sales Month Name'].apply(get_season)


# COMMAND ----------


# Preview new columns only
print("Date-based columns created:")
display(df_clean[['Year of Sales','Sales Month Name','Sales Month Name',
                  'Sales Day Name','Sales Quarter','Weekly Sales','Sales Season']].head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Key Metrics / KPIs

# COMMAND ----------

# Key Metrics / KPIs 

df_clean['Gross Profit'] = df_clean['Sales'] - df_clean['Cost Of Sales']
df_clean['Gross Profit Margin (%)'] = (df_clean['Gross Profit'] / df_clean['Sales']) * 100
df_clean['Average Selling Price'] = df_clean['Sales'] / df_clean['Quantity Sold']
df_clean['Cost per Unit'] = df_clean['Cost Of Sales'] / df_clean['Quantity Sold']
df_clean['Gross Profit per Unit'] = (df_clean['Average Selling Price'] - df_clean['Cost per Unit'])
df_clean['Gross Profit Margin per Unit (%)'] = (df_clean['Gross Profit per Unit'] / df_clean['Average Selling Price']) * 100

# Preview new KPIs
display(df_clean[['Sales', 'Cost Of Sales', 'Quantity Sold', 'Gross Profit', 'Gross Profit Margin (%)', 'Average Selling Price', 'Cost per Unit']].head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ROLLING AVERAGES
# MAGIC - These calculations smooth out daily fluctuations so that trends are easier to see over time.

# COMMAND ----------

# Calculate the 7-day average selling price
df_clean['7-Day Average Selling Price'] = (df_clean['Average Selling Price'].rolling(window=7, min_periods=1).mean())

# Calculate the 30-day average selling price
df_clean['30-Day Average Selling Price'] = (df_clean['Average Selling Price'].rolling(window=30, min_periods=1).mean())

# Calculate how much the selling price varies over the last 30 days
# A higher value means prices have been changing more.
df_clean['30-Day Selling Price Variation'] = (df_clean['Average Selling Price'].rolling(window=30, min_periods=1).std())

# Calculate the 7-day average gross profit margin
df_clean['7-Day Average Gross Profit Margin (%)'] = (df_clean['Gross Profit Margin (%)'].rolling(window=7, min_periods=1).mean())

# Calculate the 30-day average gross profit margin
df_clean['30-Day Average Gross Profit Margin (%)'] = (df_clean['Gross Profit Margin (%)'].rolling(window=30, min_periods=1).mean())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### PROMOTION DETECTION
# MAGIC - A day is classified as a promotion day when BOTH of the following are true:
# MAGIC
# MAGIC 1. The Average Selling Price is significantly lower than its normal level.
# MAGIC     - This is identified when the selling price is more than one standard deviation below the 30-day rolling average selling price.
# MAGIC 2. The Quantity Sold is higher than its normal level.
# MAGIC     - This is identified when the quantity sold is above the 30-day rolling average quantity sold.
# MAGIC     - A lower selling price combined with higher sales volume is a strong indication that a promotion or discount campaign was running.
# MAGIC

# COMMAND ----------

# Calculate the 30-day rolling average quantity sold
df_clean['30-Day Average Quantity Sold'] = (
    df_clean['Quantity Sold']
    .rolling(window=30, min_periods=1)
    .mean()
)


# Condition 1:
# The selling price is unusually lower than normal

df_clean['Low Price Condition'] = (
    df_clean['Average Selling Price']
    <
    (
        df_clean['30-Day Average Selling Price']
        - df_clean['30-Day Selling Price Variation']
    )
)


# Condition 2:
# The quantity sold is higher than normal

df_clean['High Sales Volume Condition'] = (
    df_clean['Quantity Sold']
    >
    df_clean['30-Day Average Quantity Sold']
)


# A promotion day must satisfy BOTH conditions

df_clean['Promotion Day'] = (
    df_clean['Low Price Condition']
    &
    df_clean['High Sales Volume Condition']
)


# Group consecutive promotion days into promotion periods


# Create a new group whenever the promotion status changes
df_clean['Promotion Group'] = (
    df_clean['Promotion Day']
    != df_clean['Promotion Day'].shift()
).cumsum()

promotion_periods = []


# Summarize each promotion period
for _, group in df_clean[df_clean['Promotion Day']].groupby('Promotion Group'):

    # Ignore one-day events because they are likely random fluctuations
    if len(group) < 2:
        continue

    promotion_periods.append({

        'Promotion Start Date': group['Date'].min(),
        'Promotion End Date': group['Date'].max(),
        'Promotion Length (Days)': len(group),

        'Average Selling Price During Promotion':
            group['Average Selling Price'].mean(),

        'Average Quantity Sold During Promotion':
            group['Quantity Sold'].mean(),

        'Average Gross Profit During Promotion':
            group['Gross Profit'].mean(),

        'Average Gross Profit Margin (%)':
            group['Gross Profit Margin (%)'].mean()

    })

# Convert the results into a DataFrame
promotion_summary = pd.DataFrame(promotion_periods)

# Show the longest promotion periods first
promotion_summary = (
    promotion_summary
    .sort_values(
        by='Promotion Length (Days)',
        ascending=False
    )
    .reset_index(drop=True)
)

# Only choose the top 3 promo periods
promotion_summary = promotion_summary.head(3)

display(promotion_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### PRICE ELASTICITY OF DEMAND
# MAGIC - Price Elasticity of Demand measures how much customer demand changes when the selling price changes.
# MAGIC
# MAGIC - Formula:
# MAGIC     - Price Elasticity of Demand = (% Change in Quantity Sold) / (% Change in Selling Price)
# MAGIC
# MAGIC - For each promotion period:
# MAGIC     - Compare the promotion to the 14 days before it started.
# MAGIC     - The previous 14 days are treated as the normal (baseline) period.

# COMMAND ----------

elasticity_results = []

# Calculate elasticity for each promotion period
for _, promotion in promotion_summary.iterrows():

    # Define the 14-day baseline period before the promotion
    baseline_start = (
        promotion['Promotion Start Date'] - pd.Timedelta(days=14)
    )
    baseline = df_clean[
        (df_clean['Date'] >= baseline_start) &
        (df_clean['Date'] < promotion['Promotion Start Date'])
    ]

    # Skip if there is no baseline data
    if baseline.empty:
        continue

    # Calculate average values during the baseline period
   
    baseline_price = baseline['Average Selling Price'].mean()
    baseline_quantity = baseline['Quantity Sold'].mean()
    baseline_profit_margin = baseline['Gross Profit Margin (%)'].mean()


    # Calculate percentage change during the promotion

    percentage_change_in_price = (
        (
            promotion['Average Selling Price During Promotion'] 
            - baseline_price
        )/ baseline_price
    ) * 100 if baseline_price != 0 else np.nan

    percentage_change_in_quantity = (
        (
            promotion['Average Quantity Sold During Promotion']
            - baseline_quantity
        )
        / baseline_quantity
    ) * 100 if baseline_quantity != 0 else np.nan


    # Calculate Price Elasticity of Demand
   

    if percentage_change_in_price != 0 and not np.isnan(percentage_change_in_price):
        price_elasticity = (
            percentage_change_in_quantity
            / percentage_change_in_price
        )
    else:
        price_elasticity = np.nan


    # Store the results
   

    elasticity_results.append({

        'Promotion Start Date':
            promotion['Promotion Start Date'].date(),

        'Promotion End Date':
            promotion['Promotion End Date'].date(),

        'Promotion Length (Days)':
            promotion['Promotion Length (Days)'],

        'Baseline Average Selling Price':
            round(baseline_price, 2) if not np.isnan(baseline_price) else np.nan,

        'Promotion Average Selling Price':
            round(
                promotion['Average Selling Price During Promotion'],
                2
            ),

        'Percentage Change in Selling Price':
            round(percentage_change_in_price, 1) if not np.isnan(percentage_change_in_price) else np.nan,

        'Baseline Average Quantity Sold':
            round(baseline_quantity, 0) if not np.isnan(baseline_quantity) else np.nan,

        'Promotion Average Quantity Sold':
            round(
                promotion['Average Quantity Sold During Promotion'],
                0
            ),

        'Percentage Change in Quantity Sold':
            round(percentage_change_in_quantity, 1) if not np.isnan(percentage_change_in_quantity) else np.nan,

        'Price Elasticity of Demand':
            round(price_elasticity, 2) if not np.isnan(price_elasticity) else np.nan,

        'Baseline Gross Profit Margin (%)':
            round(baseline_profit_margin, 1) if not np.isnan(baseline_profit_margin) else np.nan,

        'Promotion Gross Profit Margin (%)':
            round(
                promotion['Average Gross Profit Margin (%)'],
                1
            ),

        'Change in Gross Profit Margin (%)':
            round(
                promotion['Average Gross Profit Margin (%)']
                - baseline_profit_margin,
                1
            ) if not np.isnan(baseline_profit_margin) else np.nan
    })

# Convert the results into a DataFrame
price_elasticity_summary = pd.DataFrame(elasticity_results)

# Display the results
display(price_elasticity_summary)

# COMMAND ----------


fig = px.bar(
    price_elasticity_summary,
    x="Promotion Start Date",
    y="Price Elasticity of Demand",
    title="Price Elasticity of Demand for Promotion Periods",
    labels={
        "Promotion Start Date": "Promotion Start Date",
        "Price Elasticity of Demand": "Price Elasticity"
    },
    text="Price Elasticity of Demand"
)
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.show()

# COMMAND ----------

df_clean.to_csv('Sales_Processed.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 5 Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Univariated Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Key Metrics Summary

# COMMAND ----------

# Calculate total sales (revenue)
total_sales = round(df_clean['Sales'].sum(), 2)

# Calculate total profit
total_profit = round(df_clean['Gross Profit'].sum(), 2)

# Calculate total quantity sold
total_quantity_sold = round(df_clean['Quantity Sold'].sum(), 2)

# Date range
date_range = f"{df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}"

# Total trading days
total_trading_days = df_clean['Date'].nunique()

# Daily sales price per unit
df_clean['Daily Sales Price Per Unit'] = df_clean['Sales'] / df_clean['Quantity Sold']

# Average unit sales price of this product (R)
average_unit_sales_price = round(df_clean['Daily Sales Price Per Unit'].mean(), 2)

# Average daily sales (R)
average_daily_sales = round(df_clean['Sales'].mean(), 2)

# Average daily quantity sold
average_daily_quantity_sold = round(df_clean['Quantity Sold'].mean(), 2)

# Average daily % gross profit
average_daily_gross_profit_pct = round(df_clean['Gross Profit Margin (%)'].mean(), 2)

# Average daily % gross profit per unit
average_daily_gross_profit_per_unit = round(
    (df_clean['Gross Profit'] / df_clean['Quantity Sold']).mean(), 2)

# Days with negative gross profit
days_negative_gross_profit = (df_clean['Gross Profit'] < 0).sum()

# % of days with negative gross profit
pct_days_negative_gross_profit = round(
    100 * days_negative_gross_profit / total_trading_days, 2
)

# Best single day (sales, R)
best_single_day_sales = round(df_clean['Sales'].max(), 2)

# Worst single day (sales, R)
worst_single_day_sales = round(df_clean['Sales'].min(), 2)

# Highest quantity sold (day)
highest_quantity_sold = int(df_clean['Quantity Sold'].max())

# Number of promo periods detected
num_promo_periods = promotion_summary.shape[0] if 'promotion_summary' in locals() else 0

# Display results
display(pd.DataFrame({
    "Date Range": [date_range],
    "Total Trading Days": [total_trading_days],
    "Total Sales (Revenue)": [total_sales],
    "Total Profit": [total_profit],
    "Total Quantity Sold": [total_quantity_sold],
    "Average Unit Sales Price (R)": [average_unit_sales_price],
    "Average Daily Sales (R)": [average_daily_sales],
    "Average Daily Quantity Sold": [average_daily_quantity_sold],
    "Average Daily % Gross Profit": [average_daily_gross_profit_pct],
    "Average Daily % Gross Profit Per Unit": [average_daily_gross_profit_per_unit],
    "Days with Negative Gross Profit": [days_negative_gross_profit],
    "% of Days with Negative Gross Profit": [pct_days_negative_gross_profit],
    "Best Single Day (Sales, R)": [best_single_day_sales],
    "Worst Single Day (Sales, R)": [worst_single_day_sales],
    "Highest Quantity Sold (Day)": [highest_quantity_sold],
    "Number of Promo Periods Detected": [num_promo_periods]
}))

# COMMAND ----------

# MAGIC %md
# MAGIC - MONTHLY SALES PERFORMANCE & Month-over-Month(MoM) sales growth
# MAGIC     - Determining the average daily sales, average daily quantity sold, and average gross profit margin for each month.
# MAGIC

# COMMAND ----------

monthly_summary = (
    df_clean.groupby("Sales Month Name")
    .agg(
        **{
            "Total Quantity Sold": ("Quantity Sold", "sum"),
            "Total Sales": ("Sales", "sum"),
            "Total Cost": ("Cost Of Sales", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Gross Profit Margin (%)": ("Gross Profit Margin (%)","mean")
        }
    )
    .reindex([
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    .reset_index()
    .round(2)
)

# Calculate month-over-month sales growth
monthly_summary["Month-over-Month Sales Growth (%)"] = (
    monthly_summary["Total Sales"]
    .pct_change()
    * 100
).round(2)

monthly_summary["Month-over-Month Quantity Growth (%)"] = (
    monthly_summary["Total Quantity Sold"]
    .pct_change()
    * 100
).round(2)

# Display the monthly summary
display(monthly_summary)

# COMMAND ----------

fig = px.line(
    monthly_summary,
    x="Sales Month Name",
    y=["Total Quantity Sold", "Total Sales", "Total Cost", "Gross Profit"],
    title="Monthly Metrics: Quantity Sold, Sales, Cost, and Gross Profit",
    labels={
        "Sales Month Name": "Month",
        "value": "Metric Value",
        "variable": "Metric"
    }
)
fig.update_traces(mode='lines+markers')
fig.update_layout(legend_title_text='Metric')
fig.show()

# Monthly Average Gross Profit Margin and MoM Growth
fig_bar = px.bar(
    monthly_summary,
    x="Sales Month Name",
    y=["Average Gross Profit Margin (%)", "Month-over-Month Sales Growth (%)", "Month-over-Month Quantity Growth (%)"],
    title="Monthly Average Gross Profit Margin and MoM Growth",
    labels={
        "Sales Month Name": "Month",
        "value": "Metric Value",
        "variable": "Metric"
    },
    barmode="group"
)
fig_bar.update_traces(texttemplate='%{value:.2f}', textposition='outside')
fig_bar.update_layout(legend_title_text='Metric')
fig_bar.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### YEARLY SALES PERFORMANCE & YEAR-OVER-YEAR (YoY) GROWTH
# MAGIC - Determining total sales, total quantity sold, and the average gross profit margin for each year.

# COMMAND ----------

yearly_summary = (
    df_clean.groupby("Year of Sales")
    .agg(
        **{
            "Total Quantity Sold": ("Quantity Sold", "sum"),
            "Total Sales": ("Sales", "sum"),
            "Total Cost": ("Cost Of Sales", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Gross Profit Margin (%)": ("Gross Profit Margin (%)","mean")
        }
    )
    .reset_index()
    .round(2)
)


# Calculate how much total sales changed compared with the previous year.
yearly_summary["Year-over-Year Sales Growth (%)"] = (
    yearly_summary["Total Sales"]
    .pct_change()
    * 100
)

yearly_summary["Year-over-Year Quantity Growth (%)"] = (
    yearly_summary["Total Quantity Sold"]
    .pct_change()
    * 100
)

yearly_summary = yearly_summary.round(2)
display(yearly_summary)

# COMMAND ----------

fig = px.bar(
    yearly_summary,
    y="Year of Sales",
    x=["Total Quantity Sold", "Total Sales", "Total Cost", "Gross Profit"],
    orientation='h',
    title="Yearly Trends: Quantity Sold, Sales, Cost, and Gross Profit",
    labels={
        "Year of Sales": "Year",
        "value": "Metric Value",
        "variable": "Metric"
    }
)
fig.update_traces(texttemplate='%{value:.2f}', textposition='outside')
fig.update_layout(legend_title_text='Metric')
fig.show()

# Yearly Average Gross Profit Margin and YoY Growth
fig_bar = px.bar(
    yearly_summary,
    x="Year of Sales",
    y=["Average Gross Profit Margin (%)", "Year-over-Year Sales Growth (%)", "Year-over-Year Quantity Growth (%)"],
    title="Yearly Average Gross Profit Margin and YoY Growth",
    labels={
        "Year of Sales": "Year",
        "value": "Metric Value",
        "variable": "Metric"
    },
    barmode="group"
)
fig_bar.update_traces(texttemplate='%{value:.2f}', textposition='outside')
fig_bar.update_layout(legend_title_text='Metric')
fig_bar.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - 2015 was the worst performing Year and May was the worst Month
# MAGIC - july had a minimal loss making it a better month than the rest and 2013

# COMMAND ----------

# MAGIC %md
# MAGIC ##### MONTHLY PERFORMANCE ANALYSIS FOR EACH YEAR
# MAGIC - This analysis calculates monthly performance separately for:
# MAGIC - 2013, 2014, 2015, and 2016.
# MAGIC
# MAGIC The metrics include:
# MAGIC - Total Sales
# MAGIC - Total Quantity Sold
# MAGIC - Total Gross Profit
# MAGIC - Average Selling Price
# MAGIC - Average Gross Profit Margin
# MAGIC

# COMMAND ----------

# Define the correct calendar order for the months
month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]

# Group the data by year and month
monthly_performance_by_year = (
    df_clean.groupby(
        [
            "Year of Sales",
            "Sales Month Name"
        ]
    )
    .agg(
        **{
            # Calculate the total sales for each month
            "Total Sales": ("Sales","sum"),

            # Calculate the total quantity sold for each month
            "Total Quantity Sold": ("Quantity Sold","sum"),

            # Calculate the total gross profit for each month
            "Total Gross Profit": ("Gross Profit","sum"),

            # Calculate the average selling price for each month
            "Average Selling Price": ("Average Selling Price","mean"),

            # Calculate the average gross profit margin for each month
            "Average Gross Profit Margin (%)": ("Gross Profit Margin (%)","mean" )
        }
    )
    .reset_index()
)

# Arrange the months in the correct calendar order
monthly_performance_by_year[
    "Sales Month Name"
] = pd.Categorical(
    monthly_performance_by_year[
        "Sales Month Name"
    ],
    categories=month_order,
    ordered=True
)

# Sort the results by year and then by month
monthly_performance_by_year = (
    monthly_performance_by_year
    .sort_values(
        [
            "Year of Sales",
            "Sales Month Name"
        ]
    )
    .reset_index(drop=True)
)

# Round numerical values to two decimal places
monthly_performance_by_year = (
    monthly_performance_by_year
    .round(2)
)

# Display the complete monthly performance table
display(monthly_performance_by_year)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### MONTHLY PERFORMANCE DASHBOARD FOR EACH YEAR
# MAGIC

# COMMAND ----------


# Define the correct calendar order for the months
month_order = ["January","February","March","April","May","June",
    "July","August","September","October","November","December"
]


# Loop through every year in the monthly performance dataset
for year in sorted(
    monthly_performance_by_year["Year of Sales"].dropna().unique()
):

    # Filter the monthly data for the selected year
    df_year = (
        monthly_performance_by_year[
            monthly_performance_by_year["Year of Sales"] == year
        ]
        .copy()
    )

    # Arrange the months in calendar order
    df_year["Sales Month Name"] = pd.Categorical(
        df_year["Sales Month Name"],
        categories=month_order,
        ordered=True
    )

    df_year = (
        df_year
        .sort_values("Sales Month Name")
        .reset_index(drop=True)
    )


    fig = go.Figure()


 
    # Add total sales for each month
    fig.add_trace(
        go.Bar(
            x=df_year["Sales Month Name"],
            y=df_year["Total Sales"],
            name="Total Sales",
            marker_color="#1f77b4",
            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Total Sales:</b> R%{y:,.2f}"
                "<extra></extra>"
            )
        )
    )


    # Add total quantity sold for each month
    fig.add_trace(
        go.Bar(
            x=df_year["Sales Month Name"],
            y=df_year["Total Quantity Sold"],
            name="Total Quantity Sold",
            marker_color="#2ca02c",
            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Total Quantity Sold:</b> %{y:,.0f}"
                "<extra></extra>"
            )
        )
    )


    # Add total gross profit for each month
    fig.add_trace(
        go.Bar(
            x=df_year["Sales Month Name"],
            y=df_year["Total Gross Profit"],
            name="Total Gross Profit",
            marker_color="#d62728",
            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Total Gross Profit:</b> R%{y:,.2f}"
                "<extra></extra>"
            )
        )
    )



    # Add the average selling price line
    fig.add_trace(
        go.Scatter(
            x=df_year["Sales Month Name"],
            y=df_year["Average Selling Price"],
            name="Average Selling Price",
            mode="lines+markers",

            line=dict(
                color="#ff7f0e",
                width=3
            ),

            marker=dict(
                size=7
            ),

            yaxis="y2",

            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Average Selling Price:</b> R%{y:,.2f}"
                "<extra></extra>"
            )
        )
    )


    # Add the average gross profit margin line
    fig.add_trace(
        go.Scatter(
            x=df_year["Sales Month Name"],
            y=df_year["Average Gross Profit Margin (%)"],
            name="Average Gross Profit Margin",

            mode="lines+markers",

            line=dict(
                color="#9467bd",
                width=3,
                dash="dot"
            ),

            marker=dict(
                size=7
            ),

            yaxis="y3",

            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Average Gross Profit Margin:</b> %{y:.2f}%"
                "<extra></extra>"
            )
        )
    )


    # FORMAT THE CHART
  

    fig.update_layout(

        title={
            "text": (
                f"Monthly Sales and Profitability Performance — {year}"
            ),
            "x": 0.5,
            "xanchor": "center"
        },

        # Keep the months in calendar order
        xaxis=dict(
            title="Month",
            categoryorder="array",
            categoryarray=month_order
        ),

        # Primary y-axis for the bar charts
        yaxis=dict(
            title="Total Sales / Quantity Sold / Gross Profit"
        ),

        # Second y-axis for average selling price
        yaxis2=dict(
            title="Average Selling Price (R)",
            overlaying="y",
            side="right",
            showgrid=False
        ),

        # Third y-axis for gross profit margin
        yaxis3=dict(
            title="Average Gross Profit Margin (%)",
            overlaying="y",
            side="right",

            # Position must be between 0 and 1
            anchor="free",
            position=0.92,

            showgrid=False,
            ticksuffix="%"
        ),

        # Display the three bar charts next to each other
        barmode="group",

        # Display the legend above the chart
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5
        ),

        template="plotly_white",

        hovermode="x unified",

        width=1250,
        height=600,

        # Increase the right margin to make room for the metrics summary
        margin=dict(
            l=80,
            r=250,
            t=120,
            b=70
        )
    )


   
    # CREATE THE YEARLY PERFORMANCE SUMMARY
   

    metrics_text = (
        f"<b>YEAR {int(year)} SUMMARY</b><br><br>"

        f"<b>Total Sales:</b><br>"
        f"R{df_year['Total Sales'].sum():,.2f}<br><br>"

        f"<b>Total Quantity Sold:</b><br>"
        f"{df_year['Total Quantity Sold'].sum():,.0f}<br><br>"

        f"<b>Total Gross Profit:</b><br>"
        f"R{df_year['Total Gross Profit'].sum():,.2f}<br><br>"

        f"<b>Average Selling Price:</b><br>"
        f"R{df_year['Average Selling Price'].mean():,.2f}<br><br>"

        f"<b>Average Gross Profit Margin:</b><br>"
        f"{df_year['Average Gross Profit Margin (%)'].mean():.2f}%"
    )


    # ADD THE YEARLY PERFORMANCE SUMMARY TO THE RIGHT SIDE
    

    fig.add_annotation(

        text=metrics_text,

        xref="paper",
        yref="paper",

        # Place the summary outside the chart area
        x=1.03,
        y=0.50,

        xanchor="left",
        yanchor="middle",

        showarrow=False,

        align="left",

        font=dict(
            size=12
        ),

        bordercolor="#CCCCCC",

        borderwidth=1,

        borderpad=12,

        bgcolor="white",

        opacity=0.95
    )


    # Display the interactive chart
    fig.show()

# COMMAND ----------

import plotly.express as px

# ---------------------------------------------------------------------------
# MONTHLY SALES PERFORMANCE BY YEAR
# ---------------------------------------------------------------------------

fig = px.line(
    monthly_performance_by_year,
    x="Sales Month Name",
    y="Total Sales",
    color="Year of Sales",
    markers=True,

    title=(
        "Monthly Sales Performance by Year"
        "<br>"
        "<sup>Comparison of 2013, 2014, 2015, and 2016</sup>"
    ),

    labels={
        "Sales Month Name": "Month",
        "Total Sales": "Total Sales (R)",
        "Year of Sales": "Year"
    },

    category_orders={
        "Sales Month Name": month_order
    },

    template="plotly_white"
)


# Format the chart
fig.update_layout(
    title={
        "x": 0.5,
        "xanchor": "center"
    },

    xaxis_title="Month",

    yaxis_title="Total Sales (R)",

    yaxis=dict(
        tickprefix="R",
        tickformat=","
    ),

    hovermode="x unified",

    width=1200,
    height=600
)


# Display the interactive chart
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### DAILY PERFORMANCE

# COMMAND ----------

daily_summary = (
    df_clean.groupby("Sales Day Name")
    .agg(
        **{
            "Total Quantity Sold": ("Quantity Sold", "sum"),
            "Total Sales": ("Sales", "sum"),
            "Total Cost": ("Cost Of Sales", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Gross Profit Margin (%)": ("Gross Profit Margin (%)", "mean"),
            "Average Daily Sales Price Per Unit": ("Daily Sales Price Per Unit", "mean"),
            "Average Gross Profit Margin (%)": ("Gross Profit Margin (%)","mean")
        }
    )
    .reindex([
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])
    .reset_index()
    .round(2)
)

# Calculate average gross profit per unit for each day
daily_summary["Average Gross Profit Per Unit"] = (
    daily_summary["Gross Profit"] / daily_summary["Total Quantity Sold"]
).round(2)

# Calculate day-over-day sales growth
daily_summary["Day-over-Day Sales Growth (%)"] = (
    daily_summary["Total Sales"]
    .pct_change()
    * 100
).round(2)

daily_summary["Day-over-Day Quantity Growth (%)"] = (
    daily_summary["Total Quantity Sold"]
    .pct_change()
    * 100
).round(2)

display(daily_summary)

# COMMAND ----------

fig = px.bar(
    daily_summary,
    y="Sales Day Name",
    x=["Total Quantity Sold", "Total Sales", "Total Cost", "Gross Profit"],
    orientation='h',
    title="Daily Trends: Quantity Sold, Sales, Cost, and Gross Profit",
    labels={
        "Sales Day Name": "Day",
        "value": "Metric Value",
        "variable": "Metric"
    }
)
fig.update_traces(texttemplate='%{value:.2f}', textposition='outside')
fig.update_layout(legend_title_text='Metric')
fig.show()

# Daily Average Gross Profit Margin and Day-over-Day Growth
fig_bar = px.bar(
    daily_summary,
    x="Sales Day Name",
    y=["Average Gross Profit Margin (%)", "Day-over-Day Sales Growth (%)", "Day-over-Day Quantity Growth (%)"],
    title="Daily Average Gross Profit Margin and Day-over-Day Growth",
    labels={
        "Sales Day Name": "Day",
        "value": "Metric Value",
        "variable": "Metric"
    },
    barmode="group"
)
fig_bar.update_traces(texttemplate='%{value:.2f}', textposition='outside')
fig_bar.update_layout(legend_title_text='Metric')
fig_bar.show()

# COMMAND ----------

fig = px.line(
    df_clean,
    x="Date",
    y=["Quantity Sold", "Sales", "Cost Of Sales", "Gross Profit"],
    title="Daily Metrics: Quantity Sold, Sales, Cost, and Gross Profit",
    labels={
        "Sales Date": "Date",
        "value": "Metric Value",
        "variable": "Metric"
    }
)
fig.update_traces(mode='lines+markers')
fig.update_layout(legend_title_text='Metric')
fig.show()

# COMMAND ----------

fig = px.pie(
    daily_summary,
    names="Sales Day Name",
    values="Total Sales",
    title="Daily Sales Distribution by Day",
    labels={"Sales Day Name": "Day", "Total Sales": "Sales"}
)
fig.show()

# COMMAND ----------

weekly_summary = (
    df_clean.groupby("Weekly Sales")
    .agg(
        **{
            "Total Quantity Sold": ("Quantity Sold", "sum"),
            "Total Sales": ("Sales", "sum"),
            "Total Cost": ("Cost Of Sales", "sum"),
            "Gross Profit": ("Gross Profit", "sum"),
            "Average Gross Profit Margin (%)": ("Gross Profit Margin (%)", "mean")
        }
    )
    .reset_index()
    .round(2)
)

display(weekly_summary)

# COMMAND ----------

fig = px.pie(
    weekly_summary,
    names="Weekly Sales",
    values="Total Sales",
    title="Weekly Sales Distribution",
    labels={"Weekly Sales": "Week", "Total Sales": "Sales"}
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### AVERAGE SELLING PRICE AND GROSS PROFIT MARGIN OVER TIME
# MAGIC
# MAGIC - This chart compares:
# MAGIC 1. The 7-day average selling price
# MAGIC 2. The 7-day average gross profit margin
# MAGIC - Orange shaded areas represent detected promotion periods.
# MAGIC
# MAGIC

# COMMAND ----------


# Create the interactive Plotly figure
fig = go.Figure()

# Add the 7-day average selling price
# This line uses the left y-axis because the values are measured in Rand.


fig.add_trace(
    go.Scatter(
        x=df_clean["Date"],
        y=df_clean["7-Day Average Selling Price"],
        mode="lines",
        name="7-Day Average Selling Price",
        line=dict(
            color="#1f77b4",
            width=2
        ),
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y}<br>"
            "<b>7-Day Average Selling Price:</b> R%{y:,.2f}"
            "<extra></extra>"
        )
    )
)

# Add the 7-day average gross profit margin
# This line uses the right y-axis because the values are percentages.


fig.add_trace(
    go.Scatter(
        x=df_clean["Date"],
        y=df_clean["7-Day Average Gross Profit Margin (%)"],
        mode="lines",
        name="7-Day Average Gross Profit Margin",
        line=dict(
            color="#d62728",
            width=2
        ),
        opacity=0.8,
        yaxis="y2",
        hovertemplate=(
            "<b>Date:</b> %{x|%d %b %Y}<br>"
            "<b>7-Day Average Gross Profit Margin:</b> %{y:.2f}%"
            "<extra></extra>"
        )
    )
)

# Add orange shading to show detected promotion periods

for _, promotion in promotion_summary.iterrows():

    fig.add_vrect(
        x0=promotion["Promotion Start Date"],
        x1=promotion["Promotion End Date"],
        fillcolor="orange",
        opacity=0.20,
        line_width=0,
        layer="below"
    )


# Format the chart

fig.update_layout(

    title={
        "text": (
            "Average Selling Price and Gross Profit Margin Over Time"
            "<br>"
            "<sup>Orange shaded areas represent detected promotion periods</sup>"
        ),
        "x": 0.5,
        "xanchor": "center"
    },

    xaxis=dict(
        title="Date",
        tickformat="%Y",
        dtick="M12"
    ),

    # Left y-axis: Average Selling Price
    yaxis=dict(
        title="Average Selling Price (R)",
        tickprefix="R",
        tickformat=",.0f"
    ),

    # Right y-axis: Gross Profit Margin
    yaxis2=dict(
        title="Gross Profit Margin (%)",
        overlaying="y",
        side="right",
        ticksuffix="%"
    ),

    template="plotly_white",

    hovermode="x unified",

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="center",
        x=0.5
    ),

    width=800,
    height=400,

    margin=dict(
        l=80,
        r=80,
        t=120,
        b=70
    )
)


# Display the interactive chart
fig.show()

# COMMAND ----------

fig = px.scatter(
    df_clean,
    x="Daily Sales Price Per Unit",
    y="Gross Profit Margin (%)",
    color="Promotion Day",
    hover_data=[
        "Date",
        "Sales",
        "Quantity Sold",
        "Gross Profit"
    ],
    opacity=0.6,
    title=(
        "Average Selling Price vs Gross Profit Margin "
        "by Promotion Status"
    ),
    labels={
        "Daily Sales Price Per Unit": "Average Selling Price (R)",
        "Gross Profit Margin (%)": "Gross Profit Margin (%)",
        "Promotion Day": "Promotion Day"
    },
    template="plotly_white"
)

fig.show()