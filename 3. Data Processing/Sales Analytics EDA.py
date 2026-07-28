# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Sales Analytics EDA
# MAGIC Full metrics & KPI build
# MAGIC
# MAGIC Reads the daily sales dataset and computes:
# MAGIC   1. Daily sales price per unit
# MAGIC   2. Average unit sales price
# MAGIC   3. Daily % gross profit
# MAGIC   4. Daily % gross profit per unit
# MAGIC   5. Promotion detection + Price Elasticity of Demand for 3 promo periods
# MAGIC   6. Extra insights: rolling averages, day-of-week seasonality, monthly/yearly
# MAGIC      trends, YoY growth, negative-GP-day analysis, outlier detection

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
df_clean['Month Sales']           = df_clean['Date'].dt.month
df_clean['Sales Month Name']      = df_clean['Date'].dt.month_name()
df_clean['Sales Day Name']        = df_clean['Date'].dt.day_name()
df_clean['Sales Quarter']         = df_clean['Date'].dt.quarter
df_clean['Weekly Sales']          = df_clean['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')


# COMMAND ----------

# Seasonal Feature (Southern Hemisphere: Summer=Dec-Feb, Autumn=Mar-May, Winter=Jun-Aug, Spring=Sep-Nov)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    else:
        return 'Spring'

df_clean['Sales Season'] = df_clean['Month Sales'].apply(get_season)


# COMMAND ----------


# Preview new columns only
print("Date-based columns created:")
display(df_clean[['Year of Sales','Month Sales','Sales Month Name',
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
df_clean["30-Day Average Quantity Sold"] = (
    df_clean["Quantity Sold"]
    .rolling(window=30, min_periods=1)
    .mean()
)


# Condition 1:
# The selling price is unusually lower than normal

df_clean["Low Price Condition"] = (
    df_clean["Average Selling Price"]
    <
    (
        df_clean["30-Day Average Selling Price"]
        - df_clean["30-Day Selling Price Variation"]
    )
)


# Condition 2:
# The quantity sold is higher than normal

df_clean["High Sales Volume Condition"] = (
    df_clean["Quantity Sold"]
    >
    df_clean["30-Day Average Quantity Sold"]
)


# A promotion day must satisfy BOTH conditions

df_clean["Promotion Day"] = (
    df_clean["Low Price Condition"]
    &
    df_clean["High Sales Volume Condition"]
)


# Group consecutive promotion days into promotion periods


# Create a new group whenever the promotion status changes
df_clean["Promotion Group"] = (
    df_clean["Promotion Day"]
    != df_clean["Promotion Day"].shift()
).cumsum()

promotion_periods = []


# Summarize each promotion period
for _, group in df_clean[df_clean["Promotion Day"]].groupby("Promotion Group"):

    # Ignore one-day events because they are likely random fluctuations
    if len(group) < 2:
        continue

    promotion_periods.append({

        "Promotion Start Date": group["Date"].min(),
        "Promotion End Date": group["Date"].max(),
        "Promotion Length (Days)": len(group),

        "Average Selling Price During Promotion":
            group["Average Selling Price"].mean(),

        "Average Quantity Sold During Promotion":
            group["Quantity Sold"].mean(),

        "Average Gross Profit During Promotion":
            group["Gross Profit"].mean(),

        "Average Gross Profit Margin (%)":
            group["Gross Profit Margin (%)"].mean()

    })

# Convert the results into a DataFrame
promotion_summary = pd.DataFrame(promotion_periods)

# Show the longest promotion periods first
promotion_summary = (
    promotion_summary
    .sort_values(
        by="Promotion Length (Days)",
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
        promotion["Promotion Start Date"] - pd.Timedelta(days=14)
    )
    baseline = df_clean[
        (df_clean["Date"] >= baseline_start) &
        (df_clean["Date"] < promotion["Promotion Start Date"])
    ]

    # Skip if there is no baseline data
    if baseline.empty:
        continue

    # Calculate average values during the baseline period
   
    baseline_price = baseline["Average Selling Price"].mean()
    baseline_quantity = baseline["Quantity Sold"].mean()
    baseline_profit_margin = baseline["Gross Profit Margin (%)"].mean()


    # Calculate percentage change during the promotion

    percentage_change_in_price = (
        (
            promotion["Average Selling Price During Promotion"] 
            - baseline_price
        )/ baseline_price
    ) * 100 if baseline_price != 0 else np.nan

    percentage_change_in_quantity = (
        (
            promotion["Average Quantity Sold During Promotion"]
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

        "Promotion Start Date":
            promotion["Promotion Start Date"].date(),

        "Promotion End Date":
            promotion["Promotion End Date"].date(),

        "Promotion Length (Days)":
            promotion["Promotion Length (Days)"],

        "Baseline Average Selling Price":
            round(baseline_price, 2) if not np.isnan(baseline_price) else np.nan,

        "Promotion Average Selling Price":
            round(
                promotion["Average Selling Price During Promotion"],
                2
            ),

        "Percentage Change in Selling Price":
            round(percentage_change_in_price, 1) if not np.isnan(percentage_change_in_price) else np.nan,

        "Baseline Average Quantity Sold":
            round(baseline_quantity, 0) if not np.isnan(baseline_quantity) else np.nan,

        "Promotion Average Quantity Sold":
            round(
                promotion["Average Quantity Sold During Promotion"],
                0
            ),

        "Percentage Change in Quantity Sold":
            round(percentage_change_in_quantity, 1) if not np.isnan(percentage_change_in_quantity) else np.nan,

        "Price Elasticity of Demand":
            round(price_elasticity, 2) if not np.isnan(price_elasticity) else np.nan,

        "Baseline Gross Profit Margin (%)":
            round(baseline_profit_margin, 1) if not np.isnan(baseline_profit_margin) else np.nan,

        "Promotion Gross Profit Margin (%)":
            round(
                promotion["Average Gross Profit Margin (%)"],
                1
            ),

        "Change in Gross Profit Margin (%)":
            round(
                promotion["Average Gross Profit Margin (%)"]
                - baseline_profit_margin,
                1
            ) if not np.isnan(baseline_profit_margin) else np.nan
    })

# Convert the results into a DataFrame
price_elasticity_summary = pd.DataFrame(elasticity_results)

# Display the results
display(price_elasticity_summary)

# COMMAND ----------

display(df_clean.tail (5))


# COMMAND ----------

df_clean.to_csv('Sales_Processed.csv', index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 5 Visualization

# COMMAND ----------

