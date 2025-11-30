#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 12:20:18 2025

@author: alexandrelaurent
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 1- DATA COLLECTION

ticker1 = "EURUSD=X"
ticker2 = "EURMAD=X"
ticker3 = "EURVND=X"

# 1 year of data
start_date = "2023-01-01"
end_date   = "2025-01-01"   # tu ne l'utilises pas ici, mais ok

data1 = yf.download(ticker1, period="1y")
data2 = yf.download(ticker2, period="1y")
data3 = yf.download(ticker3, period="1y")

print("Aperçu data3 :")
print(data3.head())

# Reset index to merge
data1 = data1.reset_index()
data2 = data2.reset_index()
data3 = data3.reset_index()

# Merge on column Date (on met Date comme index pour chacun)
dftot = (
    data1.set_index("Date")
         .join(data2.set_index("Date"), rsuffix="_MAD")
         .join(data3.set_index("Date"), rsuffix="_VND")
)

print("\nAprès merge :")
print(dftot.head())

# Supp unuseful columns -> toutes les colonnes Volume
cols_to_drop = [c for c in dftot.columns if "Volume" in c]
dftot.drop(columns=cols_to_drop, inplace=True)

# handle missing days : créer un index date complet (jours fériés/week-ends compris)
full_range = pd.date_range(start=dftot.index.min(),
                           end=dftot.index.max(),
                           freq="D")

df = dftot.reindex(full_range)

print("\nAprès réindexation (avant remplissage) :")
print(df.head())

# Remplacer les NaN par la moyenne des 7 dernières valeurs (CORRIGÉ : mean())
df = df.fillna(df.rolling(7, min_periods=1).mean())

print("\nAprès remplissage rolling(7).mean() :")
print(df.head())

# 2- Core analytics

# ---- EXTRACTION DES PRIX ----
close_cols = [c for c in df.columns if "Close" in c]
close = df[close_cols]      # <<< indispensable

# ---- CORE ANALYTICS ----

# Rendements journaliers
returns = close.pct_change()

# Volatilité annualisée
vol30 = returns.rolling(30, min_periods=1).std() * np.sqrt(252)
vol90 = returns.rolling(90, min_periods=1).std() * np.sqrt(252)

# SMA 20 / 50
sma20 = close.rolling(20).mean()
sma50 = close.rolling(50).mean()

# ---- MOMENTUM SIMPLE ----
momentum20 = close / close.shift(20) - 1
print ("momentum20",momentum20)

# ========= FLATTEN COLUMN INDEX ============
# (Pour supprimer Price/Close/Ticker)

def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

close   = flatten(close)
returns = flatten(returns)
vol30   = flatten(vol30)
vol90   = flatten(vol90)
sma20   = flatten(sma20)
sma50   = flatten(sma50)
momentum20 = flatten(momentum20)

print("\nColonnes finales simples :")
print(close.columns)

# ======================
# 3. VISUALIZATION (matplotlib)
# ======================


# Au cas où les colonnes seraient MultiIndex (Price / Close / Ticker)
def get_simple_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        return df.columns.get_level_values(-1)
    return df.columns

pair_names = list(get_simple_columns(close))

# A) Price chart with 20-day and 50-day SMAs
for c in pair_names:
    plt.figure(figsize=(12, 5))
    plt.plot(close.index, close[c], label=f"{c} Close", color="black", linewidth=1)
    plt.plot(sma20.index, sma20[c], label="SMA 20", color="blue", linewidth=1)
    plt.plot(sma50.index, sma50[c], label="SMA 50", color="red", linewidth=1)
    plt.title(f"{c} - Price with 20d & 50d SMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()

# B) Rolling volatility plot (30d & 90d)
for c in pair_names:
    plt.figure(figsize=(12, 5))
    plt.plot(vol30.index, vol30[c], label="30-day Vol", linewidth=1)
    plt.plot(vol90.index, vol90[c], label="90-day Vol", linewidth=1)
    plt.title(f"{c} - Rolling Annualized Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

# C) Histogram of daily returns
for c in pair_names:
    plt.figure(figsize=(7, 4))
    plt.hist(returns[c].dropna(), bins=40, alpha=0.7)
    plt.title(f"{c} - Distribution of Daily Returns")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# D) Optional: scatter plot comparing volatilities of different pairs (last date)
plt.figure(figsize=(6, 6))

last_vol30 = vol30.iloc[-1]
last_vol90 = vol90.iloc[-1]

plt.scatter(last_vol30, last_vol90)

for c in pair_names:
    plt.annotate(c, (last_vol30[c], last_vol90[c]))

plt.xlabel("30-day Volatility")
plt.ylabel("90-day Volatility")
plt.title("Cross-Sectional FX Vol (last date)")
plt.tight_layout()
plt.show()











