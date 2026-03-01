# Enterprise-Demand-Forecasting-Inventory-Optimization-System
A production-style, end-to-end time series forecasting and inventory optimization pipeline designed to predict multi-store demand and reduce stockout events using hybrid statistical and deep learning models.
Overview

This project simulates a large-scale retail demand environment and builds an advanced forecasting ensemble combining:

•	Fourier-based additive time series decomposition

•	Custom-built NumPy LSTM (from scratch)

•	Ensemble weight optimization

•	Probabilistic inventory control using Newsvendor model

The system predicts future demand and dynamically optimizes reorder points to improve supply chain efficiency.

Architecture

1️⃣ Data Simulation Layer

10 stores × 730 days (146,000+ records)

Trend + weekly & annual seasonality

Holiday spikes & promotion effects

Realistic demand noise

Lead-time simulation

2️⃣ Prophet-Inspired Forecasting Model

Fourier basis (weekly: 4 harmonics, annual: 8 harmonics)

Piecewise linear trend with 25 changepoints

Holiday regressors

Ridge regression fitting

Normalization + residual variance handling

3️⃣ Custom LSTM (Built from Scratch)

Implemented fully in NumPy

Forget, Input, Output, Candidate gates

TBPTT backpropagation

Adam optimizer

Gradient clipping

No external deep learning frameworks used.

4️⃣ Ensemble Optimization

Grid-search weight tuning

Validation-based blending

Improves robustness over single-model forecasting

5️⃣ Inventory Optimization Engine

Newsvendor model

95% service level (z = 1.65)

Dynamic safety stock calculation

Lead-time demand distribution modeling

**Tech Stack**

Python

NumPy

SciPy

Time Series Decomposition

Optimization Algorithms

Statistical Modeling

Matplotlib
