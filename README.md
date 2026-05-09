# Benchmarking Zero-Shot Foundation Models for Time Series Forecasting

## Thesis Context

This repository was developed as part of my master's thesis in time series forecasting. It implements a standardized benchmarking framework for comparing zero-shot TimeGPT with SARIMA, XGBoost, and LSTM across financial, retail, and energy datasets. The benchmark is designed around a shared rolling-origin evaluation procedure to ensure fairness, consistency, and reproducibility across models.

## Thesis Project

This repository contains the code developed for my master's thesis on time series forecasting benchmarking. The project evaluates whether a zero-shot foundation model for time series forecasting, TimeGPT, can perform competitively against classical statistical, machine learning, and deep learning approaches under a fair and standardized rolling-origin benchmark.

The benchmark compares four forecasting models:

- SARIMA
- XGBoost
- LSTM
- TimeGPT

The study is conducted across multiple time series from three domains:

- Financial markets
- Retail sales
- Energy systems

The goal of this project is not to optimize a single model on a single dataset, but to build a consistent and reproducible benchmark that compares forecasting performance across datasets, domains, and forecast horizons.

## Reproducing the Results

To reproduce the results of this thesis project, follow the pipeline below:

1. **Download the raw datasets**  
   Download the original datasets from their respective public sources.

2. **Run the data cleaning scripts**  
   Execute the cleaning pipeline to standardize all retained datasets into a common format with:
   - `ds` for the timestamp column
   - `y` for the target column

   The cleaned files should be saved in the domain-specific folders under:

   ```bash
   datasets_cleaned/
       stock/
       sales/
       energy/

3. Excecute Benchmark_design_v3.py and than the other scripts

4. Final outputs are stored in:

outputs_v3/
    benchmark/
    sarima_v3/
    xgb_v3/
    lstm_v3/
    timegpt_v3/
    dm_tests_v3/
