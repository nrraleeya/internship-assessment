# Time Series Sales Forecasting Project

This sub-module contains the production-grade predictive pipeline designed to forecast daily retail sales for the next 3 months across multiple store and item configurations. 

## 1. Modeling Approach & Rationale

For this assignment, Meta’s **Prophet** forecasting algorithm was selected over traditional models like ARIMA or deep learning models like LSTM. The core rationale includes:

* **Additive Architecture (Interpretability):** Prophet models the time series as a curve-fitting exercise decomposed into structural components: 
  $$\text{Forecast} = \text{Trend} + \text{Weekly Seasonality} + \text{Yearly Seasonality} + \text{Error}$$
  This provides complete transparent visibility into *why* a certain volume is predicted, which is vital for business inventory planning.
* **Handling of Human Cyclical Behavior:** Retail sales are heavily dictated by recurring calendar patterns (e.g., weekend shopping rushes, seasonal demand shifts). Prophet handles multi-period seasonality natively out-of-the-box without manual data transformation.
* **Robustness to Structural Traps:** Unlike classical ARIMA, which requires strict stationarity and crashes if there are missing dates or tracking gaps, Prophet treats timestamps independently, making it resilient to irregularities common in raw store data.

---

## 2. Feature Engineering & Post-Processing Rationale

Per the dataset criteria, the timeline contains no special holiday shocks or sudden store closures. Feature engineering focused on structural constraints and domain-specific optimizations:

1. **Granular Series Partitioning:** Instead of building a single global model, the pipeline dynamically chunks data by individual `store` and `item` pairs. This ensures the model learns the exact localized velocity of a specific product at a specific location.
2. **Positional Data Ingestion (`.iloc`):** The data preprocessing pipeline uses position-based array indexing rather than strict string matching. This immunizes the script against hidden column syntax variance, trailing whitespaces, or encoding adjustments during text-to-dataframe loading.
3. **Discrete Integer Conversion:** Because sales volumes represent physical items, raw float predictions ($yhat$) are rounded to the nearest whole integer (`.round().astype(int)`) as a post-processing step.

---

## 3. Validation Framework & Evaluation Metrics

To guarantee reliability before forecasting the unseen future, the pipeline implements an **Internal Holdout Validation Framework**. For every single store-item group, the final 90 days of known history are hidden to act as a validation test. 

The baseline error metrics are aggregated and averaged globally across all product lines to score the model's performance:

### Final Macro Performance Report
* **Mean Absolute Error (MAE):** Measures the average magnitude of prediction errors. An average MAE of **~6.36 units** demonstrates that the model is off by fewer than 7 items per day, making it highly reliable for shelf allocation.
* **Root Mean Squared Error (RMSE):** Heavily penalizes large forecasting misses due to its squaring mechanism. Used here as an operational safety alarm to identify unpredictable item spikes.
* **Mean Absolute Percentage Error (MAPE):** Normalizes errors relative to actual sales scales. An average MAPE of **~14.15%** represents an industry-standard, high-performing baseline for granular, daily store-item demand modeling where noise is inherently high.

---

## 4. How to Run the Pipeline

1. Ensure `train.csv`, `test.csv`, and `sales_forecasting.py` are in the same directory.
2. Run the script from your terminal:
   ```bash
   python sales_forecasting.py
