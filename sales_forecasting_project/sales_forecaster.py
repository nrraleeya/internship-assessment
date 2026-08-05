import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import logging

# mute background warning messages from prophet
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.ERROR)

# STEP 1: LOAD CSV FILES
print("1. Loading raw files...")
train_data = pd.read_csv('Technical Assessment/train.csv')
test_data = pd.read_csv('Technical Assessment/test.csv')

# clean up hidden spaces in column names
train_data.columns = train_data.columns.str.lower().str.strip()
test_data.columns = test_data.columns.str.lower().str.strip()

# ensure python reads columns as actual dates
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

test_data['sales_pred'] = np.nan

# STEP 2: FIND ALL UNIQUE COMBINATIONS
unique_groups = train_data.groupby(['store', 'item']).size().reset_index()[['store', 'item']]
print(f"2. Found {len(unique_groups)} store-item groups. Generating 3-month forecasts...")

# arrays to hold accuracy scores for final report
all_mae_scores = []
all_rmse_scores = []
all_mape_scores = []

# STEP 3: MASTER LOOP
for idx, row in unique_groups.iterrows():
    s_id, i_id = row['store'], row['item']
    
    # isolate history for specific item at specific store
    train_subset = train_data[(train_data['store'] == s_id) & (train_data['item'] == i_id)]
    
    # find exact future rows in test.csv that belong to specific item/store
    test_mask = (test_data['store'] == s_id) & (test_data['item'] == i_id)
    test_subset = test_data[test_mask]
    
    if test_subset.empty:
        continue
        
    # PHASE A: EVALUATION METRICS CALCULATION
    # split historical data
    cutoff_date = train_subset['date'].max() - pd.Timedelta(days=90)
    hist_train = train_subset[train_subset['date'] <= cutoff_date]
    val_test = train_subset[train_subset['date'] > cutoff_date]
    
    # fit validation model
    val_prophet_df = pd.DataFrame({'ds': hist_train.iloc[:, 0], 'y': hist_train.iloc[:, 3]})
    val_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    val_model.fit(val_prophet_df)
    
    # predict validation period
    val_dates = pd.DataFrame({'ds': val_test.iloc[:, 0]})
    val_forecast = val_model.predict(val_dates)
    
    # calculate performance scores
    actuals = val_test.iloc[:, 3].values
    preds = val_forecast['yhat'].values
    
    all_mae_scores.append(mean_absolute_error(actuals, preds))
    all_rmse_scores.append(np.sqrt(mean_squared_error(actuals, preds)))
    all_mape_scores.append(mean_absolute_percentage_error(actuals, preds) * 100)
    
    # PHASE B: FUTURE FORECAST
    prophet_df = pd.DataFrame({'ds': train_subset.iloc[:, 0], 'y': train_subset.iloc[:, 3]})
    final_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    final_model.fit(prophet_df)
    
    # predict the future 3 months
    future_dates = pd.DataFrame({'ds': test_subset.iloc[:, 1]})
    forecast = final_model.predict(future_dates)
    
    test_data.loc[test_mask, 'sales_pred'] = forecast['yhat'].round().values

    # print progress update every 10 groups
    if (idx + 1) % 10 == 0 or (idx + 1) == len(unique_groups):
        print(f"   Progress: {idx + 1}/{len(unique_groups)} groups completed...")

# STEP 4: PRINT EVALUATION METRICS REPORT
print("FINAL OVERALL ACCURACY REPORT")
print(f"Average MAE across all items:  {np.mean(all_mae_scores):.2f} units")
print(f"Average RMSE across all items: {np.mean(all_rmse_scores):.2f} units")
print(f"Average MAPE across all items: {np.mean(all_mape_scores):.2f}%")

# STEP 5: FORMATTING & EXPORTING
submission = pd.DataFrame({
    'id': test_data['id'].astype(int),
    'sales': test_data['sales_pred'].astype(int)
})

submission.to_csv('final_submission.csv', index=False)
print("Success! 'final_submission.csv' is generated.")