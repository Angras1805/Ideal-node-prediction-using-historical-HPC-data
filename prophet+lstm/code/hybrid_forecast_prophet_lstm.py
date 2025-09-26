# ============================================================
# ✅ Final Robust Hybrid Forecast: Prophet + LSTM Residuals
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import os

# ============================================================
# 1. Load your .ods file (Ubuntu path)
# ============================================================

file_path = "/home/sim06/Desktop/internkr/check/check.ods"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

df = pd.read_excel(file_path, engine="odf")
df.columns = df.columns.str.strip()

# ============================================================
# 2. Robust timezone-safe parsing
# ============================================================

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ✅ Safely handle tz-aware timestamps
if df['Date'].dt.tz is not None:
    df['Date'] = df['Date'].dt.tz_convert(None)

df = df.dropna(subset=['Date']).sort_values('Date')
print("✅ Date range:", df['Date'].min(), "→", df['Date'].max())

# ============================================================
# 3. Prophet DataFrame with cap
# ============================================================

df_prophet = df[['Date', 'Available']].rename(columns={'Date': 'ds', 'Available': 'y'})
cap_value = df_prophet['y'].max() * 1.1
df_prophet['cap'] = cap_value

# ============================================================
# 4. Fit Prophet with overlap window
# ============================================================

model = Prophet(
    growth='logistic',
    daily_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.5,
    seasonality_prior_scale=10.0
)
model.fit(df_prophet)

# === Exact forecast range ===
start_date = pd.Timestamp('2025-02-28 00:10:01')
end_date   = pd.Timestamp('2025-03-07 14:11:23')

# ✅ Ensure overlap by using make_future_dataframe
periods = int((end_date - df_prophet['ds'].max()).total_seconds() / 3600)
future = model.make_future_dataframe(periods=periods, freq='1h')
future['cap'] = cap_value

forecast = model.predict(future)

print("Prophet fit range:", df_prophet['ds'].min(), "→", df_prophet['ds'].max())
print("Forecast range:   ", forecast['ds'].min(), "→", forecast['ds'].max())

model.plot(forecast)
plt.title(f'Prophet Forecast with Cap')
plt.show()

# ============================================================
# 5. Safe residual merge
# ============================================================

df_merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds', how='inner')
df_merged['residual'] = df_merged['y'] - df_merged['yhat']

print("✅ Merged shape for residuals:", df_merged.shape)
if df_merged.empty:
    raise ValueError("❌ Residual merge produced no data. Check overlap!")

# ============================================================
# 6. Residuals hourly → scaled for LSTM
# ============================================================

residuals = df_merged[['ds', 'residual']].dropna().set_index('ds')
residuals_hourly = residuals.resample('1h').mean().interpolate()

print("✅ Residuals hourly shape:", residuals_hourly.shape)

if residuals_hourly.empty:
    raise ValueError("❌ No residuals to train LSTM!")

scaler = StandardScaler()
scaled = scaler.fit_transform(residuals_hourly)

# ============================================================
# 7. Create LSTM sequences
# ============================================================

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

window_size = 24  # adjustable for your data size

X, y = create_sequences(scaled, window_size)

print(f"✅ LSTM samples: X={len(X)}, y={len(y)}")

if len(X) == 0:
    raise ValueError("❌ Not enough data for LSTM. Reduce window_size or check residuals.")

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ============================================================
# 8. Build & train LSTM
# ============================================================

model_lstm = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model_lstm.fit(
    X_train, y_train,
    validation_split=0.2 if len(X_train) >= 50 else 0.0,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# ============================================================
# 9. Evaluate residuals
# ============================================================

y_pred_test = model_lstm.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred_test).flatten()

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f'✅ Residuals MAE: {mae:.2f} | RMSE: {rmse:.2f}')

plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('LSTM Residuals Test Prediction')
plt.legend()
plt.show()

# ============================================================
# 10. Forecast residuals forward
# ============================================================

last_seq = scaled[-window_size:]
recursive_preds = []
horizon = len(forecast.loc[forecast['ds'].between(start_date, end_date)])

for _ in range(horizon):
    input_seq = last_seq[-window_size:].reshape(1, window_size, 1)
    pred = model_lstm.predict(input_seq, verbose=0)[0][0]
    recursive_preds.append(pred)
    last_seq = np.append(last_seq, [[pred]], axis=0)

residual_forecast = scaler.inverse_transform(np.array(recursive_preds).reshape(-1, 1)).flatten()

# ============================================================
# 11. Combine Prophet + LSTM residuals
# ============================================================

prophet_forecast = forecast.loc[forecast['ds'].between(start_date, end_date), 'yhat'].values
hybrid_forecast = prophet_forecast + residual_forecast
hybrid_forecast = np.clip(hybrid_forecast, 0, cap_value)

forecast_index = forecast.loc[forecast['ds'].between(start_date, end_date), 'ds'].values

final_df = pd.DataFrame({
    'Date': forecast_index,
    'Prophet': prophet_forecast,
    'Residuals': residual_forecast,
    'Hybrid_Available': hybrid_forecast.astype(int)
})

plt.figure(figsize=(14, 5))
plt.plot(final_df['Date'], final_df['Hybrid_Available'], label='Hybrid Forecast')
plt.plot(final_df['Date'], final_df['Prophet'], label='Prophet Only', linestyle='--')
plt.axhline(cap_value, color='red', linestyle='--', label='Capacity Cap')
plt.legend()
plt.title('✅ Hybrid Forecast (Exact Range)')
plt.show()

final_csv = "/home/sim06/Desktop/hybrid_forecast_final.csv"
final_df.to_csv(final_csv, index=False)
print(f"✅ Final CSV saved: {final_csv}")
