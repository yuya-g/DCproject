import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("/python/DC -project/integrated_train_data.csv")
df = df[['ds', 'OT']]  # 日付と予測対象の値のみを選択
df.rename(columns={'ds': 'ds', 'OT': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])  # "date"列をdatetime型に変換

# 訓練データとテストデータに分割
train_size = int(len(df) * 0.7)
train_data = df.iloc[:train_size].copy()
test_data = df.iloc[train_size:].copy()

# --- Prophet モデルの構築 ---
model = Prophet(    yearly_seasonality=True,  # 年次季節性
                    weekly_seasonality=False,  # 週次季節性（必要なら有効化）
                    daily_seasonality=True)  # デフォルトの「日次季節性」は無効化
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # 月次（約30.5日周期）
model.fit(train_data[['ds', 'y']])

# テストデータを予測
future = model.make_future_dataframe(periods=len(test_data), freq='H')
forecast = model.predict(future)

# テストデータ部分を抽出
forecast_test = forecast[-len(test_data):]

# --- モデル評価 ---
mae = mean_absolute_error(test_data['y'], forecast_test['yhat'])
rmse = np.sqrt(mean_squared_error(test_data['y'], forecast_test['yhat']))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# --- プロット ---
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Actual', alpha=0.6)
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', alpha=0.8)
plt.axvline(x=df.iloc[train_size]['ds'], color='red', linestyle='--', label='Train-Test Split')
plt.legend()
plt.title('Actual vs Predicted')
plt.ylabel('OT')
plt.show()
