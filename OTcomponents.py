# ライブラリのインポート
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# CSVファイルの読み込み
data = pd.read_csv("/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv") 

# 1. データフレーム準備
data = data[['date', 'OT']]  # 日付と予測対象の値のみを選択
data.rename(columns={'date': 'ds', 'OT': 'y'}, inplace=True)
data['ds'] = pd.to_datetime(data['ds'])  # "date"列をdatetime型に変換
data = data.dropna()  # 欠損値を削除

# 2. Prophetモデルの設定
model = Prophet()
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # 月次（約30.5日周期）
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)  # 年次（約365日周期）
model.fit(data)

# 5. 予測
future = model.make_future_dataframe(periods=48, freq='H')  # 将来48時間を予測
forecast = model.predict(future)

# 6. 結果のプロット
model.plot(forecast)
plt.show()

model.plot_components(forecast)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
plt.figure(figsize=(10, 5))
plot_acf(data['y'], lags=168)  
plt.title('Autocorrelation Plot')
plt.show()
