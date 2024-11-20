# ライブラリのインポート
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# CSVファイルの読み込み
df = pd.read_csv(/assignment-main/AI-Engineer/multivariate-time-series-prediction/ett.csv") 

# 1. データフレーム準備
df = df[['date', 'HUFL']]  # 日付と予測対象の値のみを選択
df.rename(columns={'date': 'ds', 'HUFL': 'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'])  # "date"列をdatetime型に変換

# 総データ数を確認
total_length = len(df)
split_point = int(total_length * 0.7)
train_data = df.iloc[:split_point].copy()  # 訓練データ（最初の70%）
test_data = df.iloc[split_point:].copy()  # テストデータ（残りの30%）

# 確認
print(f"Train Data: {train_data.shape}")
print(f"Test Data: {test_data.shape}")

# 2. Prophetモデルの設定
model = Prophet()
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # 月次（約30.5日周期）
model.add_seasonality(name='yearly', period=365.25, fourier_order=10)  # 年次（約365日周期）
model.fit(train_data)

# 5. 予測
future = model.make_future_dataframe(periods=0, freq='H')  
forecast = model.predict(future)

# 6. 結果のプロット
# 訓練データに予測結果を追加
train_data['yhat'] = forecast['yhat']
train_data['residual'] = train_data['y'] - train_data['yhat']  # 残差計算

# 異常値の閾値（残差の標準偏差の3倍）
threshold = 3 * train_data['residual'].std()

# 異常値を検出
train_data['outlier'] = abs(train_data['residual']) > threshold

# 異常値の確認
outliers = train_data[train_data['outlier']]
outlier_indices = train_data.index[train_data['outlier']] 
print(f"Number of Outliers Detected: {len(outliers)}")
print(outliers)

# 異常値の箇所をNaNにする
train_data.loc[outlier_indices, 'y'] = np.nan

# 線形補間を適用
train_data['y'] = train_data['y'].interpolate(method='linear')
# 補間後の確認
print("HUFL")
print(train_data.loc[outlier_indices, ['ds', 'y']])

# Prophetモデルの再フィッティング
model = Prophet()
model.fit(train_data[['ds', 'y']])  # 訓練データのみでフィッティング

# 再予測の準備
future = model.make_future_dataframe(periods=0)  # 訓練データのみを予測
forecast = model.predict(future)

