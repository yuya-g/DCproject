# 必要なライブラリのインポート
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from prophet import Prophet  # Prophetのインポート
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA


# 1. 各ファイルから `forecast` をインポート
from HUFL_prophet import forecast as HUFL_forecast
from HULL_prophet import forecast as HULL_forecast
from LUFL_prophet import forecast as LUFL_forecast
from LULL_prophet import forecast as LULL_forecast

# 2. OT のテストデータの読み込み
data = pd.read_csv("/python/DC -project/integrated_train_data.csv")
df = pd.DataFrame(data)

# テストデータとして最後の30%を定義
split_point = int(len(data) * 0.7)
train_data = data.iloc[:split_point].copy()
test_data = data.iloc[split_point:].copy()  # テストデータ

# Prophetを使用して目的変数のトレンドと季節性を抽出
ot_prophet_data = train_data[['ds', 'OT']].rename(columns={'OT': 'y'})
ot_prophet_data['ds'] = pd.to_datetime(ot_prophet_data['ds'])

# Prophetモデルの設定と学習
ot_model = Prophet()
ot_model.fit(ot_prophet_data)

# 将来予測用データフレームの作成
future = ot_model.make_future_dataframe(periods=0)  # 既存データの範囲のみ
forecast = ot_model.predict(future)


# 季節性とトレンドを個別に取得
train_data['OT_trend'] = forecast['trend']

# Prophetの個別コンポーネントから季節性を取得
# seasonalは add_terms/multiplicative_terms の計算に基づいて再構成
train_data['OT_seasonality'] = (
    forecast['additive_terms'] - forecast['trend']
    if 'additive_terms' in forecast.columns
    else 0  # 必要に応じてデフォルト値を指定
)

# テストデータにもOTのトレンドと季節性を予測して追加
test_future = ot_model.make_future_dataframe(periods=len(test_data), freq='H')
test_forecast = ot_model.predict(test_future)

test_data['OT_trend'] = test_forecast['trend'].iloc[-len(test_data):].values
test_data['OT_seasonality'] = (
    test_forecast['additive_terms'].iloc[-len(test_data):].values - test_forecast['trend'].iloc[-len(test_data):].values
    if 'additive_terms' in test_forecast.columns
    else 0
)
# 残差を計算
train_data['OT_residual'] = train_data['OT'] - (train_data['OT_trend'] + train_data['OT_seasonality'])
test_data['OT_residual'] = test_data['OT'] - (test_data['OT_trend'] + test_data['OT_seasonality'])
print(train_data['OT_residual'].head())

# 予測データを準備
HUFL_predic_data = HUFL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'HUFL'})
HULL_predic_data = HULL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'HULL'})
LUFL_predic_data = LUFL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'LUFL'})
LULL_predic_data = LULL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'LULL'})
data['ds'] = pd.to_datetime(data['ds'])
train_data['ds'] = pd.to_datetime(train_data['ds'])
test_data['ds'] = pd.to_datetime(test_data['ds'])
OT_data = pd.concat([train_data, test_data], ignore_index=True)
OT_trend_data = OT_data[['ds', 'OT_trend']]
OT_seasonality_data = OT_data[['ds', 'OT_seasonality']]
OT_actual_data = OT_data[['ds', 'OT']]

# 'ds'（日時）列をキーにして統合
integrated_data = pd.merge(HUFL_predic_data, HULL_predic_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, LUFL_predic_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, LULL_predic_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, OT_trend_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, OT_seasonality_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, OT_actual_data, on='ds', how='outer')


# 特徴量とターゲットを分ける
X = integrated_data[['HUFL', 'HULL', 'LUFL', 'LULL']]  # 特徴量にトレンドと季節性を追加
y_residual = pd.concat([train_data['OT_residual'], test_data['OT_residual']], ignore_index=True)

# 欠損値を補完（線形補完を例として使用）
X = X.interpolate(method='linear')
y_residual = y_residual.interpolate(method='linear')

# モデルの学習
X_train, X_test, y_residual_train, y_residual_test = train_test_split(X, y_residual, test_size=0.3, shuffle=False, random_state=42)

# 主成分分析の実行
pca = PCA(n_components=len(X.columns))  # 全ての主成分を取得
pca.fit(X_train)

# 各主成分の寄与率を確認
explained_variance_ratio = pca.explained_variance_ratio_

# 定数項を追加
X_with_constant = add_constant(X_train)

# VIFの計算
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i+1) for i in range(X.shape[1])]
vif_data["寄与率"] = explained_variance_ratio
print(vif_data)

# モデルの作成と学習
model = LinearRegression()
model.fit(X_train, y_residual_train)

# 残差の予測
y_residual_pred_test = model.predict(X_test)

# 評価指標を計算
mae_residual = mean_absolute_error(y_residual_test, y_residual_pred_test)
rmse_residual = np.sqrt(mean_squared_error(y_residual_test, y_residual_pred_test))

print(f"Residual MAE: {mae_residual}")
print(f"Residual RMSE: {rmse_residual}")

test_data['OT_final_pred'] = y_residual_pred_test + test_data['OT_trend'].iloc[-len(y_residual_pred_test):].values + test_data['OT_seasonality'].iloc[-len(y_residual_pred_test):].values
integrated_data['Predicted'] = pd.concat([
    train_data['OT_trend'] + train_data['OT_seasonality'] + model.predict(X_train),
    test_data['OT_final_pred']
], ignore_index=True)


# 最終的な評価指標
mae_final = mean_absolute_error(test_data['OT'], test_data['OT_final_pred'])
rmse_final = np.sqrt(mean_squared_error(test_data['OT'], test_data['OT_final_pred']))

print(f"Final MAE: {mae_final}")
print(f"Final RMSE: {rmse_final}")

# 実測値と予測値の可視化
plt.figure(figsize=(12, 6))
plt.plot(integrated_data['ds'], integrated_data['OT'], label='Actual', alpha=0.6)
plt.plot(integrated_data['ds'], integrated_data['Predicted'], label='Predicted', alpha=0.8)
plt.axvline(x=data.iloc[split_point]['ds'], color='red', linestyle='--', label='Train-Test Split')
plt.legend()
plt.title('Actual vs Predicted')
plt.ylabel('OT')
plt.show()
