# 必要なライブラリのインポート
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA




# 1. 各ファイルから `forecast_test` をインポート
from HUFL_prophet import forecast as HUFL_forecast
from HULL_prophet import forecast as HULL_forecast
from LUFL_prophet import forecast as LUFL_forecast
from LULL_prophet import forecast as LULL_forecast

# 2. OT のテストデータの読み込み
data = pd.read_csv("/Users/isogaiyuya/Downloads/温泉道場/python/DC -project/integrated_train_data.csv")
split_point = int(len(data) * 0.7)
data['ds'] = pd.to_datetime(data['ds'])
df = pd.DataFrame(data)

# 統合の準備
HUFL_predic_data = HUFL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'HUFL'})
HULL_predic_data = HULL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'HULL'})
LUFL_predic_data = LUFL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'LUFL'})
LULL_predic_data = LULL_forecast[['ds', 'yhat']].rename(columns={'yhat': 'LULL'})
OT_actual_data = data[['ds', 'OT']]
OT_actual_data['ds'] = pd.to_datetime(OT_actual_data['ds'])

# 'ds'（日時）列をキーにして統合
# 逐次的に結合
integrated_data = pd.merge(HUFL_predic_data, HULL_predic_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, LUFL_predic_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, LULL_predic_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, OT_actual_data, on='ds', how='outer')

# 4. 重回帰分析の実行
# 特徴量とターゲットを分ける
X = integrated_data[['HUFL', 'HULL', 'LUFL', 'LULL']]  # 特徴量
y = integrated_data['OT']  # ターゲット変数（実測 OT）
integrated_data['ds'] = pd.to_datetime(integrated_data['ds'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False, random_state=42)

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
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X)
y_pred_test = model.predict(X_test)

# 評価指標の計算
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# 実測値と予測値の可視化
plt.figure(figsize=(12, 6))
plt.plot(integrated_data['ds'], y, label='Actual', alpha=0.6)
plt.plot(integrated_data['ds'], y_pred, label='Predicted', alpha=0.8)
plt.axvline(x=data.iloc[split_point]['ds'], color='red', linestyle='--', label='Train-Test Split')
plt.legend()
plt.title('Actual vs Predicted')
plt.ylabel('OT')
plt.show()

