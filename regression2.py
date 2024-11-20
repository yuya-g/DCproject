# 必要なライブラリのインポート
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.decomposition import PCA

data = pd.read_csv("/Users/isogaiyuya/Downloads/温泉道場/python/DC -project/integrated_train_data.csv")
df = pd.DataFrame(data)
# データの準備
X = df[['HUFL', 'MUFL', 'LUFL', 'HULL', 'MULL', 'LULL']] # 特徴量
y = df['OT']  # 目的変数

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 主成分分析の実行
pca = PCA(n_components=len(X.columns))  # 全ての主成分を取得
pca.fit(X_train)

# 各主成分の寄与率を確認
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance_ratio}")

# 定数項を追加（必要条件）
X_with_constant = add_constant(X_train)

# VIFの計算
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_constant.values, i+1) for i in range(X.shape[1])]

# 出力
print(vif_data)

# モデルの作成と訓練
model = LinearRegression()
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)  
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)            # 決定係数
mae = mean_absolute_error(y_test, y_pred)



print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")


