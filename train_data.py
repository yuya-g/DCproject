import pandas as pd
from HUFL_Pre import train_data as HUFL_train_data
from HULL_Pre import train_data as HULL_train_data
from MUFL_Pre import train_data as MUFL_train_data
from MULL_Pre import train_data as MULL_train_data
from LUFL_Pre import train_data as LUFL_train_data
from LULL_Pre import train_data as LULL_train_data
from OT_Pre import train_data as OT_train_data


# 統合の準備
MULL_train_data = MULL_train_data[['ds', 'y']].rename(columns={'y': 'MULL'})
MUFL_train_data = MUFL_train_data[['ds', 'y']].rename(columns={'y': 'MUFL'})
HUFL_train_data = HUFL_train_data[['ds', 'y']].rename(columns={'y': 'HUFL'})
HULL_train_data = HULL_train_data[['ds', 'y']].rename(columns={'y': 'HULL'})
LUFL_train_data = LUFL_train_data[['ds', 'y']].rename(columns={'y': 'LUFL'})
LULL_train_data = LULL_train_data[['ds', 'y']].rename(columns={'y': 'LULL'})
OT_train_data = OT_train_data[['ds', 'y']].rename(columns={'y': 'OT'})


# 'ds'（日時）列をキーにして統合
# 逐次的に結合
integrated_data = pd.merge(HUFL_train_data, HULL_train_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, MUFL_train_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, MULL_train_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, LUFL_train_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, LULL_train_data, on='ds', how='outer')
integrated_data = pd.merge(integrated_data, OT_train_data, on='ds', how='outer')

# 結合結果を確認1
print(integrated_data.head())

# 統合結果の確認2
integrated_data.to_csv("integrated_train_data.csv", index=False)
print("統合データが 'integrated_train_data.csv' に保存されました。")