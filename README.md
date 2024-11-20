# DCproject

## 背景
現在の電力配分は、経験に基づく推定に頼っているため、過剰供給が発生し、電力や設備資源の無駄が生じています。一方で、オイル温度は変圧器の状態を反映する重要な指標であり、その予測を通じて以下の効果が期待されます：
- **最適な負荷配分の実現**
- **余分な電力供給の削減**
- **設備資源の効率的利用**

本プロジェクトでは、オイル温度の時系列データを活用し、予測モデルを構築することでこれらの課題に対応します。

---
## 前提
### Prophetを選んだ理由

### **(a) 周期性のモデリング手法**
- 上記の**フーリエ変換**による周期性分析を内部的に利用し、ユーザーが特別な知識を持たなくてもデータの周期性をモデル化できる。
- 周期性の自動分解においてSTL分解や他の分解手法と同等以上の効果を発揮。

### **(b) トレンドのモデリング手法**
- Prophetは上記の「移動平均」や「多項式回帰」によるトレンド分析を超えて、変化点に対応可能なトレンドモデリングを採用。
- **線形トレンド**と**ロジスティック成長**モデルを柔軟に切り替えられる点で、長期的・非線形的なトレンド変化への対応力が高い。

### **(c) 異常値や残差の処理**
- 上記の枠組みでは異常値の影響を受けやすい「移動平均」や「加法モデル」と比較し、Prophetは異常値に対してロバストな推定を行う仕組みを持つ。

### **(d) モデルの解釈性**
- STL分解と同様に、Prophetはトレンド・周期性・残差を独立した成分として分離するため、結果の解釈性が高い。
- また、予測結果に対する不確実性の範囲（予測区間）を提供することで、統計的な信頼性を担保。

### **(e) 残差部分のランダムフォレスト検証**
- Prophetのトレンド・周期性を除いた残差をランダムフォレストで予測したが、MAEやRMSEの精度が下がった。

---
## データの前処理
データにはProphetを用いて異常値検出と線形補間を行い、信頼性の高い入力データを生成しました。
:yuya-g-pretreatment

---
## プレ分析

### 前提モデル
scikit-learnの重回帰モデルを基盤として、多重共線性を除去したモデル構築を進めました。
説明変数の実測値を使って目的変数（OT）をどのくらい説明でき、多重共線性がないかを確認しました。
:yuya-g-preanalytics
結論は説明変数を[HUFL,LUFL,HULL,LULL]のモデルが最適でした。
:yuya-g-preanalytics/regressionMAX.py

| Feature | VIF      | 寄与率  |
|---------|----------|---------|
| HUFL    | 1.060204 | 0.868268 |
| LUFL    | 1.139002 | 0.096100 |
| HULL    | 1.144633 | 0.028035 |
| LULL    | 1.186592 | 0.007597 |
- **MAE**: 5.4453  
- **RMSE**: 6.9816  
- **R²**: 0.2861  

---

## 検証結果

## 仮説0: 
HULやLULを使わずにオイル温度の過去のデータから未来を予測するProphetモデルの精度は？
: pattern0
### 実行内容0： 
1. **データの読み込みと前処理**
   - `integrated_train_data.csv` からデータを読み込み。
   - 必要な列 (`ds`：日付, `OT`：対象変数) を選択し、Prophet用に列名を変更。
   - 日付を適切な形式に変換。
2. **データの分割**
   - データを **70%を訓練データ**、**残り30%をテストデータ**に分割。
3. **モデルの構築**
   - Prophetモデルを構築し、以下の季節性を設定:
     - **年次季節性**（有効）
     - **月次季節性**（約30.5日周期、Fourier次数5）
     - **日次季節性**（有効）
   - 訓練データを使用してモデルを学習。
4. **予測**
   - テストデータ期間を含む将来の値を予測。
   - テストデータ部分の予測値を抽出。

## 結果0:
- **MAE**: 9.2080  
- **RMSE**: 11.2260
--- 

## 仮説1: 
HUFL,LULLなどを時系列データとしてProphetで予測し、重回帰でOTの予測を立てる方法はどうか。
: pattern1
### 実行内容1: 
1. **データの準備**
   - 各予測モデル（HUFL, HULL, LUFL, LULL）から予測結果 (`forecast_test`) をインポート。
   - 実測値データ (`OT`) と予測値データを日時 (`ds`) をキーに統合。
2. **データの分割**
   - 特徴量（HUFL, HULL, LUFL, LULL）と目的変数（実測値 OT）を設定。
   - データを **訓練データ**（70%）と **テストデータ**（30%）に分割。
3. **モデルの学習**
   -  **訓練データ** でscikit-learnで重回帰分析を行い、その関数をテストデータに持ち込んで検証。
   -  特徴量のテストデータはProphetによる予測値を使って、目的変数を予測。
4. **可視化**
   - 実測値（Actual）と予測値（Predicted）を時系列グラフにプロット。
   - 訓練データとテストデータの分割点をプロット上に表示。
   
## 結果1: 
- **MAE**: 6.5489  
- **RMSE**: 7.7795  
---

## 仮説2: 
目的変数（OT）の時系列特性（季節性・トレンド）を特徴量として取り出し、重回帰モデルを構築すれば精度が上がるのではないか。
: pattern2
### 実行内容2: 
1. **データの準備**
   - 各予測モデル（HUFL, HULL, LUFL, LULL）の予測結果 (`forecast_test`) を読み込み。
   - 実測データ (`OT`) に対し、Prophetを使用して **トレンド** と **季節性** を分解。
   - 実測値、予測値、トレンド、季節性を日時 (`ds`) をキーに統合。
2. **データの補完と分割**
   - 特徴量（説明変数予測値 + トレンド + 季節性）とターゲット変数（実測 OT）を設定。
   - データを **訓練データ**（70%）と **テストデータ**（30%）に分割。
3. **モデルの構築と学習**
   - scikit-learnの重回帰モデルを用いて、訓練データでモデルを学習。
4. **結果の可視化**
   - 実測値（Actual）と予測値（Predicted）を時系列グラフにプロット。
   - 訓練データとテストデータの分割点を表示。
     
## 結果2: 
- **MAE**: 8.3586  
- **RMSE**: 10.1403
### 課題2: 
トレンド成分と季節性成分のVIF値が非常に高い、寄与率は非常に低い。
---

### 仮説3: 
目的変数から季節性・トレンドを差し引き、残差に対して重回帰を適用し、予測に季節性・トレンドを再付加するモデルで精度は上がるのではないか。
:pattern3
### 実行内容3: 
1. **データの準備**
   - 実測データ (`OT`) に対し、Prophetを用いて **トレンド** と **季節性** を分解。
   - 残差（実測値 - トレンド - 季節性）を計算。
2. **予測データの統合**
   - 複数の予測モデル（HUFL, HULL, LUFL, LULL）の予測結果を日時 (`ds`) をキーに統合。
   - 統合データにトレンド・季節性・残差を追加。
3. **残差予測モデルの構築**
   - トレンドと季節性を除去した残差部分をターゲットとして、特徴量（HUFL, HULL, LUFL, LULL）を用いて学習。
   - scikit-learnの重回帰モデルを使用し、訓練データで残差を予測。
4. **最終的な予測**
   - 残差予測値にトレンドと季節性を加算し、最終的な予測値を算出。
5. **結果の可視化**
   - 実測値と最終予測値を時系列グラフにプロット。
   - 訓練データとテストデータの分割点を表示。
     
## 結果3: 
- **Residual MAE**: 6.7543  
- **Residual RMSE**: 8.0062  
---

## 今後の課題
1. **さらなるモデル改善**  
   - 他のアルゴリズム（例：LSTMやXGBoost）の導入。
   - 非線形特性を考慮したモデルの構築。
2. **システムへの応用**  
   - 本予測モデルを電力配分システムに統合し、負荷配分の最適化を図る。

---

## 連絡先
本プロジェクトに関する質問やフィードバックは以下にお願いします：
- Email: yuyaiso@icloud.com

--- 

