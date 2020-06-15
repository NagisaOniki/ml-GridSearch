# pandasのインポート
import pandas as pd
# train_test_splitのインポート
from sklearn.model_selection import train_test_split
# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT
# グリッドサーチのインポート
from sklearn.model_selection import GridSearchCV
# matplotlib.pyplotを省略名pltとしてインポート
import matplotlib.pyplot as plt
# AUC
from sklearn.metrics import roc_auc_score



#----------データ作成------------------------
# データの読み込み
df = pd.read_csv('../input/train.csv', index_col='id')
tf = pd.read_csv('../input/test.csv', index_col='id')
# データのダミー変数化
df = pd.get_dummies(df)
tf = pd.get_dummies(tf)
# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']
# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.1, random_state=0)


############グリッドサーチ付き予測#######################
#--------------モデル作成---------------------
# 決定木モデルの準備
tree = DT(random_state=0)
#-------------グリッドサーチ----------------------
# パラメータの準備
parameters = {
'max_depth':[2,3,4,5,6,7,8,9,10],
"criterion": ["gini", "entropy"],
"splitter": ["best", "random"],
"min_samples_split": [i for i in range(2, 11)],
"min_samples_leaf": [i for i in range(1, 11)],
# "random_state": [i for i in range(0, 101)],
'max_leaf_nodes' :  [2,4,6,8,10,12,14,16,18,20],
# "max_features": ['log2', 'sqrt','auto'],
# "n_estimators":[50,100,200,300,400,500],
'class_weight':[{0: w} for w in [1, 2, 4, 6, 10]]}
}

# グリッドサーチの設定
gcv = GridSearchCV(tree, parameters, cv=5, scoring='roc_auc', return_train_score=True)

# グリッドサーチの実行
gcv.fit(train_X, train_y)

# 最適なパラメータの表示
print( gcv.best_params_ )


#----------グリッドサーチ後の最適モデル---------------
# 最適なパラメータで学習したモデルの取得
best_model = gcv.best_estimator_

#------------予測----------------------------
# 評価用データの予測
pred_y = best_model.predict_proba(test_X)[:,1]
# 予測結果の表示
print( pred_y )


#---------------AUC------------------------
# 実測値test_y,予測値pred_y1を使ってAUCを計算
auc = roc_auc_score(test_y,pred_y)

# 評価結果の表示
print( auc )



#-----------本番予測-----------------------
pred = best_model.predict_proba(tf)[:,1]
print( pred )

#----------予測値ファイル作成-----------------
submit = pd.read_csv("../input/sample_submission.csv", header=None)
submit[1] = pred
submit.to_csv("submit.csv", index=False, header=False)
