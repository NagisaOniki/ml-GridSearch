
#全課題）ソースコード

# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データの先頭10行の表示
print( df.head(10) )

# データの行数・列数の表示
print( df.shape )

# データのカラム情報の表示
print( df.info() )

# データの数値型データについて基本統計量の表示
print( df.describe() )

# データのobject型データについて基本統計量の表示
print( df.describe(include=['O']) )

# 変数dfの型の表示
print( type(df) )

# カラムyの表示
print( df['y'] )

# カラムyのみを取り出したデータの型の表示
print( type(df['y']) )

# カラムage, job, yをこの順に取り出したデータの表示
print( df[['age', 'job', 'y']] )

# インデックス2,3 カラムage, job, yをこの順に取り出したデータの表示
print( df.loc[[2, 3], ['age', 'job', 'y']] )

# カラムyを取り除いたデータの表示
print( df.drop('y', axis=1) )

# カラムpoutcomeの要素の種類と出現数の表示
print( df['poutcome'].value_counts() )

# カラムyの要素の種類と出現数の表示
print( df['y'].value_counts() )

# クロス集計の実行
cross = pd.crosstab(df['poutcome'], df['y'], margins=True)

# クロス集計結果の表示
print( cross )

# 申込率の計算
rate = cross[1] / cross['All']

# クロス集計結果に申込率cvrを追加
cross['cvr'] = rate

# 申込率を追加したクロス集計表の表示
print( cross )

# 量的データの相関係数の計算
corr_matrix = df.corr()

# 量的データの相関係数の表示
print( corr_matrix )

# matplotlib.pyplotのインポート
import matplotlib.pyplot as plt

# seabornのインポート
import seaborn as sns

# heatmapを作成
sns.heatmap(corr_matrix, cmap="Reds")

# yの値が1のデータを表示
print( df[df['y']==1] )

# yの値が1のデータのdurationを表示
print( df[df['y']==1]['duration'] )


#-----------ヒストグラム-------------------
# matplotlib.pyplotのインポート
import matplotlib.pyplot as plt

# seabornのインポート
import seaborn as sns

#durationの抜き出し
duration_0 = df[df['y']==0]['duration']
duration_1 = df[df['y']==1]['duration']

# ヒストグラムの作成
sns.distplot(duration_0, label='y=0')
sns.distplot(duration_1, label='y=1')

# グラフにタイトルを追加
plt.title('duration histgram')

# グラフのx軸に名前を追加
plt.xlabel('duration')

# グラフのy軸に名前を追加
plt.ylabel('frequency')

# x軸の表示範囲の指定
plt.xlim(0, 2000)

# グラフに凡例を追加
plt.legend()

# グラフを表示
plt.show()

#--------------ダミー変数化---------------
# カラムjobをダミー変数化し先頭5行を表示
print( pd.get_dummies(df['job']).head(5) )

# データの列数の表示
print( df.shape[1] )

# ダミー変数化後のデータの列数の表示
print( pd.get_dummies(df).shape[1] )


#---------------説明変数と目的変数の分割--------------
# データのダミー変数化
df = pd.get_dummies(df)

# data_yに目的変数を代入
data_y = df['y']

# data_yの表示
print( data_y )

# data_Xに説明変数を代入
data_X = df.drop('y', axis=1)

# data_Xの表示
print( data_X )

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X = df.drop('y', axis=1)
data_y = df['y']

#----------------------学習用データと評価用データの分割---------------------------
# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 学習用データの説明変数の行数の表示
print( train_X.shape[0])

# 評価用データの説明変数の行数の表示
print( test_X.shape[0])


#-------------評価関数ライブラリのimport----------------------
# roc_auc_scoreのインポート
from sklearn.metrics import roc_auc_score

# AUCの計算結果の表示
print( roc_auc_score([0, 0, 1], [0, 1, 1]) )


#------------モデルの前準備---------------------------
# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth = 2, random_state = 0 )


#---------------モデルの学習------------------------
# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データのダミー変数化
df = pd.get_dummies(df)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth = 2, random_state = 0)

# 決定木モデルの学習
tree.fit(train_X, train_y)

#-----------------------説明変数の重要度の確認--------------------------
# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データのダミー変数化
df = pd.get_dummies(df)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth = 2, random_state = 0)

# 決定木モデルの学習
tree.fit(train_X, train_y)

# 重要度の表示
print( tree.feature_importances_ )

# 重要度に名前を付けて表示
print( pd.Series(tree.feature_importances_, index=train_X.columns) )


#-------------予測の実行--------------------------------
# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データのダミー変数化
df = pd.get_dummies(df)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth = 2, random_state = 0)

# 決定木モデルの学習
tree.fit(train_X, train_y)

# 評価用データの予測
pred_y1 = tree.predict_proba(test_X)[:,1]

# 予測結果の表示
print( pred_y1 )

#-------------------AUCの計算------------------------------------
import pandas as pd
from sklearn.metrics import roc_auc_score
df = pd.read_csv('data.csv', index_col='id')
df = pd.get_dummies(df)
data_X, data_y = df.drop('y', axis=1), df['y']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier as DT
tree = DT(max_depth = 2, random_state = 0)
tree.fit(train_X, train_y)

# 評価用データの予測
pred_y1 = tree.predict_proba(test_X)[:,1]

# 実測値test_y,予測値pred_y1を使ってAUCを計算
auc1 = roc_auc_score(test_y,pred_y1)

# 評価結果の表示
print( auc1 )


#----------------------ROC曲線の描画-------------------------------
from sklearn.metrics import roc_auc_score

# AUCの計算
auc1 = roc_auc_score(test_y, pred_y1)

# roc_curveのインポート
from sklearn.metrics import roc_curve

# 実測値test_yと予測値pred_y1を使って偽陽性率、真陽性率、閾値の計算
fpr, tpr, thresholds = roc_curve(test_y, pred_y1)

# ラベル名の作成
roc_label = 'ROC(AUC={:.2}, max_depth=2)'.format(auc1)

# ROC曲線の作成
plt.plot(fpr, tpr, label=roc_label)

# 対角線の作成
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

# グラフにタイトルを追加
plt.title('ROC')

# グラフのx軸に名前を追加
plt.xlabel('FPR')

# グラフのy軸に名前を追加
plt.ylabel('TPR')

# x軸の表示範囲の指定
plt.xlim(0, 1)

# y軸の表示範囲の指定
plt.ylim(0, 1)

# 凡例の表示
plt.legend()

# グラフを表示
plt.show()

#----------------------決定木の描画-----------------------------
# 決定木描画ライブラリのインポート
from sklearn.tree import export_graphviz

# 決定木グラフの出力
export_graphviz(tree, out_file="tree.dot", feature_names=train_X.columns, class_names=["0","1"], filled=True, rounded=True)

# 決定木グラフの表示
from matplotlib import pyplot as plt
from PIL import Image
import pydotplus
import io

g = pydotplus.graph_from_dot_file(path="tree.dot")
gg = g.create_png()
img = io.BytesIO(gg)
img2 = Image.open(img)
plt.figure(figsize=(img2.width/100, img2.height/100), dpi=100)
plt.imshow(img2)
plt.axis("off")
plt.show()


#---------------------モデルの前準備---------------------
# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データのダミー変数化
df = pd.get_dummies(df)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 評価関数のインポート
from sklearn.metrics import roc_auc_score

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth=10, random_state = 0)

#--------------------モデルの学習・予測・評価-----------------------
# pandasのインポート
import pandas as pd

# データの読み込み
df = pd.read_csv('data.csv', index_col='id')

# データのダミー変数化
df = pd.get_dummies(df)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df.drop('y', axis=1), df['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.25, random_state=0)

# 評価関数のインポート
from sklearn.metrics import roc_auc_score

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth=10, random_state=0)

# 決定木モデルの学習
tree.fit(train_X, train_y)

# 評価用データの予測
pred_y2 = tree.predict_proba(test_X)[:,1]

# AUCの計算
auc2 = roc_auc_score(test_y, pred_y2)

# 評価結果の表示
print( auc2 )


#----------------------グリッドサーチの前準備---------------------------
# 評価関数のインポート
from sklearn.metrics import roc_auc_score

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# グリッドサーチのインポート
from sklearn.model_selection import GridSearchCV

# 決定木モデルの準備
tree = DT(random_state=0)

# パラメータの準備
parameters = {'max_depth':[2,3,4,5,6,7,8,9,10]}

# グリッドサーチの設定
gcv = GridSearchCV(tree, parameters, cv=5, scoring='roc_auc', return_train_score=True)

#----------------グリッドサーチの実行-------------------
# グリッドサーチの設定
gcv = GridSearchCV(tree, parameters, cv=5, scoring='roc_auc', return_train_score=True)

# グリッドサーチの実行
gcv.fit(train_X, train_y)

#----------------グリッドサーチの結果の確認---------------------------
# 評価スコアの取り出し
train_score = gcv.cv_results_['mean_train_score']
test_score = gcv.cv_results_['mean_test_score']
print(train_score)
print(test_score)


#------------------train_scoreとtest_scoreの比較--------------------
# 評価スコアの取り出し
train_score = gcv.cv_results_["mean_train_score"]
test_score = gcv.cv_results_["mean_test_score"]

# matplotlib.pyplotを省略名pltとしてインポート
import matplotlib.pyplot as plt

# 学習に用いたデータを使って評価したスコアの描画
plt.plot([2,3,4,5,6,7,8,9,10], train_score, label="train_score")

# 学習には用いなかったデータを使って評価したスコアの描画
plt.plot([2,3,4,5,6,7,8,9,10], test_score, label="test_score")

# グラフにタイトルを追加
plt.title('train_score vs test_score')

# グラフのx軸に名前を追加
plt.xlabel('max_depth')

# グラフのy軸に名前を追加
plt.ylabel('AUC')

# 凡例の表示
plt.legend()

# グラフの表示
plt.show()


#--------------------最適パラメータモデルによる予測・評価------------------------
# 最適なパラメータの表示
print( gcv.best_params_ )

# 最適なパラメータで学習したモデルの取得
best_model = gcv.best_estimator_

# 評価用データの予測
pred_y3 = best_model.predict_proba(test_X)[:,1]

# AUCの計算
auc3 = roc_auc_score(test_y,pred_y3)

# AUCの表示
print ( auc3 )


#---------------------ROC曲線の描画------------------------------------
# matplotlib.pyplotのインポート
from matplotlib import pyplot as plt

# roc_curveのインポート
from sklearn.metrics import roc_curve

# 偽陽性率、真陽性率、閾値の計算
# なお、予測結果は以下の変数に代入されているものとします。
# pred_y1：max_depth=2の場合の予測結果
# pred_y2：max_depth=10の場合の予測結果
# pred_y3：max_depth=6の場合の予測結果
# また、それぞれの戻り値を代入する変数は以下とします。
# fpr1,tpr1,thresholds1：max_depth=2の場合の偽陽性率、真陽性率、閾値
# fpr1,tpr1,thresholds1：max_depth=10の場合の偽陽性率、真陽性率、閾値
# fpr1,tpr1,thresholds1：max_depth=6の場合の偽陽性率、真陽性率、閾値
fpr1, tpr1, thresholds1 = roc_curve(test_y, pred_y1)
fpr2, tpr2, thresholds2 = roc_curve(test_y, pred_y2)
fpr3, tpr3, thresholds3 = roc_curve(test_y, pred_y3)

# ラベル名の作成
# なお、それぞれの戻り値を代入する変数は以下とします。
# roc_label1：max_depth=2の場合のラベル名
# roc_label2：max_depth=10の場合のラベル名
# roc_label3：max_depth=6の場合のラベル名
roc_label1='ROC(AUC={:.2}, max_depth=2)'.format(auc1)
roc_label2='ROC(AUC={:.2}, max_depth=10)'.format(auc2)
roc_label3='ROC(AUC={:.2}, max_depth=6)'.format(auc3)

# ROC曲線の作成
plt.plot(fpr1, tpr1, label=roc_label1)
plt.plot(fpr2, tpr2, label=roc_label2)
plt.plot(fpr3, tpr3, label=roc_label3)

# 対角線の作成
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')

# グラフにタイトルを追加
plt.title('ROC')

# グラフのx軸に名前を追加
plt.xlabel('FPR')

# グラフのy軸に名前を追加
plt.ylabel('TPR')

# x軸の表示範囲の指定
plt.xlim(0, 1)

# y軸の表示範囲の指定
plt.ylim(0, 1)

# 凡例の表示
plt.legend()

# グラフを表示
plt.show()


#--------------------アタックリストの作成--------------------
# 申込率を含むアタックリストの作成
attack_list = pd.DataFrame(index=test_y.index, data={"cvr":pred_y3})

# 期待できる収益の計算
attack_list['return'] = 2000 * attack_list['cvr']

# 期待できるROIの計算
attack_list['ROI'] = attack_list['return'] / 300 * 100

# ROIで降順に並べ替え
attack_list = attack_list.sort_values('ROI', ascending=False)

# ROIが100%以上の顧客idを切り出し
attack_list = attack_list[attack_list['ROI'] >= 100]

# アタックリストの行数・列数の表示
print( attack_list.shape )

# アタックリストの先頭5行の表示
print( attack_list.head(5) )
