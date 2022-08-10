# Google Play Store Apps 下載量分析
大數據分析與應用-蔡孟勳 課程期末專題
<br>組員：陳彥妤、林旭容

* 資料集
<br>[Google Play Store Apps | Kaggle](https://www.kaggle.com/datasets/lava18/google-play-store-apps)
* 研究動機
<br>分析APP的各項資料與下載量的關係，了解哪些因素對下載量的影響較大。提供App開發者一項依據來設定App的推出方向。

## 資料預處理
1. 處理缺失值
<br>將有缺失值的資料整筆刪除
```
data.dropna(inplace=True)
```
2. 類別型資料轉換
<br>利用LabelEncoder將類別型資料轉換成數值
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Category'] = le.fit_transform(data['Category']) 
data['Content Rating'] = le.fit_transform(data['Content Rating']) 
data['Type'] = le.fit_transform(data['Type'])
data['Genres'] = le.fit_transform(data['Genres'])
data['Last Updated'] = le.fit_transform(data['Last Updated']) 
data['Current Ver'] = le.fit_transform(data['Current Ver']) 
data['Android Ver'] = le.fit_transform(data['Android Ver'])
```
3. 個別欄位處理
* Size
```
# 把資料中的"M"去掉
df_size['Size'] = df_size['Size'].map(lambda x: str(x)[:-1])
# String轉float
df_size['Size'] = df_size['Size'].astype('float')
# 標記1~100M, 100M~
for i in range(len(df_size)):
  if df_size['Size'].iat[i] < 100:
    df_size['Size'].iat[i] = '2'
  else:
    df_size['Size'].iat[i] = '3'
# 合併Dataframe
data = pd.concat([data,df_size])
data['Size'] = data['Size'].astype('int')
```
* Install
```
# 把資料中的"+"去掉
data['Installs'] = data['Installs'].map(lambda x: str(x)[:-1])  # 把資料中的","去掉
data['Installs'] = data['Installs'].str.replace(',', '')  # String轉Int
data['Installs'] = data['Installs'].astype('int')

# 再利用replace()將Installs分割成4個區間(1 ~ 5000+, 100000+ ~ 5000000+, 10000000+ ~ 500000000+, 1000000000+)
data['Installs'] = data['Installs'].replace([1,5,10,50,100,500,1000,5000], 0)
data['Installs'] = data['Installs'].replace([10000,50000,100000,500000,1000000,5000000], 1)
data['Installs'] = data['Installs'].replace([10000000,50000000,100000000,500000000], 2)
data['Installs'] = data['Installs'].replace([1000000000], 3)
```
* Price
```
# 把資料中的"$"去掉
data['Price'] = data['Price'].str.lstrip('$')
# String轉float
data['Price'] = data['Price'].astype('float')
```
4. 資料標準化 & 去除離群值
<br>Reviews欄位做z-score標準化，再去除離群值(z>3)
```
data['Reviews'] = data['Reviews'].astype('int')
mean = data['Reviews'].mean()
std = data['Reviews'].std()
data['Reviews'] = data['Reviews'].apply(lambda x: (x-mean)/ std)

for i in range(len(data)):
  if data['Reviews'].iat[i]>3:
    data.drop(labels=i)
data = data.reset_index(drop=True)
```
5. 特徵選取
<br>使用PCA(0.95)配合Info Gain與Gain Ratio做特徵選取
```
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95, whiten=True)
features_pca = pca.fit_transform(features)
print("Original number of features:", features.shape[1])
print("Reduce number of features:", features_pca.shape[1])
```
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/PCA%20result.png" width=40%>

```
# Info Gain與Gain Ratio
!pip install info_gain

# Installs欄位為target column
X=data.drop(['Installs'],axis=1)
y=data.Installs

from info_gain import info_gain
import pandas as pd
print('info_gain:')
infogain={}
for i in data.columns:
  ig = info_gain.info_gain(data[i], data['Installs'])
  infogain[i]=ig
a=sorted(infogain.items(),key=lambda item:item[1])
for i in a:
  print(i)
print('\ngain_ratio:')
gainratio={}
for i in data.columns:
  igr = info_gain.info_gain_ratio(data[i], data['Installs'])
  gainratio[i]=igr
b=sorted(gainratio.items(),key=lambda item:item[1])
for i in b:
  print(i)
```
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/info%20gain.png" width=40%>
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/gain%20ratio.png" width=40%>

```
# 綜合PCA和Info Gain/Gain Ratio結果，刪除'Type', 'Content Rating'兩個欄位
data = data.drop(labels=['Type', 'Content Rating'],axis=1)  
```

## 模型建立
Decision Tree, Random Forest, KNN, ANN
* Decision Tree
```
from sklearn import model_selection, tree, metrics
dtc = tree.DecisionTreeClassifier()
dtc.fit(X_train, y_train)
print(metrics.classification_report(y_true=y_test, y_pred=dtc.predict(X_test)))
metrics.confusion_matrix(y_true=y_test, y_pred=dtc.predict(X_test))

import pydotplus
from IPython.display import Image

feature_names = ['Category', 'Rating', 'Reviews', 'Size', 'Price', 'Genres', 'Last Updated', 'Current Ver', 'Android Ver']
traget_name = ['0','1','2','3']
dot_data = tree.export_graphviz(dtc, out_file=None, rounded=True, special_characters=True, feature_names=feature_names, class_names=traget_name, max_depth=3)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
```
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/decision%20tree.png">

* Random Forest
```
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,roc_auc_score,auc,accuracy_score,confusion_matrix,classification_report
rfc=RandomForestClassifier(n_estimators=5)
rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))

imp=rfc.feature_importances_
names=data.columns
zip(imp,names)
imp, names= zip(*sorted(zip(imp,names)))
plt.barh(range(len(names)),imp,align='center')
plt.yticks(range(len(names)),names)
plt.xlabel('Importance of Features')
plt.ylabel('Features')
plt.title('Importance of Each Feature')
plt.show()
```
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/random%20forest.png" width=40%>

* KNN
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import roc_curve,roc_auc_score,auc,accuracy_score,confusion_matrix,classification_report
import pydotplus
from IPython.display import Image

error_rate=[]
for i in range(1,10):
  knn=KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train,y_train)
  pred_i=knn.predict(X_test)
  error_rate.append(np.mean(pred_i!=y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue',linestyle='dashed',marker='o',markerfacecolor='red',markersize=10)
plt.title('Error rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# with k=9
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)
print('WITH k=9')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
```
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/k%20value%20compare.png" width=40%>

* ANN
```
import pandas as pd
from sklearn import preprocessing, neural_network, model_selection
mms = preprocessing.MinMaxScaler()
mlp = neural_network.MLPClassifier()
mlp.fit(X_train, y_train)
#mlp.predict([[0, 4.1, 0, 0, 0, 9, 561, 2582, 8]])
mlp.score(X_test, y_test)
```

## 結果比較
* ROC curve
看出Random Forest的準確度較高
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/ROC%20curve.png" width=40%>

* Random Forest Visulization
發現 Rating和下載量的高低最相關
<img src="https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/random%20forest.png" width=40%>


