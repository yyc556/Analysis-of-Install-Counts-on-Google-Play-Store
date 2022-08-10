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
![image](https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/PCA%20result.png)
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
![image](https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/info%20gain.png)
![image](https://github.com/yyc556/analysis-of-install-counts-on-google-play-store/blob/main/images/gain%20ratio.png)

## 模型建立
Decision Tree, Random Forest, KNN, ANN
