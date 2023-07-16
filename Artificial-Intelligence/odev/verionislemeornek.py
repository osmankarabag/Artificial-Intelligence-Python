#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('C:\\Users\\mrtkr\\OneDrive\\Masaüstü\\Ders\\yapayzekacalisma\\odev\\tenis.csv')
#pd.read_csv("veriler.csv")

from sklearn import preprocessing
#veri on isleme
from sklearn.preprocessing import LabelEncoder

#encoder:  Kategorik -> Numeric
# bu satır 0 ve 1 lere dönüştürme işlemini tek seferde bütün tablo üzerinde yapıyor tek tek yapmamızı zaman kaybını önlüyor
veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler], axis = 1)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)

print(y_pred)
#backward elimination
import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

sonveriler = sonveriler.iloc[:,1:]

import statsmodels.api as sm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )
X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)







