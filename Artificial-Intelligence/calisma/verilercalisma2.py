import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# veri ön işleme 
# veri yükleme
veriler = pd.read_csv("C:\\Users\\mrtkr\\OneDrive\\Masaüstü\\Ders\\yapayzekacalisma\\calisma\\satislar.csv")

print(veriler)
# veri ön işleme

aylar = veriler [['Aylar']]
print(aylar)

satislar = veriler [['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

# verilerin eğitim ve test için bölünmesi

from sklearn.model_selection import train_test_split

x_train, x_test ,y_train, y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

#model inşası (lineer regresyon)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

tahmin = lr.predict(x_test)
print(tahmin)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,tahmin)

plt.title('aylara göre satiş')
plt.xlabel('Aylar')
plt.ylabel('Satislar')
plt.show()