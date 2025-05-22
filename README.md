# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.

2.Analyse the data.

3.Use modelselection and Countvectorizer to preditct the values.

4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: R.SABARINATH
RegisterNumber: 212223100048
*/
```
```
import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:

![image](https://github.com/user-attachments/assets/86f534e8-5a02-4bd6-a4a7-7a4a101e7224)

![image](https://github.com/user-attachments/assets/7db6dd8e-a586-4b76-ae53-f47c43839fc6)

![image](https://github.com/user-attachments/assets/04c37fe8-e9df-4396-8d6b-6eeb63283fe4)

![image](https://github.com/user-attachments/assets/1112f21b-191d-4079-b382-36f40bfbe846)

![image](https://github.com/user-attachments/assets/6ab9098c-bdf8-4604-a626-9843345e4d70)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
