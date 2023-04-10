import pandas as pd
import numpy as np
df=pd.read_csv("breast-cancer.data",header=None,na_values="?")
for i in range(0,10):
    print(df[i].unique())
print(df.isna().sum())
from sklearn.impute import SimpleImputer
df_i=pd.DataFrame(SimpleImputer(strategy="most_frequent").fit_transform(df))
print(df_i.isna().sum())
from sklearn.preprocessing import LabelEncoder
for i in range(0,10):
    df_i[i]=LabelEncoder().fit_transform(df_i[i])
print(df_i.head())
target=df_i[0]
data=df_i.drop(columns=[0])
print(target.shape,data.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,ytrain,ytest=train_test_split(data,target,test_size=0.3)
print(x_train.shape,x_test.shape,ytrain.shape,ytest.shape)

from sklearn.linear_model import Perceptron
model=Perceptron()

from sklearn.metrics import accuracy_score
pred_train=model.predict(x_train)
pred_test=model.predict(x_test)
print("Accuracy for training ",accuracy_score(pred_train,ytrain))
print("Accuracy for testing ",accuracy_score(pred_test,ytest))

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,ytrain)
from sklearn.metrics import accuracy_score
pred_train=model.predict(x_train)
pred_test=model.predict(x_test)
print("Accuracy for training ",accuracy_score(pred_train,ytrain))
print("Accuracy for testing ",accuracy_score(pred_test,ytest))
