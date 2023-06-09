import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_squared_error
from  sklearn.model_selection import train_test_split
data=pd.read_csv("carbon_nanotubes_correct.csv")
print(data.head())
print(data.isna().sum())
print(data.info())
print(data.describe())
X = data["Calculated atomic coordinates w'"]
Y = data.drop(columns = "Calculated atomic coordinates w'" )
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size = 0.3)

#linear Regression Model
lr  = LinearRegression()
lr.fit(xtrain.values.reshape(-1,1),ytrain)
lr_pred = lr.predict(xtest.values.reshape(-1,1))
lr# print(r2_score(ytest,lr_pred))
lr_accuracy = r2_score(ytest,lr_pred)
print(lr_accuracy)

#Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(xtrain.values.reshape(-1,1),ytrain)
dt_pred = dt.predict(xtest.values.reshape(-1,1))
# print(r2_score(ytest,dt_pred))
dt_accuracy = r2_score(ytest,dt_pred)
print(dt_accuracy)

#Rendom Forest Regressor
rfr = RandomForestRegressor()
rfr.fit(xtrain.values.reshape(-1,1),ytrain)
rfr_pred = rfr.predict(xtest.values.reshape(-1,1))

rfr_accuracy = r2_score(ytest,rfr_pred)
print(rfr_accuracy)

print(mean_squared_error(ytest,lr_pred))
print(mean_squared_error(ytest,rfr_pred))
print(mean_squared_error(ytest,dt_pred))

#Logistic Regression
X = data[["Initial atomic coordinate w" ]]
y = data["Calculated atomic coordinates u'"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy: ", accuracy)
print("Confusion matrix:\n", confusion_mat)
