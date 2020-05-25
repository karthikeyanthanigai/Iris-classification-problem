
#<-------------logistic regression------------->
#Import the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the Dataset
Dataset = pd.read_csv("iris_dataset.csv")
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:, -1].values


#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Logistic Regression model. 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=1000,random_state=0)
classifier.fit(X_train, y_train)

#predict our test set
y_pred = classifier.predict(X_test)


#evaluate our test set
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
acc_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred, average= 'weighted')

print("Test set results")
print("ACCURACY for test set(logistic regression method)",acc_test)
print("F1 SCORE for test set(logistic regression method)",f1_test)



#<-------------Random forest-------------->
#Import the Dataset
Dataset = pd.read_csv("iris_dataset.csv")
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:, -1].values


#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=32)


#Random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators=10,random_state=0)
classifier1.fit(X_train, y_train)

#predict our test set
y_pred1 = classifier1.predict(X_test)


#evaluate our test set
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

acc_test1 = accuracy_score(y_test, y_pred1)
f1_test1 = f1_score(y_test, y_pred1, average= 'weighted')

print("Test set results")
print("ACCURACY for test set(Random forest method)",acc_test1)
print("F1 SCORE for test set(Random forest method)",f1_test1)


#<-------------AdaBoost------------->
#Import the Dataset
Dataset = pd.read_csv("iris_dataset.csv")
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:, -1].values

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=32)

#Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
classifier2 = AdaBoostClassifier(n_estimators=10,random_state=0)
classifier2.fit(X_train, y_train)

#predict our test set
y_pred2 = classifier2.predict(X_test)


#evaluate our test set
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

acc_test2 = accuracy_score(y_test, y_pred2)
f1_test2 = f1_score(y_test, y_pred2, average= 'weighted')

print("Test set results")
print("ACCURACY for test set(Adaboost classifier method)",acc_test2)
print("F1 SCORE for test set(Adaboost classifier method)",f1_test2)
