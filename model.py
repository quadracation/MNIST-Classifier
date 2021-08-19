from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



data = fetch_openml('mnist_784', data_home = "C:/Users/neb/Desktop/git/")
model = make_pipeline(StandardScaler(), SVC(gamma= 'auto'))

print("Data Loaded\n")


X= data.data 
y = data.target

print("length of data arr: ",len(X))
print("length of label arr: ",len(y))
print("shape of data X,y", X.shape, y.shape )


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, shuffle = True)



model.fit(X_train, y_train)
print("score:", model.score(X_test, y_test)) 





