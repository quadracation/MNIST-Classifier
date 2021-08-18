from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



data = fetch_openml('mnist_784', data_home = "C:/Users/neb/Desktop/git")
model = DecisionTreeClassifier()


features= data.data 
labels = data.target

print(len(data.data), "is length of data")
print(len(data.target), "is length of labels")


X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size= 0.33, shuffle = True)


model.fit(X_train, y_train)
print("score:", model.score(X_test, y_test)) 





