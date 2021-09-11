from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd



## use openml to fetch dataset, presplit between data and target 
data = fetch_openml('mnist_784')
model = make_pipeline(StandardScaler(), SVC(gamma= 'auto'))

print("Data Loaded\n")


X= data.data 
y = data.target

##Double checking number of samples and shape of object
print("length of data arr: ",len(X))
print("length of label arr: ",len(y))
##Should be 784 features
print("shape of data X,y", X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.33, shuffle = True)



model.fit(X_train, y_train)
#returns y_pred as an array
predict = model.predict(X_test)

#Performing f1_score calculation prior to cross validation
f1 = f1_score(y_test, predict, average = "micro")

print("F1 Score with test size = 23100:", f1)

##Cross Validating data and keeping metrics
scores = cross_val_score(model, X, y, scoring= 'f1_micro')


#printing scores

print("Avg F1 After 5 fold validation: ", np.mean(scores))




