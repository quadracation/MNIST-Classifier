from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd



data = fetch_openml('mnist_784')
model = DecisionTreeClassifier()


features= data.data 
labels = data.target

print(len(data.data), "is length of data")
print(len(data.target), "is length of labels")






