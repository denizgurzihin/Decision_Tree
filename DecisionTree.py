import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import *
from time import time

col_names = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13', 'att14', 'att15', 'att16','label']
# load dataset
DATA = pd.read_csv("Final_Data.csv", header=None, names=col_names)

#DATA.info()
feature_cols = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6', 'att7', 'att8', 'att9', 'att10', 'att11', 'att12', 'att13', 'att14', 'att15', 'att16']

X = DATA[feature_cols] # Features
y = DATA.label # Target variable

accuracy = 0
Confusion_Matrix = 0
f1 = 0
precision = 0
recall = 0
timee = 0

for x in range(10):
  test_start = time()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=x) # 80% training and 20% test

  # Create Decision Tree classifer object
  clf = DecisionTreeClassifier(criterion="entropy")

  # Train Decision Tree Classifer
  clf = clf.fit(X_train, y_train)

  #Predict the response for test dataset
  y_pred = clf.predict(X_test)
  
  accuracy = accuracy + accuracy_score(y_test, y_pred)
  f1 = f1 + f1_score(y_test, y_pred, average='weighted')
  precision = precision + precision_score(y_test, y_pred, average='weighted')
  recall = recall + recall_score(y_test, y_pred, average='weighted')
  Confusion_Matrix = Confusion_Matrix + confusion_matrix(y_test, y_pred)
  test_finish = time()
  timee = timee + test_finish-test_start

print("Overall accuracy for decision tree with gain ratio and hold out method(10 times) :", accuracy/10)
print("Overall f1_score for decision tree with gain ratio and hold out method(10 times) :", f1/10)
print("Overall precision_score for decision tree with gain ratio and hold out method(10 times) :", precision/10)
print("Overall recall_score for decision tree with gain ratio and hold out method(10 times) :", recall/10)

print("       Confusion Matrix : ")
print(Confusion_Matrix/10)

print("Overall time needed for decision tree with gain ratio and hold out method(10 times) :", ((timee/10)*10*10*10), " miliseconds")
