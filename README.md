# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
Step 1:
Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the simple linear regression model for predicting the marks scored.

Step 2:
Set variables for assigning dataset values and implement the .iloc module for slicing the values of the variables X and y.

Step 3:
Import the following modules for linear regression; from sklearn.model_selection import train_test_split and also from sklearn.linear_model import LinearRegression.

Step 4:
Assign the points for representing the points required for the fitting of the straight line in the graph.

Step 5:
Predict the regression of the straight line for marks by using the representation of the graph.

Step 6:
Compare the graphs (Training set, Testing set) and hence we obtained the simple linear regression model for predicting the marks scored using the given datas.

Step 7:
End the program.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored...
Developed by: A NAVEEN KUMAR
RegisterNumber: 212221240032
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset= pd.read_csv('student_scores.csv')
dataset.head()
X=dataset.iloc[:,:-1].values
X
y=dataset.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred
y_test 
plt.scatter(X_train,y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='cyan')
plt.title("Hour vs scores(Training set)")
#plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(X_test,y_test,color='cyan')
plt.plot(X_test,regressor.predict(X_test),color='orange')#plotting the regression line
plt.title("Hours vs scores(Testing set)")
#plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
 
*/
```

## Output:
![simple linear regression model for predicting the marks scored](out.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
