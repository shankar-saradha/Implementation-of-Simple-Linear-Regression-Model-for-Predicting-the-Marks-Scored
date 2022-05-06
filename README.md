# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```Python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile 
df.head()

df.tail()

# Segregating data to variables
X = df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
#displaying predicted values
Y_pred
#displaying actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="pink")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="yellow") 
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

## Output:
##Contents in the data file (head, tail):
![o1](https://user-images.githubusercontent.com/93978702/167156888-e5e344df-fffe-429d-936d-c12cc5749bf9.png)
![o2](https://user-images.githubusercontent.com/93978702/167156908-f107d2b5-16eb-4fa2-bd68-678cb8f205a7.png)

##Graph for Training Data:
![o6](https://user-images.githubusercontent.com/93978702/167156989-87846b51-c506-4fdd-b310-4acc62b98e99.png)

##Graph for Test Data:
![o7](https://user-images.githubusercontent.com/93978702/167157051-58271945-807d-42dc-b84a-779618cbc89a.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
