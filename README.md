# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BEJIN B
RegisterNumber:  212222230021
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:


df.head()

![ML21](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/ccd41780-f45b-4554-91f0-56a7367eec8e)

df.tail()

![ML22](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/752023fb-7ab0-4cff-8354-4275eb9e50ce)

Array value of X

![ML23](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/4d43dd30-097f-4cc2-8031-c74fb58c224f)

Array value of Y

![ML24](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/575321f6-db38-4e79-848a-3f64a84db0b9)

Values of Y prediction

![ML25](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/8553ccf9-272b-4681-9ab2-618753789a82)

Array values of Y test

![ML26](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/a3c944f0-a250-4f9a-9e4b-4271d5436655)

Training Set Graph

![ML27](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/6f261eef-062d-4e5c-93e3-bb32304944ff)

Test Set Graph

![ML28](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/dee270a1-ec27-4da4-9689-758d653e2433)

Values of MSE, MAE and RMSE

![ML29](https://github.com/BejinB/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118367518/4e7cecd4-1646-42f1-9162-911f3eb5cf24)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
