# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, separate features (X) and labels (Y), and split the data into training and testing sets.
2. Train a LinearRegression model using the training data.
3. Predict the scores for test data and visualize the regression line on both training and test sets.
4. Evaluate the model using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Program:

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Venkatachalam S
RegisterNumber:  212224220121

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("/content/student_scores.csv")

df.head()

df.tail()

X = df.iloc[:,:-1].values
X

Y = df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,Y_train)
Y_pred = reg.predict(X_test)
Y_pred

Y_test

plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

mse = mean_squared_error(Y_test,Y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/1726b925-96d9-4c8c-8f51-f0063aa17459)
![image](https://github.com/user-attachments/assets/1c047017-2d05-4c58-979d-9c9381caac0a)
![image](https://github.com/user-attachments/assets/d66989de-d502-403d-837a-bec480d742fa)
![image](https://github.com/user-attachments/assets/13633238-eec9-44c3-9c2b-0db13afb522a)
![image](https://github.com/user-attachments/assets/c3d078bd-f029-41fa-9918-eff94f96f43b)
![image](https://github.com/user-attachments/assets/ec57a22a-1ee2-46bb-9074-a0101b696baf)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
