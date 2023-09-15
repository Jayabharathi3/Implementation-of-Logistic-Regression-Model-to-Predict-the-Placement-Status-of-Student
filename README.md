# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:

Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

3.Import LabelEncoder and encode the dataset.

4.Import LogisticRegression from sklearn and apply the model on the dataset.

5.Predict the values of array.

6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

7.Apply new unknown values 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JAYABHARATHI .S
RegisterNumber:  212222100013
*/


#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading the file
dataset = pd.read_csv('Placement_Data_Full_Class.csv')
dataset

dataset.head(20)

dataset.tail(20)

#droping tha serial no salary col
dataset = dataset.drop('sl_no',axis=1)
#dataset = dataset.drop('salary',axis=1)

dataset = dataset.drop('gender',axis=1)
dataset = dataset.drop('ssc_b',axis=1)
dataset = dataset.drop('hsc_b',axis=1)
dataset

dataset.shape

dataset.info()

#catgorising col for further labelling
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes

dataset.info()

dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset

dataset.info()

dataset

#selecting the features and labels
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2)
dataset.head()

x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.linear_model import LogisticRegression
clf= LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

clf.predict([[0, 87, 0, 95, 0, 2, 78, 2, 0]])
*/
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
DATASET:

dataset.head():


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/af7659b7-77cc-4625-a4ac-1ca852965978)


dataset.tail():


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/2ef000c1-10ad-405e-a1f7-bcbf0d7ef76d)


dataset after dropping:


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/05c72f71-7bc3-4cfc-858e-230f101cc88e)


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/adb68890-1c71-4296-b826-882dfb15a222)


datase.shape:


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/d9983b11-e4fa-468e-9299-eedab244faa0)


dataset.info()


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/c2a5bbf0-f3a4-4b60-a8ec-2f80e3280874)


dataset.dtypes


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/802058bd-9d34-44f0-acdc-6e0c3cc8200d)


dataset.info()


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/84dfa710-6bc3-4217-bb1f-2c6ac323eeeb)


dataset.codes


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/5448b084-4d35-452a-9099-ec5d18c1a40b)


selecting the features and labels


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/bfbe0844-9e37-4d5f-921e-5fe35e6c644e)


dataset.head()


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/b6e8bab8-6f67-439e-a7cf-ae7461e4d1fe)


x_train.shape


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/b6463aa0-cbf6-4257-bc79-70327ccf89a7)


x_test.shape


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/3148d17b-7205-4476-8a42-eab83da3f061)


y_train.shape


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/9f6ab8d7-75b4-4cf9-a921-1b65dd42a54a)


y_test.shape:


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/d360598c-8e23-4b37-87f7-715680c8c8eb)



![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/1adc57b2-8f2e-4929-9bfe-8416e978c875)


clf.predict

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/c6169a36-37c4-4c94-a5f9-f093e81c9da6)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
