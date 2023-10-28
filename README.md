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
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JAYABHARATHI .S
RegisterNumber:  212222100013
*/


#import modules
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

1.Placement Data
![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/a664d351-5452-489e-8323-6f8642b2bd65)


2.Salary Data


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/e79158e0-cde1-4ac3-ad7b-741fab7bf6b8)



3. Checking the null function()


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/3177a6d9-bdcb-4c56-b689-ce60f5114072)



4.Data Duplicate

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/536c075a-aae4-4f13-ac61-be9986936eb7)



5.Print Data

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/cc12a379-bdc9-4046-86de-8fb0812aa5e7)

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/cc3ee47b-5b2b-4b0c-a87a-d13f4f541962)


6.Data Status

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/11bd5866-a98f-4fad-837c-919c022a10f2)


7.y_prediction array

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/f388a783-bbe2-47cf-abb2-1b4cb9d92267)


8.Accuracy value


![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/29072677-493f-4f5e-ba2b-08a24d8cd96d)



9.Confusion matrix

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/15f2f330-fcf1-4bc2-88fc-f93337af126e)



10.Classification Report

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/9ef17dab-bd87-48e9-b2c6-1ee85027abdc)



11.Prediction of LR

![image](https://github.com/Jayabharathi3/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120367796/3feb3940-8814-4bf3-a8a8-74356556ad6e)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
