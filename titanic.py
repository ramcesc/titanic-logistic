#Regression on Kaggle's Titanic Competition

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

X_train = dataset_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y_train = dataset_train.iloc[:, 1]
X_test = dataset_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]


#Dealing with nan values in Embarked
X_train["Embarked"] = X_train["Embarked"].fillna("S")
X_train["Age"] = X_train["Age"].fillna(X_train["Age"].median())

X_train.loc[X_train["Sex"] == "male", "Sex"] = 0
X_train.loc[X_train["Sex"] == "female", "Sex"] = 1

X_train.loc[X_train["Embarked"] == "S", "Embarked"] = 0
X_train.loc[X_train["Embarked"] == "C", "Embarked"] = 1
X_train.loc[X_train["Embarked"] == "Q", "Embarked"] = 2

X_train = X_train.as_matrix()

#This is for X_test set

#Dealing with nan values in Embarked
X_test["Age"] = X_test["Age"].fillna(X_test["Age"].median())
X_test["Fare"] = X_test["Fare"].fillna(X_test["Fare"].median())

X_test.loc[X_test["Sex"] == "male", "Sex"] = 0
X_test.loc[X_test["Sex"] == "female", "Sex"] = 1

X_test.loc[X_test["Embarked"] == "S", "Embarked"] = 0
X_test.loc[X_test["Embarked"] == "C", "Embarked"] = 1
X_test.loc[X_test["Embarked"] == "Q", "Embarked"] = 2

X_test = X_test.as_matrix()

y_train = y_train.as_matrix()

#Using Logistic Regression
from sklearn.linear_model import LogisticRegression
lregressor = LogisticRegression(random_state=0)
lregressor.fit(X_train, y_train)

y_pred_log = lregressor.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": dataset_test["PassengerId"],
    "Survived": y_pred_log
})

submission.to_csv("log_titanic.csv", index=False)

