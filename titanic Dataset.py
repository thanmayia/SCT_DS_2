import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Download the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
response = requests.get(url)
with open("titanic.csv", "wb") as f:
    f.write(response.content)

titanic = pd.read_csv("titanic.csv")
titanic.head()

titanic.isnull().sum()

titanic['Age'].fillna(titanic['Age'].median(), inplace=True)
titanic['Embarked'].fillna(titanic['Embarked'].mode()[0], inplace=True)
titanic.drop(columns=['Cabin', 'Name', 'Ticket'], inplace=True)

titanic['Sex'] = titanic['Sex'].map({'male':0, 'female':1})
titanic['Embarked'] = titanic['Embarked'].map({'S':0, 'C':1, 'Q':2})

sns.countplot(x='Survived', data=titanic)
plt.title("Survival Count")
plt.show()

sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title("Survival Rate by Gender")
plt.show()

sns.histplot(titanic['Age'], bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title("Survival Rate by Passenger Class")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()