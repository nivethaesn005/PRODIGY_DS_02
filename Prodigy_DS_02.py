import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("\nSample data:\n", train.head())

# DATA CLEANING
print("\nMissing values before cleaning:\n", train.isnull().sum())

# Fill missing values
train['Age'].fillna(train['Age'].median(), inplace=True)
test['Age'].fillna(test['Age'].median(), inplace=True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)

for dataset in [train, test]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset.loc[dataset['FamilySize'] > 1, 'IsAlone'] = 0
    dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

print("\nData Cleaning Done Successfully!")
print("\nMissing values after cleaning:\n", train.isnull().sum())

title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)

drop_columns = ['Name', 'Ticket', 'PassengerId']
train.drop(columns=drop_columns, inplace=True)
test.drop(columns=['Name', 'Ticket'], inplace=True)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
print("\nFeatures selected for training:\n", features)

X = train[features]
y = train['Survived']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\nModel Training Completed!")

y_pred = model.predict(X_valid)

print("\nClassification Report:\n", classification_report(y_valid, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_valid, y_pred))
print("\nAccuracy Score:", round(accuracy_score(y_valid, y_pred)*100, 2), "%")

test_pred = model.predict(test[features])

submission = pd.DataFrame({
    "PassengerId": test['PassengerId'],
    "Survived": test_pred
})

submission.to_csv('hina_titanic_submission.csv', index=False)

numeric_train = train.select_dtypes(include=[np.number])

# Stacked Bar
pclass_survival = pd.crosstab(train['Pclass'], train['Survived'])

pclass_survival.plot(kind='bar', stacked=True, color=['red','green'], figsize=(8,6))
plt.title('Survival by Passenger Class - Stacked Bar')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.legend(['Did Not Survive','Survived'])
plt.show()

# Survival Count Pie Chart
survived = y.value_counts()
plt.figure(figsize=(6,6))
colors = ['#ff9999','#66b3ff']
plt.pie(survived, labels=['Not Survived', 'Survived'], colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Titanic Survival Rate', fontsize=16, color='navy')
plt.tight_layout()
plt.show()

# KDE Plot
plt.figure(figsize=(8,6))
sns.kdeplot(data=train, x='Age', hue='Survived', shade=True, palette="Set2")
plt.title('Age Distribution by Survival',fontsize=16, color='darkblue')
plt.xlabel('Age')
plt.ylabel('Density')
plt.show()

# Box Plot
plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y='Fare', data=train, palette="coolwarm")
plt.title('Fare Distribution by Passenger Class - Box Plot',fontsize=16, color='teal')
plt.show()

# Half Heatmap
plt.figure(figsize=(12,10))
corr = train.select_dtypes(include=[np.number]).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# Bar Plot
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=train)
plt.title('Survival Rate by Sex',fontsize=14, color='purple')
plt.show()

print("\n Titanic Data Cleaning, EDA, Modeling Completed by Nivetha! ")

