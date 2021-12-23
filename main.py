import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# The goal: we want to find patterns in train.csv that help us predict whether the passengers in test.csv survived.

# Read in data
test = pd.read_csv("./test.csv")
train = pd.read_csv("./train.csv")
gender_submission = pd.read_csv("./gender_submission.csv")

# The code below calculates the percentage of female passengers (in train.csv) who survived
women = train.loc[train.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

# The code below calculates the percentage of male passengers (in train.csv) who survived
men = train.loc[train.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

print("Helloworld!")

# y = train["Survived"]

# features = ["Pclass", "Sex", "SibSp", "Parch"]
# X = pd.get_dummies(train[features])
# X_test = pd.get_dummies(test[features])

# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# model.fit(X, y)
# predictions = model.predict(X_test)

# output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")
