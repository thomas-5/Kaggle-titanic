import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read in data
test_data = pd.read_csv("./test.csv")
train_data = pd.read_csv("./train.csv")
# Data cleaninig
# Fill the missing Age with the average Age in train_data
train_age_mean = train_data.dropna(axis=0).Age.mean()
train_data.Age = train_data.Age.fillna(value = train_age_mean)
# Turn Sex into binary variate
gender = pd.get_dummies(train_data["Sex"])
train_data = pd.concat((train_data, gender), axis=1)
train_data = train_data.drop(["Sex", "female"], axis=1)
train_data = train_data.rename(columns={"male":"Sex"})

# Do the same for test_data
test_age_mean = test_data.dropna(axis=0).Age.mean()
test_data.Age = test_data.Age.fillna(value = test_age_mean)
test_gender = pd.get_dummies(test_data["Sex"])
test_data = pd.concat((test_data, test_gender), axis=1)
test_data = test_data.drop(["Sex", "female"], axis=1)
test_data = test_data.rename(columns={"male":"Sex"})

fare_mean = test_data.dropna(axis=0).Fare.mean()
test_data.Fare = test_data.Fare.fillna(value = fare_mean)

#Choose target and features
y = train_data.Survived
features = ['Age', 'Pclass', 'Sex', 'Fare']
X = train_data[features]

# Train the model
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state= 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

best_tree_size = 10
min_mae = get_mae(10, train_X, val_X, train_y, val_y)
for node_number in range(10, 200, 10):
    cur_mae = get_mae(node_number, train_X, val_X, train_y, val_y)
    if (cur_mae < min_mae):
        min_mae = cur_mae
        best_tree_size = node_number
    
# Define & fit the model and make prediction
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state=0)
final_model.fit(X, y)
test_X = test_data[features]
predictions = final_model.predict(test_X)
predictions = [round(num) for num in predictions]

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")