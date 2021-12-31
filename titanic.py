import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read in data
test_data = pd.read_csv("./test.csv")
train_data = pd.read_csv("./train.csv")

# Filter rows with missing values
train_data = train_data.dropna(axis=0)
test_data = test_data.dropna(axis=0)

# Choose target and features
y = train_data.Survived
features = ['Pclass', 'Age']
X = train_data[features]

# Split the data into training and validation data, for both feature and target
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 114514)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0)
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
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state = 114514)
final_model.fit(X, y)
test_X = test_data[features]
#print(test_X)

predictions = final_model.predict(test_X)
predictions = [round(num) for num in predictions]

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")