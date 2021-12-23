import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read in data
test = pd.read_csv("./test.csv")
train = pd.read_csv("./train.csv")
gender_submission = pd.read_csv("./gender_submission.csv")


print(test.head(5))
print(train.head(5))
print(gender_submission.head(5))