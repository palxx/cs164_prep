# Import the necessary modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read in the file fg_attempt.csv
iris_dataset = pd.read_csv('iris_data.csv')

print (f"{iris_dataset.shape}")
x1Name = 'petal_length'
x2Name = 'petal_width'
x3Name = 'sepal_length'
x4Name = 'sepal_width'

# Select the two features you want
X = iris_dataset[[x1Name, x2Name]]

# Create a dataframe y containing 'class'
y = iris_dataset[['class']]

# Flatten y into an array
yArray = np.ravel(y)

# Initialize a LogisticRegression() model
logisticModel = LogisticRegression()

# Fit the model
logisticModel.fit(X, yArray)

# Input feature values for a sample instance

x1 = float(input())
x2 = float(input())

# Create a new dataframe with user-input Distance and ScoreDiffPreKick
XNew = pd.DataFrame([{x1Name: x1, x2Name: x2}])

# Predict the outcome from the new data
pred = logisticModel.predict(XNew)
print(pred)

# Determine the accuracy of the model logisticModel
score = logisticModel.score(X, yArray)
print(f"{score:.2f}")

