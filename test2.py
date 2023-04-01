# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('Mat.csv')

# Split the dataset into training and testing sets
X = data.drop('Material', axis=1)
y = data['Material']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier on the training set
clf.fit(X_train, y_train)

# Test the accuracy of the model on the testing set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Use the trained model to predict the best material for a given application
new_data = pd.DataFrame({'Density': [1.2], 'Tensile Strength': [500], 'Thermal Conductivity': [0.5]})
predicted_material = clf.predict(new_data)
print('Predicted Material:', predicted_material)
