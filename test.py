# Import required libraries
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import pandas as pd
import pydotplus
from sklearn.model_selection import cross_val_score


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Data.csv')

# Merge the first three columns using string concatenation
df['Material'] = df[['Std', 'Material', 'Heat treatment']].fillna('').agg(' '.join, axis=1)

# Remove any string values from Sy column
df['Sy'] = df['Sy'].str.replace(' max', '').astype(int)

# Drop the unnecessary columns
df.drop(['Std','ID', 'Heat treatment', 'Desc','A5','Bhn','pH','Desc','HV'], axis=1, inplace=True)

# Add the 'Use' column based on specific conditions
df['Use'] = (
    (df['Su'].between(292, 683)) &
    (df['Sy'].between(212, 494)) &
    (df['E'].between(196650, 217350)) &
    (df['G'].between(47400, 110600)) &
    (df['mu'].between(0.225, 0.375)) &
    (df['Ro'].between(6288, 9432))
).map({True: 'Yes', False: 'No'})

# Insert the 'Use' column at the second position
df.insert(7, 'Use', df.pop('Use'))

# # Write the updated data to a new file
df.to_csv('Data Use.csv', index=False)

# Separate the features (X) and labels (y)
X = df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro']]
y = df['Use']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a decision tree classifier on the training data
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

# Use cross-validation to evaluate the model
scores = cross_val_score(dtc, X, y, cv=10)
print('Cross-validation scores:', scores)
print('Average cross-validation score:', scores.mean())

# Calculate the accuracy of the model on the training data
train_pred = dtc.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)

# Calculate the accuracy of the model on the testing data
test_pred = dtc.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

# Calculate other metrics related to the decision tree
precision = precision_score(y_test, test_pred, pos_label='Yes')
recall = recall_score(y_test, test_pred, pos_label='Yes')
f1 = f1_score(y_test, test_pred, pos_label='Yes')

# Print the metrics related to the decision tree
print('Training Accuracy:', train_acc)
print('Testing Accuracy:', test_acc)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)

# Generate the graphviz representation of the decision tree
dot_data = export_graphviz(dtc, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, out_file=None)

# Render the graph as a PNG image
graph = graphviz.Source(dot_data, format='png')
graph.render('decision_tree', view=True)
graph
