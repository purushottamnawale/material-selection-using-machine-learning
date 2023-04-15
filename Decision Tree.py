from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pandas as pd

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
    (df['Su'].between(336, 505)) &
    (df['Sy'].between(251, 376)) &
    (df['E'].between(165000, 248000)) &
    (df['G'].between(63000, 94000)) &
    (df['mu'].between(0.24, 0.36)) &
    (df['Ro'].between(6200, 9400))
).map({True: 'True', False: 'False'})

# Insert the 'Use' column at the second position
df.insert(7, 'Use', df.pop('Use'))

# Write the updated data to a new file
df.to_csv('material.csv', index=False)


# Separate the features (X) and labels (y)
X = df[['Su', 'Sy', 'E', 'G', 'mu', 'Ro']]
y = df['Use']

# Fit a decision tree classifier on the data
dtc = DecisionTreeClassifier()
dtc.fit(X, y)

# Generate the graphviz representation of the decision tree
dot_data = export_graphviz(dtc, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, out_file=None)

# Render the graph as a PNG image
graph = graphviz.Source(dot_data, format='png')
graph.render('decision_tree', view=True)



