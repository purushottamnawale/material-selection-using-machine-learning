import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# Load the data into a pandas dataframe
df = pd.read_csv('materials_preprocessed.csv')

# Convert the Material column to one-hot encoded features
ohe = OneHotEncoder(sparse_output=False)
material_features = ohe.fit_transform(df['Material'].to_numpy().reshape(-1, 1))
material_feature_names = ohe.get_feature_names_out(['Material'])
material_df = pd.DataFrame(material_features, columns=material_feature_names)

# Combine the material features with the other features
X = pd.concat([material_df, df[['Su', 'Sy', 'E', 'G', 'mi', 'Ro']]], axis=1)

# Convert the label column to 0s and 1s
y = df['Use'].map({'Yes': 1, 'No': 0})

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Export the decision tree as a graphviz dot file
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)

# Display the decision tree
graph

graph = graphviz.Source(dot_data)
graph.format = 'png'
graph.render('decision_tree')