# Import required libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import graphviz

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Data.csv')

# Merge the first three columns using string concatenation
df['Material'] = df[['Std', 'Material', 'Heat treatment']].fillna('').agg(' '.join, axis=1)

# Remove any string values from Sy column
df['Sy'] = df['Sy'].str.replace(' max', '').astype(int)

# Drop the unnecessary columns
df.drop(['Std','ID', 'Heat treatment', 'Desc','A5','Bhn','pH','Desc','HV'], axis=1, inplace=True)

# Define the rating function
def get_rating(row):
    if (438.3 <= row['Su'] <= 535.7 and
        318.6 <= row['Sy'] <= 389.4 and
        204930 <= row['E'] <= 209070 and
        71100 <= row['G'] <= 86900 and
        0.285 <= row['mu'] <= 0.315 and
            7467 <= row['Ro'] <= 8253):
        return 5
    elif (389.6 <= row['Su'] <= 584.4 and
          283.2 <= row['Sy'] <= 424.8 and
          202860 <= row['E'] <= 211140 and
          63200 <= row['G'] <= 94800 and
          0.27 <= row['mu'] <= 0.33 and
          7074 <= row['Ro'] <= 8646):
        return 4
    elif (340.9 <= row['Su'] <= 633.1 and
          247.8 <= row['Sy'] <= 460.2 and
          200790 <= row['E'] <= 213210 and
          55300 <= row['G'] <= 102700 and
          0.255 <= row['mu'] <= 0.345 and
          6681 <= row['Ro'] <= 9039):
        return 3
    elif (292.2 <= row['Su'] <= 681.8 and
          212.4 <= row['Sy'] <= 495.6 and
          198720 <= row['E'] <= 215280 and
          47400 <= row['G'] <= 110600 and
          0.24 <= row['mu'] <= 0.36 and
          6288 <= row['Ro'] <= 9432):
        return 2
    else:
        return 1


# Calculate the rating for each row
df['rating'] = df.apply(get_rating, axis=1)

# Save the results to a new file data.csv
df.to_csv('Data Rating.csv', index=False)


# Separate the features and label columns
X = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Train a random forest classifier with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Predict the rating of a new material
new_material = np.array([440, 325, 207000, 79000, 0.3, 7860]).reshape(1, -1)
new_rating = rf.predict(new_material)
print("Predicted rating of the new material:", new_rating)

# Visualize the decision trees in the random forest
dot_data = export_graphviz(rf.estimators_[0], out_file=None, 
                           feature_names=['Su', 'Sy', 'E', 'G', 'mu', 'Ro'],
                           class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True)

# graph = graphviz.Source(dot_data)
# graph.render('decision_tree')  # Save the visualization as a pdf file

# Render the graph as a PNG image
graph = graphviz.Source(dot_data, format='png')
graph.render('decision_tree', view=True)
