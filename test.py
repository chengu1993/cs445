#import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from sklearn import svm, tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import pydotplus
#from sklearn.externals.six import StringIO
#from IPython.display import Image
#import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import decomposition

# import data
raw_data = pd.read_csv('out.csv', header=0)

# target data set
target = raw_data['FGM']

# train data set
train_data = raw_data[['SHOT_DIST', 'Loc',
                       'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST', 'DEFENSE_LEVEL', 'OFFENSE_LEVEL']]

# PCA data
X = raw_data[['SHOT_DIST', 'Loc', 'PERIOD',
              'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST',
              'SHOT_NUMBER', 'TOUCH_TIME', 'PTS_TYPE', 'DEFENSE_LEVEL', 'OFFENSE_LEVEL']]
pca = decomposition.PCA(n_components=7)
pca.fit(X)
X = pca.transform(X)



tree_clf = tree.DecisionTreeClassifier(min_impurity_split=0.45, max_depth=10, min_samples_split=5000, min_samples_leaf=50)

# Test decision tree classifier model
tree_scores = cross_val_score(tree_clf, X, target, cv=10)

print("Score of Decision Tree:")
print(tree_scores)
print(tree_scores.mean())

rf_clf = RandomForestClassifier(min_impurity_split=0.45, max_depth=10, min_samples_split=5000, min_samples_leaf=50)
rf_scores = cross_val_score(rf_clf, X, target, cv=10)

print("Score of Random Forest:")
print(rf_scores)
print(rf_scores.mean())

# tree_clf = tree_clf.fit(train_data, target)
#
#
# dot_data = tree.export_graphviz(tree_clf, out_file=None,
#                                 feature_names=list(train_data.columns.values),
#                                 class_names=["missed", "made"],
#                                 filled=True, rounded=True,
#                                 special_characters=True)
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("decision_tree.pdf")