#import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
from sklearn import svm, tree
from sklearn.cross_validation import cross_val_score
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

# PCA data
X = raw_data[['SHOT_DIST', 'FINAL_MARGIN', 'PERIOD',
              'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST',
              'SHOT_NUMBER', 'TOUCH_TIME', 'PTS_TYPE', 'DEFENSE_LEVEL', 'OFFENSE_LEVEL']]
pca = decomposition.PCA(n_components=6)
pca.fit(X)
X = pca.transform(X)

tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(X, target)
dot_data = tree.export_graphviz(tree_clf, out_file=None,
                                class_names=["missed", "made"],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_PCA.pdf")