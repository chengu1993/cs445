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

# train data set
train_data = raw_data[['SHOT_DIST', 'FINAL_MARGIN', 'PERIOD',
                       'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST', 'DEFENSE_LEVEL', 'OFFENSE_LEVEL']]

tree_clf = tree.DecisionTreeClassifier()
tree_clf = tree_clf.fit(train_data, target)
dot_data = tree.export_graphviz(tree_clf, out_file=None,
                                feature_names=list(train_data.columns.values),
                                class_names=["missed", "made"],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_data.pdf")