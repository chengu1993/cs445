import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, tree
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image

# import data
raw_data = pd.read_csv('modified_data.csv', header=0)

# target data set
target = raw_data['FGM']

# train data set
train_data = raw_data[['SHOT_DIST', 'TOUCH_TIME', 'FINAL_MARGIN', 'PERIOD',
                       'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST']]

# # SVM Classifier definition
# svm_clf = svm.SVC()
#
# # Test svm classifier model
# svm_scores = cross_val_score(svm_clf, train_data, target, cv=10)
#
# # Decision tree classifier definition
tree_clf = tree.DecisionTreeClassifier()
#
# # Test decision tree classifier model
# tree_scores = cross_val_score(tree_clf, train_data, target, cv=10)
#
# # Naive Bayes Classifier definition
# gnb = GaussianNB()
#
# # Test Naive Bayes classifier model
# gnb_scores = cross_val_score(gnb, train_data, target, cv=10)
#
# print(svm_scores)
#
# print(tree_scores)
#
# print(gnb_scores)
tree_clf = tree_clf.fit(train_data[:20], target[:20])
dot_data = tree.export_graphviz(tree_clf, out_file=None,
                                feature_names=list(train_data.columns.values),
                                class_names=["missed", "made"],
                                filled=True, rounded=True,
                                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris.pdf")
# dot_data = StringIO()
# tree.export_graphviz(tree_clf, out_file=dot_data, feature_names=X.columns)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_png("decision_tree.png")

# Test result
# SVM scores
# [ 0.6036852   0.60103061  0.59806371  0.60134291  0.5950652   0.60580978
#   0.59815711  0.60135874  0.59472122  0.59401843]

# decision tree scores
# [ 0.54044347  0.53864772  0.53810119  0.54247345  0.5354884   0.545057
#   0.54052788  0.54021552  0.54044979  0.53506169]
# Naive Bayes scores
# [ 0.59775141  0.59423798  0.59353529  0.59814179  0.59420629  0.59831329
#   0.58488209  0.59003592  0.59237857  0.59503358]
