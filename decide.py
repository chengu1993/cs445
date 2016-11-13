import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, tree
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB

# import data
raw_data = pd.read_csv('modified_data.csv', header=0)

# target data set
target = raw_data['FGM']

# train data set
train_data = raw_data[['SHOT_DIST', 'TOUCH_TIME', 'FINAL_MARGIN', 'PERIOD',
                       'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST', 'PTS']]

# SVM Classifier definition
svm_clf = svm.SVC()

# Test svm classifier model
svm_scores = cross_val_score(svm_clf, train_data, target, cv=10)

# Decision tree classifier definition
tree_clf = tree.DecisionTreeClassifier()

# Test decision tree classifier model
tree_scores = cross_val_score(tree_clf, train_data, target, cv=10)

# Naive Bayes Classifier definition
gnb = GaussianNB()

# Test Naive Bayes classifier model
gnb_scores = cross_val_score(gnb, train_data, target, cv=10)

print(svm_scores)

print(tree_scores)

print(gnb_scores)

# Test result
# SVM scores
# [ 0.99234853  0.99164585  0.98797626  0.99086508  0.99398766  0.99250351
#  0.99406528  0.99351866  0.99211307  0.99023895]

# decision tree scores
# [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]

# Naive Bayes scores
# [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]


