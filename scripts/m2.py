import pandas as pd
from sklearn import svm, tree
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
import pydotplus
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
# SVM Classifier definition
svm_clf = svm.SVC(C=2.0, kernel='rbf')

# Test svm classifier model
svm_scores = cross_val_score(svm_clf, X, target, cv=10)

print("Score of SVM with RBF:")
print(svm_scores)
print(svm_scores.mean())

# SVM Classifier definition
svm_poly_clf = svm.SVC(C=2.0, kernel='poly')

# Test svm classifier model
svm_poly_scores = cross_val_score(svm_poly_clf, X, target, cv=10)

print("Score of SVM with RBF:")
print(svm_poly_scores)
print(svm_poly_scores.mean())

# Decision tree classifier definition
tree_clf = tree.DecisionTreeClassifier()

# Test decision tree classifier model
tree_scores = cross_val_score(tree_clf, X, target, cv=10)

print("Score of Decision Tree:")
print(tree_scores)
print(tree_scores.mean())

# Naive Bayes Classifier definition
gnb = GaussianNB()

# Test Naive Bayes classifier model
gnb_scores = cross_val_score(gnb, X, target, cv=10)

print("Score of Naive Bayes:")
print(gnb_scores)
print(gnb_scores.mean())

rf_clf = RandomForestClassifier()
rf_scores = cross_val_score(rf_clf, X, target, cv=10)

print("Score of Random Forest:")
print(rf_scores)
print(rf_scores.mean())

