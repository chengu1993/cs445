import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.cross_validation import cross_val_score

# import data
raw_data = pd.read_csv('modified_data.csv', header=0)

# target data set
target = raw_data['FGM']

# train data set
train_data = raw_data[['SHOT_DIST', 'TOUCH_TIME', 'FINAL_MARGIN', 'PERIOD',
                       'SHOT_CLOCK', 'DRIBBLES', 'CLOSE_DEF_DIST', 'PTS']]

# Classifier definition
clf = svm.SVC()

# Train classifier model
scores = cross_val_score(clf, train_data, target, cv=10)

# prediction
print(scores)
