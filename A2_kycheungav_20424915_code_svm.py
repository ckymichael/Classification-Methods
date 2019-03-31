import numpy as np
import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from scipy.io import loadmat

#import data
test_images =loadmat('test_images.mat').get("test_images")
test_labels =loadmat('test_labels.mat').get("test_labels")
train_images =loadmat('train_images.mat').get("train_images")
train_labels =loadmat('train_labels.mat').get("train_labels")


y = train_labels.reshape(-1,)
X = train_images
y_test = test_labels.reshape(-1,)
X_test = test_images

start_time = time.time()
params = {'gamma' : [1E-3,1E-2,1E-1,1]}
grid = GridSearchCV(SVC(kernel='rbf', random_state=10) ,param_grid=params, cv=5)
grid.fit(X, y)
print("The best classifier is: ", grid.best_estimator_)


print('param_gamma:\t\t', grid.cv_results_['param_gamma'])
print('mean_test_score:\t', grid.cv_results_['mean_test_score'])
print('std_test_score:\t\t', grid.cv_results_['std_test_score'])
print('mean_train_score:\t', grid.cv_results_['mean_train_score'])
print('std_train_score\t\t', grid.cv_results_['std_train_score'])


classifier = grid.best_estimator_

"""
classifier = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""
classifier.fit(X,y)

expected = y
predicted = classifier.predict(X)
print("train accruacy score is", accuracy_score(expected, predicted))

expected = y_test
predicted = classifier.predict(X_test)
print("test accruacy score is", accuracy_score(expected, predicted))


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("--- %s seconds ---" % (time.time() - start_time))