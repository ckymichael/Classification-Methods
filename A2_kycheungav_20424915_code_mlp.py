import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time




from scipy.io import loadmat

start_time = time.time()
#import data
test_images =loadmat('test_images.mat').get("test_images")
test_labels =loadmat('test_labels.mat').get("test_labels")
train_images =loadmat('train_images.mat').get("train_images")
train_labels =loadmat('train_labels.mat').get("train_labels")

test_labels =test_labels.ravel()
train_labels =train_labels.ravel()


params = {'hidden_layer_sizes' : [1, 5, 10, 20, 50]}

grid = GridSearchCV(MLPClassifier(solver='sgd'), param_grid = params, cv = 5) 
grid.fit(train_images,train_labels)
print("The best classifier is: ", grid.best_estimator_)

print('param_hidden_layer_sizes:\t', grid.cv_results_['param_hidden_layer_sizes'])
print('mean_test_score:\t\t', grid.cv_results_['mean_test_score'])
print('std_test_score:\t\t\t', grid.cv_results_['std_test_score'])
print('mean_train_score:\t\t', grid.cv_results_['mean_train_score'])
print('std_train_score\t\t\t', grid.cv_results_['std_train_score'])

classifier = grid.best_estimator_
classifier.fit(train_images,train_labels)

expected = train_labels
predicted = classifier.predict(train_images)
print("train accruacy score is", accuracy_score(expected, predicted))
expected = test_labels
predicted = classifier.predict(test_images)
print("test accruacy score is", accuracy_score(expected, predicted))



print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("--- %s seconds ---" % (time.time() - start_time))