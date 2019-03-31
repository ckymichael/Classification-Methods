# import data
from scipy.io import loadmat
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import time

import matplotlib.pylab as plt

start_time = time.time()

test_images =loadmat('test_images.mat').get("test_images")
test_labels =loadmat('test_labels.mat').get("test_labels")
train_images =loadmat('train_images.mat').get("train_images")
train_labels =loadmat('train_labels.mat').get("train_labels")


y = train_labels.reshape(-1,)
X = train_images
y_test = test_labels.reshape(-1,)
X_test = test_images


classifier = RandomForestClassifier(oob_score=True, random_state=10)
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