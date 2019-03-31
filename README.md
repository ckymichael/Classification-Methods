# Classification-Methods
# 
### COMP 4331 Knowledge Discovery and Data
### Mining, Fall 2018
#### Major Tasks
##### This assignment consists of the following tasks:
###### • To acquire a better understanding of classification methods by using scikitlearn.
###### • To learn to use the ensemble method for classification.
###### • To learn to use a SVM model for classification.
###### • To learn to use a single-hidden-layer neural network model for classification.
###### • To conduct empirical study to compare several classification methods.
### Classification Methods
###### Implement the three classifiers and compare the performance achieved by different classifiers.
###### • Ensemble Method Apply Random Forest onto the dataset. You are required to evaluate the accuracy and record the training time.
###### • SVM For the SVM model with RBF kernel, the kernel parameter γ should be determined using cross validation. The generalization performance of the model is estimated for each candidate value of γ ∈ {1, 0.1, 0.01, 0.001}. This is done by 5-fold cross-validation on the training instances. The value γ that gives the best performance among the 4 choices of γ can then be found. Then, a SVM model with γ is trained from scratch using all the training instances available.
###### • Neurak Network For the single-hidden-layer neural network model, the number of hidden units H should be determined using cross validation. The generalization performance of the model is estimated for each candidate value of H ∈ {1, 5, 10, 20, 50}. You are required to adopt the same cross validation strategy as for SVM model with the candidate values of H ∈ {1, 5, 10, 20, 50}.
#### Data Description
###### One 10-classes classification data set which are available from https://github.com/comp4331fall18/sample-code-DM/tree/master/Assignment2-Dataset. The dataset consists of a training set of 10,000 examples and a test set of 1,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
