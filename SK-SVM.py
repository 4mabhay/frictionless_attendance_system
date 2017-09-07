import config as cfg
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from time import time
import logging
from sklearn.model_selection import GridSearchCV


def train_SVC(train_data_npy,label_data_npy):
    model_file = cfg.model_file

    X_train, X_test, y_train, y_test = train_test_split( train_data_npy, label_data_npy, test_size=cfg.test_size, random_state=42)

    print " Train split size :: %d \n Test split size :: %d" % (X_train.shape[0],X_test.shape[0])


    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier = GridSearchCV(SVC(kernel='rbf',probability=True), param_grid)
    classifier = classifier.fit(X_train, y_train.ravel())
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(classifier.best_estimator_)


    # Now predict the value of the digit on the second half:
    predicted = classifier.predict(X_test)
    model_accuracy = np.mean(predicted==y_test.ravel())

    print "model accuracy is %.2f " % round(model_accuracy*100,2)

    pickle.dump(classifier,open(model_file, 'wb'))


if __name__=="__main__":
    import utilities as utl
    train_data_npy, label_data_npy,_ = utl.load_and_prepare_data()
    train_SVC(train_data_npy, label_data_npy)
