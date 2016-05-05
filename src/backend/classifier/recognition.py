from sklearn import linear_model, svm
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import random

DATA_PATH = "data/prepared_data/"
MODEL_PATH = "models/"

X = np.load(DATA_PATH + "X.npy")
y = np.load(DATA_PATH + "y.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier = linear_model.LogisticRegression(C=0.0001, solver="lbfgs", multi_class="multinomial")
#classifier = svm.SVC()
classifier.fit(X_train, y_train)
with open(MODEL_PATH + "logreg.pkl", 'wb') as f:
    pickle.dump(classifier, f)
print "Training set accuracy: %s" % metrics.accuracy_score(y_train, classifier.predict(X_train))
print "Test set accuracy: %s" % metrics.accuracy_score(y_test, classifier.predict(X_test))


def recognize(classifier, X, y, num):
    for _ in xrange(num):
        i = random.randint(0, y.shape[0] - 1)
        print "ground truth label: %s" % y[i]
        print "Predicted label: %s" % classifier.predict(X[i:i+1, :])[0]
        plt.imshow(X[i, :].reshape(20, 20), cmap="gray")
        plt.show()

if __name__ == "__main__":
    with open(MODEL_PATH + "logreg.pkl", 'rb') as f:
        logreg = pickle.load(f)
    X = np.load(DATA_PATH + "X.npy")
    y = np.load(DATA_PATH + "y.npy")
    #recognize(logreg, X, y, 50)