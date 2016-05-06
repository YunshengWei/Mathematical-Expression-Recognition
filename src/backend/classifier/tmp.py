import convnet
import cPickle as pickle
import random
import matplotlib.pyplot as plt
import numpy as np

nn = convnet.load_model("models/convnet/convnet.ckpt")
with open('data/prepared_data/CROHME.pkl', 'rb') as f:
    CROHME = pickle.load(f)
train, val, test = CROHME['train'], CROHME['val'], CROHME['test']

X, y = test[0], test[1]
for _ in xrange(1000):
    i = random.randint(0, y.shape[0] - 1)
    gtl = CROHME['num2sym'][np.argmax(y[i])]
    if gtl != 'w':
        continue
    print "ground truth label: %s" % gtl
    print "Predicted label: %s" % CROHME['num2sym'][nn.predict(X[i])[0]]
    plt.imshow(X[i].reshape(28, 28), cmap="gray")
    plt.show()