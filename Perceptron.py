# Perceptron.py
# --------------
# Licensing Information:  You are free to use or extend this project for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to The Ohio State University, including a link to http://aritter.github.io/courses/5525_spring19.html
#
# Attribution Information: This assignment was developed at The Ohio State University
# by Alan Ritter (ritter.1492@osu.edu).

import sys

import numpy as np
from Eval import Eval

from imdb import IMDBdata

class Perceptron:
    def __init__(self, X, Y, N_ITERATIONS):
        self.N_ITERATIONS = N_ITERATIONS
        #TODO: Initalize parameters
        self.w = np.zeros(X.shape[1])
        self.b = 1
        self.wa = np.zeros(X.shape[1])
        self.ba = 1
        self.c = 1
        
        self.Train(X,Y)

    def ComputeAverageParameters(self):
        #TODO: Compute average parameters (do this part last)
        
        self.w = self.w - (self.wa / self.c)
        self.b = self.b - (self.ba / self.c)
        return

    def Train(self, X, Y):
        #TODO: Estimate perceptron parameters
        for i in range(self.N_ITERATIONS):
            for doc in range(X.shape[0]):
                #print(self.w, np.sum(self.w))
                if i == 0 and doc ==0:
                    a = X.getrow(doc).dot(self.w)
                else:
                    a = X.getrow(doc).dot(self.w.transpose())
                a = a[0] + self.b
                if a * Y[doc] < 0:
                    self.w = self.w + X.getrow(doc).multiply(Y[doc])
                    self.wa += self.c * X.getrow(doc).multiply(Y[doc])
                    self.b += int(Y[doc])
                    self.ba += self.c * int(Y[doc])
                self.c += 1
            
        return

    def Predict(self, X):
        #TODO: Implement perceptron classification
        predictions = np.zeros(X.shape[0])
        for x in range(X.shape[0]):
            a = X.getrow(x).dot(self.w.transpose())
            a += self.b
            if a > 0:
                predictions[x] = 1.0
            else:
                predictions[x] = -1.0
        return predictions

    def Eval(self, X_test, Y_test):
        Y_pred = self.Predict(X_test)
        ev = Eval(Y_pred, Y_test)
        return ev.Accuracy()

    def getW(self):
        return self.w

if __name__ == "__main__":
    train = IMDBdata("%s/train" % sys.argv[1])
    test  = IMDBdata("%s/dev" % sys.argv[1], vocab=train.vocab)
    
    ptron = Perceptron(train.X, train.Y, int(sys.argv[2]))
    ptron.ComputeAverageParameters()
    print(ptron.Eval(test.X, test.Y))

    #TODO: Print out the 20 most positive and 20 most negative words
    w = ptron.getW()
    w = np.array(w)[0]
    indices = np.argsort(w)
    for x in indices[:-20:-1]:
        print(train.vocab.GetWord(x), w[x])
    for x in indices[:20]:
        print(train.vocab.GetWord(x), w[x])
        
    
    
