#!/usr/bin/python
import numpy as np
import csv
from sklearn.svm import NuSVR
from sklearn.externals import joblib
import random
import time

def read_csv(csv_file):
    labels = [] # [[name, score]]
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            labels.append(float(row[1]))
            #labels.append([row[0],float(row[1])])
    return labels

csv_file = 'VSD_dataset.csv'
#K = joblib.load('K-tmp.pkl')
K = joblib.load('feature/K.pkl')
print(K.shape)
numSamples = K.shape[0] # (n_samples, n_samples)
labels = np.array(read_csv(csv_file)[:numSamples])
#assert(len(labels)==numSamples)

# Split data: train/val/test=2/1/1
random.seed(1)
p = range(numSamples)
random.shuffle(p)
half = int(numSamples/2)
fourth = int(numSamples/4)

trainIdx = p[0:half]
valIdx = p[half: half + fourth]
testIdx = p[half + fourth:numSamples]
# scores
trainLabels = labels[trainIdx]
valLabels = labels[valIdx]
testLabels = labels[testIdx]
# kernels
Kx = K[testIdx][:, trainIdx]
Kv = K[valIdx][:, trainIdx]
Kt = K[trainIdx][:, trainIdx]

#n = len(trainIdx)
#nv = len(valIdx)
#nx = len(testIdx)

#Train Support Vector Regression
# C = 10.^(-2:1:2);
C = [0.1]
for c in C:
    print("C = %f"%c)
    tic = time.time()
    svr = NuSVR(C=c, kernel='precomputed')
    svr.fit(Kt, trainLabels)
    toc = time.time()
    print("train cost %f s"%(toc-tic))
    trainScores = svr.predict(Kt)
    mseTrain = np.mean((trainLabels - trainScores) ** 2)
    valScores = svr.predict(Kv)
    mseVal = np.mean((valLabels - valScores) ** 2)
    testScores = svr.predict(Kx)
    mseTest = np.mean((testLabels - testScores) ** 2)
    print('Train MSE : %g'%mseTrain)
    print('val MSE : %g'%mseVal)
    print('Test MSE : %g'%mseTest)

    # use all samples to train
    svr = NuSVR(C=c, kernel='precomputed')
    svr.fit(K, labels)
    joblib.dump(svr, 'svr.pkl', compress=3)
