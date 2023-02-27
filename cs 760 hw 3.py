#!/usr/bin/env python
# coding: utf-8

# In[123]:


f = open("hw3-1/data/D2z.txt")
xvals = []
yvals = []
xyvals = []
labels = []
for x in f:
    s = x.split(" ")
    xvals.append(float(s[0]))
    yvals.append(float(s[1]))
    xyvals.append([float(s[0]),float(s[1])])
    labels.append(int(s[2]))


# In[124]:


import numpy as np
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(xyvals)


# In[125]:


plotlist = []
for i in range(-20,21):
    for j in range(-20,21):
        plotlist.append([i/10,j/10])


# In[126]:


plotneighbors = neigh.kneighbors(plotlist,1)
plotTestdata1 = [plotlist[i] for i in range(len(plotlist)) if labels[plotneighbors[1][i][0]]==1]
plotTestdata0 = [plotlist[i] for i in range(len(plotlist)) if labels[plotneighbors[1][i][0]]==0]
plotTraindata1 = [xyvals[i] for i in range(len(xyvals)) if labels[i] == 1]
plotTraindata0 = [xyvals[i] for i in range(len(xyvals)) if labels[i] == 0]

import matplotlib.pyplot as plt
plt.scatter([plotTestdata1[i][0] for i in range(len(plotTestdata1))],[plotTestdata1[i][1] for i in range(len(plotTestdata1))], c = "blue")
plt.scatter([plotTestdata0[i][0] for i in range(len(plotTestdata0))],[plotTestdata0[i][1] for i in range(len(plotTestdata0))], c = "red")

plt.scatter([plotTraindata1[i][0] for i in range(len(plotTraindata1))],[plotTraindata1[i][1] for i in range(len(plotTraindata1))], marker = "s")
plt.scatter([plotTraindata0[i][0] for i in range(len(plotTraindata0))],[plotTraindata0[i][1] for i in range(len(plotTraindata0))], marker = "^")

plt.show()


# In[127]:


f = open("hw3-1/data/emails.csv")
f.readline()
emailInfo = [ [0]*3000 for i in range(5000)]
labelInfo = []
i = 0
for x in f:
    s = x.split(",")
    for j in range(1,3001):
        emailInfo[i][j-1] = int(s[j])
    labelInfo.append(int(s[3001]))
    i += 1
    


# In[128]:


import numpy as np
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(emailInfo)


# In[162]:


def trainingSetTestSet(fold, emailInfo):
    indexTraining = 0
    indexTest = 0
    dictTraining = {}
    dictTest = {}
    trainingSet = []
    testSet = []
    for i in range(0,5000):
        if(fold*1000 <= i < (fold+1)*1000):
            dictTest[indexTest] = i
            testSet.append(emailInfo[i])
            indexTest += 1
        else:
            dictTraining[indexTraining] = i
            trainingSet.append(emailInfo[i])
            indexTraining += 1
    return trainingSet, testSet, dictTraining, dictTest

def accuracy(predictions, actual):
    counter = 0
    for i in range(1000):
        if(predictions[i] == actual[i]):
            counter += 1
    return counter/1000

def recall(predictions, actual):
    counter = 0
    counterT = 0
    for i in range(1000):
        if(actual[i] == 1):
            counterT += 1
            if(predictions[i] == actual[i]):
                counter += 1
#     print(counter)
#     print(counterT)
    return counter/counterT

def precision(predictions, actual):
    counter = 0
    counterT = 0
    for i in range(1000):
        if(predictions[i] == 1):
            counterT += 1
            if(predictions[i] == actual[i]):
                counter += 1
    return counter/counterT


# In[142]:


for fold in range(0,5):
    trainingSet, testSet, dictTraining, dictTest = trainingSetTestSet(fold,emailInfo)
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(trainingSet)
    
    NN = neigh.kneighbors(testSet,1)
    predictions = [labelInfo[dictTraining[NN[1][i][0]]] for i in range(len(testSet))]
    actual = [labelInfo[dictTest[i]] for i in range(len(testSet))]
    print(fold, accuracy(predictions, actual), recall(predictions, actual), precision(predictions, actual))


# In[173]:


import numpy as np
def sigma(z):
#     if(z<-100):
#         return 0
#     if(z>100):
#         return 1
    try:
        retval = 1/(1+math.exp(-z))    
        return retval
    except:
        print(z)
def logisticRegression(lr, theta0, x, y):
#     for l in x:
#         l.append(1)
#     theta0.append(0)
    
#     x = [x[i] for i in range(100)]
#     y = [y[i] for i in range(100)]
#     theta0 = [theta0[i] for i in range(100)]
    for j in range(10):
        for l in range(len(x)):
            factor = -lr
            if(y[l] == 1):
                factor *= math.log(sigma(np.dot(theta0,x[l])))
            else:
                factor *= -math.log(1-sigma(np.dot(theta0,x[l])))
            for k in range(3000):
                theta0[k] += factor * x[l][k]
#         print(j)
# #         print(theta0)
#         factor = (-sum([math.log(sigma(np.dot(theta0,x[i]))) for i in range(4000) if y[i] == 1]))
#         factor += (-sum([math.log(1-sigma(np.dot(theta0,x[i]))) for i in range(4000) if y[i] == 0]))
#         factor *= 1/4000 * (-lr)
#         print(factor)
#         for l in range(4000):
#             for k in range(3000):
#                 theta0[k] += factor * x[l][k]
    return theta0
def linearClassifier(theta,testSet,t=0.5):
    predictions = []
    for i in range(len(testSet)):
        if(sigma(np.dot(theta,testSet[i]))>t):
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions


# In[164]:


import math
for fold in range(0,5):
    trainingSet, testSet, dictTraining, dictTest = trainingSetTestSet(fold,emailInfo)
    theta = logisticRegression(0.000001, [0 for i in range(3000)], trainingSet, [labelInfo[i] for i in [dictTraining[i] for i in range(4000)]])
    
    predictions = linearClassifier(theta, testSet)
    actual = [labelInfo[dictTest[i]] for i in range(len(testSet))]
    print(fold, accuracy(predictions, actual), recall(predictions, actual), precision(predictions, actual))


# In[ ]:


Y = [0,2/6,4/6,6/6,6/6]
X = [0,0,1/4,2/4,1]
plt.scatter(X,Y)
plt.plot(X,Y)
plt.show()


# In[168]:


for k in [1,3,5,7,10]:
    totAccuracy = 0
    for fold in range(0,5):
        trainingSet, testSet, dictTraining, dictTest = trainingSetTestSet(fold,emailInfo)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(trainingSet)
        predictions = []
        NN = neigh.kneighbors(testSet,k)
        for i in range(len(testSet)):
            tot = 0
            for j in range(k):
                tot += labelInfo[dictTraining[NN[1][i][j]]]
            if(tot/k>1/2):
                predictions.append(1)
            else:
                predictions.append(0)
#         predictions = [labelInfo[dictTraining[NN[1][i][0]]] for i in range(len(testSet))]
        actual = [labelInfo[dictTest[i]] for i in range(len(testSet))]
        totAccuracy += accuracy(predictions, actual)
    totAccuracy /= 5
    print(k, totAccuracy)


# In[169]:


Y = [.8332,.8422,.8408,.8462, .8556]
X = [1,3,5,7,10]
plt.scatter(X,Y)
plt.plot(X,Y)
plt.show()


# In[175]:


def FP(predictions, actual):
    counter = 0
    counterF = 0
    for i in range(1000):
        if(actual[i] == 0):
            counterF += 1
            if(predictions[i] != actual[i]):
                counter += 1
    return counter/counterF

X = []
Y = []
k = 5
for threshold in range(0,6):
    for fold in range(1):
        trainingSet, testSet, dictTraining, dictTest = trainingSetTestSet(fold,emailInfo)
        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(trainingSet)
        predictions = []
        NN = neigh.kneighbors(testSet,k)
        for i in range(len(testSet)):
            tot = 0
            for j in range(k):
                tot += labelInfo[dictTraining[NN[1][i][j]]]
            if(tot>threshold):
                predictions.append(1)
            else:
                predictions.append(0)
        actual = [labelInfo[dictTest[i]] for i in range(len(testSet))]
        X.append(FP(predictions, actual))
        Y.append(recall(predictions, actual))
plt.scatter(X,Y)
plt.plot(X,Y)

X = []
Y = []
for fold in range(1):
    
    trainingSet, testSet, dictTraining, dictTest = trainingSetTestSet(fold,emailInfo)
    theta = logisticRegression(0.000001, [0 for i in range(3000)], trainingSet, [labelInfo[i] for i in [dictTraining[i] for i in range(4000)]])
    for threshold in range(0,101):
        predictions = linearClassifier(theta, testSet, t=threshold/100)
        actual = [labelInfo[dictTest[i]] for i in range(len(testSet))]
        X.append(FP(predictions, actual))
        Y.append(recall(predictions, actual))
plt.scatter(X,Y, color = "red")
plt.plot(X,Y, color = "red")

plt.show()


# In[ ]:




