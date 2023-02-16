#!/usr/bin/env python
# coding: utf-8

# In[6]:


import math

class Node(object):
    def __init__(self, label, geq, oneOrTwo):
        self.label = label
        self.geq = geq
        self.oneOrTwo = oneOrTwo
        self.left = -1
        self.right = -1
        if not hasattr(Node, 'static'):
            Node.static = 0
def entropy(numOnes, numZeros):
    if(numOnes == 0 or numZeros == 0):
        return 0
    total = numOnes + numZeros
    return -(numOnes/total*math.log2(numOnes/total)+numZeros/total*math.log2(numZeros/total))
def sortKey(data):
    return data[0]

# sortedLinearData is a 2d array with sortedLinearData[i][0] the feature, and sortedLinearData[i][1] the label.
# sortedLinearData is sorted by sortedLinearData[i][0] from smallest to largest.
def findBestIndexSplitLinear(sortedlinearData, numOnes, numZeros):
    maxGainRatio = 0
    entropyFull = entropy(numOnes, numZeros)
    currOnes = 0
    currZeros = 0
    bestSplit = 0
    bestOnes = 0
    bestZeros = 0
    i = 0
    while(i<len(sortedlinearData)-1):
        while(sortedlinearData[i][0] == sortedlinearData[i+1][0]):
            if(sortedlinearData[i][1] == 0):
                currZeros += 1
            else:
                currOnes += 1
            i += 1
            if(i == len(sortedlinearData) - 1):
                break;
        if(i == len(sortedlinearData) - 1):
            break;
        if(sortedlinearData[i][1] == 0):
            currZeros += 1
        else:
            currOnes += 1

        infoGain = (entropyFull-(((i+1)/len(sortedlinearData)*entropy(currOnes,currZeros)+(len(sortedlinearData)-(i+1))/len(sortedlinearData)*entropy(numOnes-currOnes,numZeros-currZeros))))
        gainRatio = infoGain/entropy(i+1,len(sortedlinearData)-(i+1))
#         print(sortedlinearData[i+1],gainRatio)

        if(gainRatio > maxGainRatio):
            maxGainRatio = gainRatio
            bestSplit = i+1
            bestOnes = currOnes
            bestZeros = currZeros
        i += 1
    if(maxGainRatio == 0):
        return -1, 0, 0, 0
    else:
        return sortedlinearData[bestSplit][0], bestOnes, bestZeros, maxGainRatio
    
def MakeSubtree(subtreeData, numOnes, numZeros):
    if(numOnes == 0):
        return Node(0,0,0)
    if(numZeros == 0):
        return Node(1,0,0)
    l1 = [(subtreeData[i][0],subtreeData[i][2]) for i in range(len(subtreeData))]
    l1.sort(key = sortKey)
    l2 = [(subtreeData[i][1],subtreeData[i][2]) for i in range(len(subtreeData))]
    l2.sort(key = sortKey)

    cutoff1, bestOnes1, bestZeros1, maxGainRatio1 = findBestIndexSplitLinear(l1, numOnes, numZeros)
    cutoff2, bestOnes2, bestZeros2, maxGainRatio2 = findBestIndexSplitLinear(l2, numOnes, numZeros)
    
    if(maxGainRatio1 == 0 and maxGainRatio2 == 0):
        return Node(1,0,0)
        
    cutoff = cutoff1
    bestOnes = bestOnes1
    bestZeros = bestZeros1
    oneOrTwo = 1

    if(maxGainRatio1 < maxGainRatio2):
        cutoff = cutoff2
        bestOnes = bestOnes2
        bestZeros = bestZeros2
        oneOrTwo = 2
    
    retVal = Node(-1, cutoff, oneOrTwo)
    retVal.right = MakeSubtree([data for data in subtreeData if data[oneOrTwo-1] >= cutoff], numOnes - bestOnes, numZeros - bestZeros)
    retVal.left = MakeSubtree([data for data in subtreeData if data[oneOrTwo-1] < cutoff], bestOnes, bestZeros)
    
    return retVal
def dictionaryMaker(root,d):
    d[root] = Node.static
    Node.static += 1
    if(root.label != -1):
        return;
    else:
        dictionaryMaker(root.right,d)
        dictionaryMaker(root.left,d)
def treeOutputter(root, d, d2):
    if(root.label != -1):
        d2[root] = "Output:", root.label
    else:
        d2[root] = "If coordinate", root.oneOrTwo, "is greater than or equal to", root.geq, "then go to", d[root.right],". Otherwise, go to",d[root.left]
        treeOutputter(root.right,d, d2)
        treeOutputter(root.left,d, d2)

def decisionTreePredictor(tree, x1, x2):
    if(tree.label != -1):
        return tree.label
    if(tree.oneOrTwo == 1):
        if(x1 >= tree.geq):
            return decisionTreePredictor(tree.right, x1, x2)
        return decisionTreePredictor(tree.left, x1, x2)
    if(x2 >= tree.geq):
        return decisionTreePredictor(tree.right, x1, x2)
    return decisionTreePredictor(tree.left, x1, x2)


# In[7]:


f = open("Dbig.txt", "r")
arr = []
numOnes = 0
numZeros = 0
for x in f:
    s = x.split()
    arr.append([float(s[0]),float(s[1]), int(s[2])])
    if(int(s[2]) == 0):
        numZeros += 1
    else:
        numOnes += 1
T = MakeSubtree(arr, numOnes, numZeros)
d = {}
d2 = {}
dictionaryMaker(T,d)
# print(d)
treeOutputter(T,d, d2)
print(d2.keys())
for key in d.keys():
    print(d[key],d2[key])
    print()


# In[8]:


import matplotlib.pyplot as plt
import numpy as np

x = np.array([0,1])
y = np.array([0,1])
plt.scatter(x, y)

#day two, the age and speed of 15 cars:
x = np.array([0,1])
y = np.array([1,0])
plt.scatter(x, y)

plt.show()


# In[9]:


import matplotlib.pyplot as plt
import numpy as np

arrOneX = [data[0] for data in arr if data[2] == 1]
arrOneY = [data[1] for data in arr if data[2] == 1]
arrZeroX = [data[0] for data in arr if data[2] == 0]
arrZeroY = [data[1] for data in arr if data[2] == 0]

x = np.array(arrOneX)
y = np.array(arrOneY)
plt.scatter(x, y)

#day two, the age and speed of 15 cars:
x = np.array(arrZeroX)
y = np.array(arrZeroY)
plt.scatter(x, y)

plt.show()


# In[28]:


f = open("Dbig.txt", "r")
arr = []
for x in f:
    s = x.split()
    arr.append([float(s[0]),float(s[1]), int(s[2])])

arrPerm = np.random.permutation(arr)
plotlist = []
for n in [32, 128, 512, 2048, 8192]:
    numOnes = 0
    numZeros = 0
    for i in range(n):
        if(int(arrPerm[i][2]) == 0):
            numZeros += 1
        else:
            numOnes += 1
    T = MakeSubtree([arrPerm[i] for i in range(n)], numOnes, numZeros)
    errors = 0
    for i in range(8192,len(arrPerm)):
        prediction = decisionTreePredictor(T, arr[i][0], arr[i][1])
        if(prediction != arr[i][2]):
            errors += 1
    plotlist.append((n, errors/(len(arrPerm)-8192)))

print(plotlist)
x = np.array([data[0] for data in plotlist])
y = np.array([data[1] for data in plotlist])
plt.scatter(x, y)
plt.show()

# d = {}
# d2 = {}
# dictionaryMaker(T,d)
# # print(d)
# treeOutputter(T,d, d2)
# print(d2.keys())
# for key in d.keys():
#     print(d[key],d2[key])
#     print()


# In[25]:


from sklearn import tree

f = open("Dbig.txt", "r")
arr = []
for x in f:
    s = x.split()
    arr.append([float(s[0]),float(s[1]), int(s[2])])

arrPerm = np.random.permutation(arr)
plotlist = []
for n in [32, 128, 512, 2048, 8192]:
    clf = DecisionTreeClassifier()
    clf = clf.fit([(arrPerm[i][0],arrPerm[i][1]) for i in range(n)], [arrPerm[i][2] for i in range(n)])
    errors = 0
    for i in range(8192,len(arrPerm)):
        prediction = clf.predict([[arr[i][0], arr[i][1]]])[0]
        if(prediction != arr[i][2]):
            errors += 1
    plotlist.append((n, errors/(len(arrPerm)-8192)))

print(plotlist)
x = np.array([data[0] for data in plotlist])
y = np.array([data[1] for data in plotlist])
plt.scatter(x, y)
plt.show()


# In[15]:


sys.path.append('/Users/johng23/Anaconda34/Lib/site-packages')


# In[16]:


import sys
print(sys.path)


# In[60]:


import numpy as np
import random as rndm
import math
from scipy.interpolate import lagrange


# In[54]:


num = 100
trainingX = [random.uniform(0,1) for i in range(num)]
trainingY = [math.sin(trainingX[i]) for i in range(num)]
testX = [random.uniform(0,1) for i in range(num)]
testY = [math.sin(trainingX[i]) for i in range(num)]


# In[55]:


f = lagrange(trainingX, trainingY)
errorTraining = 0
for i in range(num):
    errorTraining += abs(trainingY[i]-f(trainingX[i]))
print(errorTraining)


# In[57]:


f = lagrange(trainingX, trainingY)
errorTesting = 0
for i in range(100):
    errorTesting += abs(testY[i]-f(testX[i]))
print(errorTesting)


# In[80]:


trainingNoisyX = [trainingX[i] + np.random.normal(0,1) for i in range(num)]


# In[81]:


f = lagrange(trainingNoisyX, trainingY)
errorTraining = 0
for i in range(num):
    errorTraining += abs(trainingY[i]-f(trainingX[i]))
print(errorTraining)


# In[82]:


f = lagrange(trainingNoisyX, trainingY)
errorTesting = 0
for i in range(100):
    errorTesting += abs(testY[i]-f(testX[i]))
print(errorTesting)


# In[ ]:




