#!/usr/bin/env python
# coding: utf-8

# In[1]:


eFiles = [open("e"+str(i)+".txt").read() for i in range(20)]
jFiles = [open("j"+str(i)+".txt").read() for i in range(20)]
sFiles = [open("s"+str(i)+".txt").read() for i in range(20)]


# In[21]:


def charToNum(char):
    if(ord(char)==32):
        return 26;
    else:
        return ord(char)-97


# In[44]:


counts = [0 for i in range(27)]

for i in range(10):
    for char in eFiles[i]:
        if(ord(char) == 10):
            continue;
        else:
            counts[charToNum(char)] += 1
tot = sum(counts)
thetae = [(counts[i]+alpha)/(tot+alpha*K) for i in range(27)]

counts = [0 for i in range(27)]

for i in range(10):
    for char in jFiles[i]:
        if(ord(char) == 10):
            continue;
        else:
            counts[charToNum(char)] += 1
tot = sum(counts)
thetaj = [(counts[i]+alpha)/(tot+alpha*K) for i in range(27)]

counts = [0 for i in range(27)]

for i in range(10):
    for char in sFiles[i]:
        if(ord(char) == 10):
            continue;
        else:
            counts[charToNum(char)] += 1
tot = sum(counts)
thetas = [(counts[i]+alpha)/(tot+alpha*K) for i in range(27)]


# In[45]:


counts = [0 for i in range(27)]
for char in eFiles[10]:
    if(ord(char) == 10):
        continue;
    else:
        counts[charToNum(char)] += 1


# In[58]:


import math
pxye = 0
for i in range(27):
    pxye += math.log(thetae[i])*counts[i]
print(pxye)

pxyj = 0
for i in range(27):
    pxyj += math.log(thetaj[i])*counts[i]
print(pxyj)

pxys = 0
for i in range(27):
    pxys += math.log(thetas[i])*counts[i]
print(pxys)


# In[63]:


print(pxye + math.log(1/3))
print(pxyj + math.log(1/3))
print(pxys + math.log(1/3))


# In[64]:


def naiveBayesClassifier(thetae,thetaj,thetas,counts):
    pxye = 0
    for i in range(27):
        pxye += math.log(thetae[i])*counts[i]
    pxyj = 0
    for i in range(27):
        pxyj += math.log(thetaj[i])*counts[i]
    pxys = 0
    for i in range(27):
        pxys += math.log(thetas[i])*counts[i]
    
    m = max(pxye,pxyj,pxys)
    if(m == pxye):
        return 0
    if(m == pxyj):
        return 1
    if(m == pxys):
        return 2
    


# In[66]:


mat = [[0 for i in range(3)] for j in range(3)]
for i in range(10,20):
    counts = [0 for j in range(27)]
    for char in eFiles[i]:
        if(ord(char) == 10):
            continue;
        else:
            counts[charToNum(char)] += 1
    prediction = naiveBayesClassifier(thetae,thetaj,thetas,counts)
    mat[0][prediction] += 1
    
    counts = [0 for j in range(27)]
    for char in jFiles[i]:
        if(ord(char) == 10):
            continue;
        else:
            counts[charToNum(char)] += 1
    prediction = naiveBayesClassifier(thetae,thetaj,thetas,counts)
    mat[1][prediction] += 1
    
    counts = [0 for j in range(27)]
    for char in sFiles[i]:
        if(ord(char) == 10):
            continue;
        else:
            counts[charToNum(char)] += 1
    prediction = naiveBayesClassifier(thetae,thetaj,thetas,counts)
    mat[2][prediction] += 1


# In[67]:


print(mat)


# In[71]:


from sklearn.datasets import fetch_openml
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)


# In[878]:


import numpy as np

alpha = .0001
batch_size = 32
d1 = 300
d2 = 200
d3 = 100
d = 784
k = 10

def sigma(z):
    if(z>500):
        print("hi")
        return 1
    if(z<-500):
        print("hi0")
        return 0
    
    return 1/(1+math.exp(-z))

def sigma_vec(z):
    return [[sigma(z[i][0])] for i in range(len(z))]
def g(zbar):
    n = len(zbar)
    vOut = [[0] for i in range(n)]
    for i in range(n):
        try:
            vOut[i][0] = 1/sum([math.exp(zbar[j][0] - zbar[i][0]) for j in range(n)])
        except:
#             print(zbar)
            print("hi1")
    return np.array(vOut)
def onehot(y):
    vOut = [[0] for i in range(10)]
    vOut[y] = [1]
    return np.array(vOut)
def D(zbar):
    n = len(zbar)
    mOut = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        mOut[i][i] = sigma(zbar[i][0])*(1-sigma(zbar[i][0]))
    return np.array(mOut)


# In[791]:


def cost(y, yhat):
    return -sum([y[i]*math.log(yhat[i]) for i in range(len(y))])
def predict(W1, W2, W3, z0):
    z1 = np.matmul(W1, z0)
    z2 = np.matmul(W2, sigma_vec(z1))
    z3 = np.matmul(W3, sigma_vec(z2))
    prediction = g(z3)
#     print("z0",z0)
#     print("z1",z1)
#     print("z2",z2)
#     z3 += np.array([[1111],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    print("z3",z3)
    maxindex = 0
    maxprob = 0
    for i in range(10):
        if(prediction[i] > maxprob):
            maxprob = prediction[i]
            maxindex = i
    return maxindex
def learn(X, y, batches, alpha, batch_size, d1, d2, d, k):
    W1 = np.array([[(np.random.rand()-.5)/1000 for i in range(d)] for j in range(d1)], dtype = "float")
    W2 = np.array([[(np.random.rand()-.5)/100 for i in range(d1)] for j in range(d2)], dtype = "float")
    W3 = np.array([[(np.random.rand()-.5)/100 for i in range(d2)] for j in range(k)], dtype = "float")
    costArr = []
    for j in range(batches):
        W1adjust = np.array([[0 for i in range(d)] for j in range(d1)], dtype = "float")
        W2adjust = np.array([[0 for i in range(d1)] for j in range(d2)], dtype = "float")
        W3adjust = np.array([[0 for i in range(d2)] for j in range(k)], dtype = "float")
        totCost = 0
#         print(W3)

        for k1 in range(batch_size):
            l = j*batch_size + k1
            l = k1
            z0 = [[X[l][i]] for i in range(len(X[l]))]
            z1 = np.matmul(W1, z0)
            print(z1)
            z1sig = sigma_vec(z1)
            z2 = np.matmul(W2, z1sig)
            z2sig = sigma_vec(z2)
            z3 = np.matmul(W3, z2sig)
#             print(W3)
            prediction = g(z3)
            totCost += cost(onehot(int(y[l])), prediction)
#             print(z3)
#             print(g(z3))
            partial3 = np.transpose(np.subtract(prediction,onehot(int(y[l]))))
            partial2 = np.matmul(np.matmul(partial3, W3), D(z2))
            partial1 = np.matmul(np.matmul(partial2, W2), D(z1))
#             print(partial1)
#             if(np.any(partial2)):
#                 print(partial2-partial2[0][0]*np.array([[1.0] for i in range(len(partial2))]))
#             print(len(W3adjust), len(W3adjust[0]))
#             print(np.any(z0), np.any(partial1), np.any(np.matmul(z0,partial1)))
            W3adjust = np.add(W3adjust, np.transpose(np.matmul(z2sig,partial3)))
            W2adjust = np.add(W2adjust, np.transpose(np.matmul(z1sig,partial2)))
            W1adjust = np.add(W1adjust, np.transpose(np.matmul(z0,partial1)))
#             print(np.any(z0), np.any(partial1), np.any(np.matmul(z0,partial1)))
            
        W1adjust *= alpha/batch_size
        W2adjust *= alpha/batch_size
        W3adjust *= alpha/batch_size
        
#         print("W1adjust", W1adjust[0])
#         print("W2", W2)
#         print("W3", W3)
#         print("p1", partial1)
#         print("p2", partial2)
#         print("p3", partial3)
#         print("test", partial3+100*np.transpose(onehot(int(y[0]))))
#         print("p3W3", np.matmul(partial3+100*np.transpose(onehot(int(y[0]))), W3))
#         print("--------------------")
        W1 -= W1adjust
        W2 -= W2adjust
        W3 -= W3adjust
        costArr.append(totCost)
    return W1,W2,W3,costArr


# In[792]:


W1, W2, W3, costArr = learn(X, y, batches = 100, alpha = 0.05, batch_size = 2, d1 = 30, d2 = 10, d = 784, k = 10)


# In[784]:


import matplotlib.pyplot as pl
pl.scatter([i for i in range(len(costArr))],costArr)


# In[786]:



correctCount = 0
for l in range(500,511):
    prediction = predict(W1, W2, W3, [[specialSet[l][i]] for i in range(len(specialSet[l]))])
    print(prediction,specialSety[l])
    if(prediction == int(specialSety[l])):
        correctCount += 1
        


# In[764]:


print(W1)
print(W2)
print(W3)


# In[738]:


specialSet = [X[i] for i in range(0,30000) if int(y[i]) < 2]
specialSety = [y[i] for i in range(0,30000) if int(y[i]) < 2]
print(len(specialSet))


# In[302]:


print(np.any(W1))


# In[785]:


print(y[0],y[1])


# In[800]:


X0 = np.array([[X[0][i]] for i in range(len(X[0]))])
X1 = np.array([[X[1][i]] for i in range(len(X[1]))])

np.matmul(W2,sigma_vec(np.matmul(W1,(X0-X1))))


# In[879]:


def cost(y, yhat):
    return -sum([y[i]*math.log(yhat[i]) for i in range(len(y))])
def predict(W3, z0):
    z1 = np.matmul(W3, z0)
    prediction = g(z3)
#     print("z0",z0)
#     print("z1",z1)
#     print("z2",z2)
#     z3 += np.array([[1111],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    print("z3",z3)
    maxindex = 0
    maxprob = 0
    for i in range(10):
        if(prediction[i] > maxprob):
            maxprob = prediction[i]
            maxindex = i
    return maxindex
def learn(X, y, batches, alpha, batch_size, d1, d2, d, k):
    W2 = np.array([[(np.random.rand()-.5)/100 for i in range(d)] for j in range(d2)], dtype = "float")
    W3 = np.array([[(np.random.rand()-.5)/100 for i in range(d2)] for j in range(k)], dtype = "float")
    costArr = []
    for j in range(batches):
        W2adjust = np.array([[0 for i in range(d)] for j in range(d2)], dtype = "float")
        W3adjust = np.array([[0 for i in range(d2)] for j in range(k)], dtype = "float")
        totCost = 0

        for k1 in range(batch_size):
            l = j*batch_size + k1
            l = k1
            
            z0 = [[X[l][i]] for i in range(len(X[l]))]
            z2 = np.matmul(W2, z0)
            z2sig = sigma_vec(z2)
            z3 = np.matmul(W3, z2sig)
            prediction = g(z3)
            
            totCost += cost(onehot(int(y[l])), prediction)
            
            partial3 = np.transpose(np.subtract(prediction,onehot(int(y[l]))))
            partial2 = np.matmul(np.matmul(partial3, W3), D(z2))
            
            W3adjust = np.add(W3adjust, np.transpose(np.matmul(z2sig,partial3)))
            W2adjust = np.add(W2adjust, np.transpose(np.matmul(z0,partial2)))
            
        W2adjust *= alpha/batch_size
        W3adjust *= alpha/batch_size
        
        W2 -= W2adjust
        W3 -= W3adjust
        costArr.append(totCost)
    return W2, W3,costArr


# In[880]:


W2, W3, costArr = learn(X, y, batches = 100, alpha = 0.01, batch_size = 2, d1 = 30, d2 = 10, d = 784, k = 10)


# In[881]:


import matplotlib.pyplot as pl
pl.scatter([i for i in range(len(costArr))],costArr)


# In[884]:


np.matmul(np.array([[2,4],[1,3],[1,1]]),np.array([[1],[2]]))


# In[ ]:




