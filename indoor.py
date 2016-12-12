from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def readData(filename):
    f=file(filename).read().split('\r')
    x=[]
    y=[]
    for line in f:
        attributes=line.split("\t")
        x.append([])
        for i in range(6):
            x[-1].append(float(attributes[i]))
        y.append(int(attributes[-1]))
    return (np.array(x), np.array(y))

def readData2(filename):
    f=file(filename).read().split('\r')
    x=[]
    y=[]
    testx=[]
    testy=[]
    for line in f:
        attributes=line.split("\t")
        if(random.random()>0.4):
            x.append([])
            for i in range(6):
                x[-1].append(float(attributes[i]))
            y.append(int(attributes[-1]))
        else:
            testx.append([])
            for i in range(6):
                testx[-1].append(float(attributes[i]))
            testy.append(int(attributes[-1]))
    return (np.array(x), np.array(y),  np.array(testx),  np.array(testy))

"""x,y=readData("icube_full.txt")"""
"""testx,testy=readData("icube_test.txt")"""
x,y=readData("icube_train2.txt")
testx,testy=readData("icube_test2.txt")
"""x, y, testx,testy=readData2("icube_full.txt")"""
"""output=OneVsOneClassifier(LinearSVC(random_state=0)).fit(x, y).predict(x)"""
"""output=LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(x, y).predict(x)"""
output=SVC(kernel='rbf').fit(x, y).predict(x)
output=np.array(output)
print(sum(output==y))
print(len(x))
