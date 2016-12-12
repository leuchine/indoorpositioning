from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def readData(filename):
    global hasmag;
    f=file(filename).read().split('\n')
    x=[]
    y=[]
    for line in f:
        if line=="":
            continue
        attributes=line.split(",")
        x.append([])
        if hasmag:
            for i in range(4097):
                x[-1].append(float(attributes[i]))
        else:
            for i in range(4096):
                x[-1].append(float(attributes[i]))

        y.append(int(attributes[-1]))
    return (np.array(x), np.array(y))

def readData2(filename):
    f=file(filename).read().split('\n')
    x=[]
    y=[]
    testx=[]
    testy=[]
    for line in f:
        if line=="":
            continue
        attributes=line.split(",")
        if(random.random()>0.2):
            x.append([])
            if hasmag:
                for i in range(4097):
                    x[-1].append(float(attributes[i]))
            else:
                for i in range(4096):
                    x[-1].append(float(attributes[i]))
            y.append(int(attributes[-1]))
        else:
            testx.append([])
            if hasmag:
                for i in range(4097):
                    testx[-1].append(float(attributes[i]))
            else:
                for i in range(4096):
                    testx[-1].append(float(attributes[i]))
            testy.append(int(attributes[-1]))
    return (np.array(x), np.array(y),  np.array(testx),  np.array(testy))
hasmag=True
#x,y=readData("train2.csv")
#testx,testy=readData("test2.csv")
x, y, testx,testy=readData2("pgpwithmag.csv")
"""output=OneVsOneClassifier(LinearSVC(random_state=0)).fit(x, y).predict(x)"""
print(x.shape)
print(y.shape)
print(testx.shape)
print(testy.shape)
output=LogisticRegression(solver='lbfgs',multi_class='multinomial').fit(x, y).predict(testx)
#output=SVC(kernel='rbf').fit(x, y).predict(testx)
output=np.array(output)
print('output:')
for i in output:
    print(i)
print('correct:')
for i in testy:
    print(i)

print(sum(output==testy))
print(len(testy))
