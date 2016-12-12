from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def readData(filename):
    f=file(filename).read().split('\n')
    x=[]
    for line in f:
        if line=="":
            continue
        attributes=line.split(",")
        x.append([])
        for i in range(4096):
            x[-1].append(attributes[i])
        x[-1].append(attributes[-2])
        x[-1].append(attributes[-2].split('_')[1].split(".")[0])

    f=open('newtest.csv','w')
    for i in x:
        f.write(','.join(i))
        f.write('\n')
readData('test.csv')
