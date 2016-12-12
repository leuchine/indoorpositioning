from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import random

def readData(filename):
    global mag
    f=file(filename).read().split('\n')
    x=[]
    for line in f:
        if line=="":
            continue
        attributes=line.split(",")
        x.append([])
        for i in range(4096):
            x[-1].append(attributes[i])
        label=int(attributes[-2].split('_')[1].split(".")[0])
        x[-1].append(str(mag[label]))
        x[-1].append(attributes[-2])
        x[-1].append(attributes[-2].split('_')[0])

    f=open('pgpwithmag.csv','w')
    for i in x:
        f.write(','.join(i))
        f.write('\n')

def readData2(filename):
    global mag
    f=file(filename).read().split('\n')
    x=[]
    for line in f:
        if line=="":
            continue
        attributes=line.split(",")
        x.append([])
        for i in range(4096):
            x[-1].append(attributes[i])
        label=int(attributes[-1])
        x[-1].append(str(mag[label]))
        x[-1].append(attributes[-2])
        x[-1].append(attributes[-1])

    f=open('train2.csv','w')
    for i in x:
        f.write(','.join(i))
        f.write('\n')


mag={}
count={}
f=file('pgpmag.csv').read().split('\r')
for line in f:
    elements=line.split(";")
    if len(elements)==2:
        print(line)
        label=int(float(elements[1]))
        value=float(elements[0])
        mag[label]=mag.get(label, 0)+value
        count[label]=count.get(label, 0)+1

for key in mag:
    mag[key]=mag[key]/count[key]
print(mag)
readData('pgp.csv')

