import mxnet as mx
import numpy as np

def readData(filename):
    f=file(filename).read().split('\r')
    x=[]
    y=[]
    for line in f:
        attributes=line.split("\t")
        x.append([])
        for i in range(0,6):
            x[-1].append(float(attributes[i]))
        y.append(int(attributes[-1]))
    return (np.array(x), np.array(y))


data_set,y=readData("icube_full.txt")
y=np.array(y)
y=y-1
"""data_set,y=readData("icube_train2.txt")"""
num_epoch=2000
data = mx.symbol.Variable('data')
fc1 = mx.symbol.FullyConnected(data, name='fc1', num_hidden=192)
act1 = mx.symbol.Activation(fc1, name='act1', act_type='relu')
fc2 = mx.symbol.FullyConnected(act1, name='fc2', num_hidden=192)
act2 = mx.symbol.Activation(fc2, name='act2', act_type='relu')
"""154 class number"""
fc3 = mx.symbol.FullyConnected(act2, name='fc3', num_hidden=154)
softmax = mx.symbol.SoftmaxOutput(fc3, name='softmax')
model = mx.model.FeedForward.create(
     softmax,
     X=data_set,
     num_epoch=num_epoch,
     learning_rate=0.01, y=y, numpy_batch_size=500)
output=np.array(model.predict(data_set))
print(output[0][93])
print(output[1])
print(output[1][93])
output=np.argmax(output, axis=1)
print(output)
print(sum(output==y))
print(len(data_set))
