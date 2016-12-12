import os
import numpy as np
import scipy.io as sio
from os.path import isfile,join
import sys,getopt

# Main path to your caffe installation
caffe_root = '/Users/leuchine/Documents/caffe'

# Model prototxt file
model_prototxt = caffe_root+'/_temp/model/hybridCNN_deploy_FC7.prototxt'

# Model caffemodel file
model_trained = caffe_root + '/_temp/model/hybridCNN_iter_700000.caffemodel'

# Path to the mean image (used for input processing)
mean_path = caffe_root + '/_temp/model/hybridCNN_mean.binaryproto'

# Name of the layer we want to extract
layer_name = 'fc7'

import caffe

caffe.set_mode_cpu()
#caffe.set_mode_gpu()
net = caffe.Net(model_prototxt,model_trained,caffe.TEST)

#data preprocessing
transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
blob = caffe.proto.caffe_pb2.BlobProto()
mean_data = open(mean_path,'rb').read()
blob.ParseFromString(mean_data)
arr = np.array(caffe.io.blobproto_to_array(blob))
m_min,m_max = arr[0].min(),arr[0].max()
normal_mean = (arr[0]-m_min)/(m_max-m_min)
dim=np.array((227,227,3))
#print normal_mean.transpose((1,2,0)).shape
mean = caffe.io.resize_image(normal_mean.transpose((1,2,0)),dim).transpose((2,0,1))*(m_max-m_min)+m_min

transformer.set_mean('data',mean)
transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data',(2,1,0))
transformer.set_raw_scale('data',255.0)

framedir=caffe_root + '/_temp/images'
destdir=caffe_root + '/_temp/features'

videos = next(os.walk(framedir))[1]
print(videos)
#vidnum = np.size(videos)
#print vidnum
for v in videos:
  output=file(v+".csv",'w')
  print(v)
  folder=join(framedir,v)
  frames = [ f for f in os.listdir(folder) if isfile(join(folder,f))]
  if '.DS_Store' in frames:
    frames.remove('.DS_Store')
  fnum = np.size(frames)
  #fnum=1
  #print fnum
  net.blobs['data'].reshape(fnum,3,227,227)
  cnt=0
  for f in frames:
    img_path = join(folder,f)
    print img_path
    img=caffe.io.load_image(img_path)
    net.blobs['data'].data[cnt] = transformer.preprocess('data',img)
    cnt=cnt+1
    #break;
 
  net.forward()
  fc7=net.blobs[layer_name].data.copy()
  
  cnt=0;

  for f in frames:
    fc=fc7[cnt,:]
    print(len(fc))
 
    for i in fc:
      output.write(str(round(i,2)))
      output.write(',')
    output.write(f+",")
    output.write(f.split('_')[0]+"\n")
    cnt=cnt+1



