import numpy as np
import os, sys, getopt

# Main path to your caffe installation
caffe_root = '/home/yifang/caffe-master/'

# Model prototxt file
model_prototxt = caffe_root+'examples/_temp/model/hybridCNN_deploy_FC7.prototxt'

# Model caffemodel file
model_trained = caffe_root + 'examples/_temp/model/hybridCNN_iter_700000.caffemodel'

# File containing the class labels
#imagenet_labels = caffe_root + 'data/ilsvrc12/synset_words.txt'

# Path to the mean image (used for input processing)
mean_path = caffe_root + '/examples/_temp/model/hybridCNN_mean.binaryproto'

# Name of the layer we want to extract
layer_name = 'fc7'

import caffe

def main(argv):
    inputfile = ''
    outputfile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print 'caffe_feature_extractor.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg

    print 'Reading images from "', inputfile;
    print 'Writing vectors to "', outputfile;

    # Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_cpu()
    # Loading the Caffe model, setting preprocessing parameters
    
    net = caffe.Net(model_prototxt,model_trained,caffe.TEST);
    
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
    
    net.blobs['data'].reshape(1,3,227,227) #because we only process three images
    img_path='/home/yifang/caffe-master/examples/images/cat.jpg'
    im = caffe.io.load_image(img_path)
    net.blobs['data'].data[...] = transformer.preprocess('data',im)

    net.forward()
    dlfea=net.blobs[layer_name].data.copy()

    print dlfea.shape
    
    # Loading class labels
    #with open(imagenet_labels) as f:
    #    labels = f.readlines()

    # This prints information about the network layers (names and sizes)
    # You can uncomment this, to have a look inside the network and choose which layer to print
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()

    # Processing one image at a time, printint predictions and writing the vector to a file
    #with open(inputfile, 'r') as reader:
    #    with open(outputfile, 'w') as writer:
    #        writer.truncate()
    #        for image_path in reader:
    #            image_path = image_path.strip()
    #            input_image = caffe.io.load_image(image_path)
    #            prediction = net.predict([input_image], oversample=False)
    #            print os.path.basename(image_path), ' : ' , labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')'
    #            np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')

if __name__ == "__main__":
    main(sys.argv[1:])
