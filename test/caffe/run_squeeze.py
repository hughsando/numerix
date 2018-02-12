import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from PIL import Image

#model = "../model/data/squeezenet_v1.1"
model = "../model/data/v22_train"

net = caffe.Net(model+".prototxt", model+".caffemodel", caffe.TEST)

im = np.array(Image.open('../model/data/Person.png'))[:,:,0:3]

#im = im - np.array([104,117,123]);
im = np.moveaxis(im,2,0)
print(im.shape)

net.blobs[ net.inputs[0] ].data[0] = im

net.forward()

result = net.blobs[ net.outputs[0] ].data[0];

if False:
    for k in net.blobs.keys():
        data = net.blobs[k].data[0]
        shape = data.shape;
        if (len(shape)==4):
            print( k,shape," =>")
        elif (len(shape)==3):
            print( k,shape," =>", data[ shape[0]-1, shape[1]-1, shape[2]-1] )
        elif (len(shape)==2):
            print( k,shape," =>", data[ shape[0]-1, shape[1]-1])
        else:
            print( k,shape," =>", data[ shape[0]-1])

#print( result.argmax(), "=", result[ result.argmax() ]);
print( result.min(), "...", result.max() );
