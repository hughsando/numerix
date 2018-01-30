import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from PIL import Image

model = "../model/data/squeezenet_v1.1";

net = caffe.Net(model+".prototxt", model+".caffemodel", caffe.TEST)

im = np.array(Image.open('../model/data/eagle_227.png'))

im = im - np.array([104,117,123]);
im = np.moveaxis(im,2,0)
print(im.shape)

net.blobs[ net.inputs[0] ].data[0] = im

net.forward()

result = net.blobs[ net.outputs[0] ].data[0];
for k in net.blobs.keys():
    print( k,"=", net.blobs[k].data[0].flatten()[0] )
print( result.argmax(), "=", result[ result.argmax() ]);
