import caffe
import numpy as np
from caffe import layers as L
from caffe import params as P
from PIL import Image

#model = "../model/data/squeezenet_v1.1"

if False:
    model = "../model/data/v3_train"
    im = np.array(Image.open('../model/data/Person_454.png'))[:,:,0:3]
else:
    model = "../model/data/v22_train"
    im = np.array(Image.open('../model/data/Person.png'))[:,:,0:3]

net = caffe.Net(model+".prototxt", model+".caffemodel", caffe.TEST)


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

if False:
    for k in net.blobs.keys():
        data = net.blobs[k].data[0]
        print( k, "->", data.min(), "...", data.max() );

#print( result.argmax(), "=", result[ result.argmax() ]);
print( result.min(), "...", result.max() );
s = result.shape;
if (s[0]==1):
    result = np.squeeze(result)
    result = np.where(result>0.5, 255, 0).astype(np.uint8)
    result = np.repeat(result,3).reshape(s[1],s[2],3)
    Image.fromarray(result).save('result.png');
