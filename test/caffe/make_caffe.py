import caffe
import numpy as np
import sys
from caffe import layers as L
from caffe import params as P

O = 8
I = 8
H = 2
W = 3

net = caffe.NetSpec()
net.data = L.Input(shape=[dict(dim=[O, I, H, W])])
net.deconv = L.Deconvolution(net.data, convolution_param=dict(kernel_size=4, stride=2,num_output=O, pad=1), name="mydeconv" )

#net.relu = L.ReLU(net.conv)
#net.maxp = L.Pooling(net.relu,pool=P.Pooling.MAX, kernel_size=2, stride=2, pad=0 )
#net.concat = L.Concat(net.maxp,net.maxp)
#net.conv2 = L.Convolution(net.concat, kernel_size=3, stride=1,num_output=4, pad=1, name="myconv2" )
#net.ave = L.Pooling(net.conv2,pool=P.Pooling.AVE, global_pooling=True )
#net.softmax = L.Softmax(net.ave)

proto = net.to_proto()


with open('mynet.prototxt', 'w') as f:
   f.write(str(proto))

net = caffe.Net("mynet.prototxt", caffe.TEST)

print( net.params['mydeconv'][0].data.shape );

np.random.seed( 4 )

def randomize(name):
   param = net.params[name][0]
   param.data[:,:,:,:] = np.random.random_sample( list(param.data.shape) ) - 0.5
   #param.data[:,1:8,:,:] = 0.0
   #for y in range(2):
   #   for x in range(2):
   #      param.data[0,0,y,x] = (y+1)*10+(x+1);
   param = net.params[name][1]
   param.data[:] = np.random.rand( param.data.size ) - 0.5
   #param.data[:] = 0

randomize('mydeconv')
#randomize('myconv2')

net.save("mynet.caffemodel");

inp = np.zeros( shape=(1,I,H,W) )
for chan in range(I):
   for y in range(H):
      for x in range(W):
         inp[0,chan,y,x] = ( ((chan*9+12)%13) + ((y*39 + 6)%17) + ((x*37 + 8)%19)  ) * 0.1
#inp[0,0,1,1] = 1

net.blobs[ net.inputs[0] ].data[0] = inp

net.forward()

print( net.outputs[0], " ->  ", net.blobs[ net.outputs[0] ].data[0].shape );
np.set_printoptions(suppress=True) 
print( net.blobs[ net.outputs[0] ].data[0] );

#test = caffe.Net("mynet.prototxt", "mynet.caffemodel", caffe.TEST)
#test.forward()
