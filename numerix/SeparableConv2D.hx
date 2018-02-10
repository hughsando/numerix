package numerix;
using numerix.Hdf5Tools;

class SeparableConv2D extends Conv2D
{
   var pweights:Tensor;


   override public function setWeights(inWeights:Array<Tensor>)
   {
      weights = inWeights[0];
      pweights = inWeights[1];
      bias = inWeights[2];
      release();

      handle = Conv2D.layCreateConv2D(strides, activation, padding, weights,pweights,bias,false,isDeconvolution);
   }

   override public function toString() return 'SeparableConv2D($name:$kernelSize x $filters $activation $weights %pweights $bias)';
}

