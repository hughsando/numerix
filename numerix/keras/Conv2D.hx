package numerix.keras;
import numerix.*;
using numerix.Hdf5Tools;

class Conv2D extends Layer
{
   var kernelSize:Array<Int>;
   var dilationRate:Array<Int>;
   var strides:Array<Int>;
   var filters:Int;
   var useBias:Bool;
   var activation:String;
   var padding:String;
   var weights:Tensor;
   var bias:Tensor;
   var result:Tensor;

   public function new(file:hdf5.Group, config:Dynamic, input:Layer)
   {
      super(config,input);
      kernelSize = config.kernel_size;
      dilationRate = config.dilation_rate;
      strides = config.strides;
      filters = config.filters;
      useBias = config.use_bias;
      activation = config.activation;
      padding = config.padding;
      config.kernel_initializer  = null;

      weights = file.read('model_weights/$name/$name/kernel:0');
      if (useBias)
         bias = file.read('model_weights/$name/$name/bias:0');
   }

   override public function getOutput() : Tensor
   {
      if (!valid)
      {
         valid = true;
         var src = inputs[0].getOutput();
         if (!validSize)
         {
            validSize = true;
            var inShape = src.shape;
            if (inShape.length!=3)
                throw "Conv2D should be HWC format";
            // TODO - shape / strides
            var h = inShape[0];
            var w = inShape[1];
            result = new Tensor( Nx.float32, [h, w, filters]);
         }

         // Run( src, result );
      }

      return result;
   }

   override public function toString() return 'Conv2D($name:$kernelSize x $filters $activation $weights $bias)';


}



