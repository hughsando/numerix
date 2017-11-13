package numerix;
using numerix.Hdf5Tools;

class Conv2D extends Layer
{
   public var kernelSize(default,null):Array<Int>;
   public var dilation(default,null):Array<Int>;
   public var strides(default,null):Array<Int>;
   public var filters(default,null):Int;
   public var useBias(default,null):Bool;
   public var activation(default,null):String;
   public var padding(default,null):String;
   public var weights(default,null):Tensor;
   public var bias(default,null):Tensor;
   public var result(default,null):Tensor;

   public function new(file:hdf5.Group, config:Dynamic, input:Layer)
   {
      super(config,input);

      kernelSize = config.kernelSize;
      dilation = config.dilation;
      strides = config.strides;
      filters = config.filters;
      useBias = config.useBias;
      activation = config.activation;
      padding = config.padding;
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
            if (result!=null)
               result.release();
            result = Tensor.create( Nx.float32, [h, w, filters]);
         }

         // Run( src, result );
      }

      return result;
   }


   override public function setWeights(inWeights:Array<Tensor>)
   {
      weights = inWeights[0];
      bias = inWeights[1];
   }


   override public function toString() return 'Conv2D($name:$kernelSize x $filters $activation $weights $bias)';


}



