package numerix;

class Conv2D extends Layer
{

   public var kernelSize(default,null):Array<Int>;
   public var dilation(default,null):Array<Int>;
   public var strides(default,null):Array<Int>;
   public var filters(default,null):Int;
   public var useBias(default,null):Bool;
   public var activation(default,null):Int;
   public var padding(default,null):Int;
   public var weights(default,null):Tensor;
   public var bias(default,null):Tensor;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      kernelSize = config.kernelSize;
      dilation = config.dilation;
      strides = config.strides;
      filters = config.filters;
      useBias = config.useBias;

      var act:String = config.activation;
      if (act=="linear" || act=="" || act==null)
         activation = Layer.ACT_LINEAR;
      else if (act=="relu")
         activation = Layer.ACT_RELU;
      else if (act=="sigmoid" || act=="logistic")
         activation = Layer.ACT_SIGMOID;
      else if (act=="leaky")
         activation = Layer.ACT_LEAKY;
      else
         throw 'Unknown activation $act';


      var pad:String = config.padding;
      if (pad=="same")
         padding = Layer.PAD_SAME;
      else if (pad=="valid")
         padding = Layer.PAD_VALID;
      else
         throw 'Unknown padding $pad';
   }


   override public function setWeights(inWeights:Array<Tensor>)
   {
      // TODO - set filters/kernel
      weights = inWeights[0];
      bias = inWeights[1];
      release();

      handle = layCreateConv2D(strides, activation, padding, weights, null, bias);
   }


   override public function toString() return 'Conv2D($name:$kernelSize x $filters $activation $weights $bias)';



   static var layCreateConv2D = Loader.load("layCreateConv2D","oiioooo");


}



