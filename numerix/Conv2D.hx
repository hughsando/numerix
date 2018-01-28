package numerix;

class Conv2D extends Layer
{
   static var conv2DId = 0;
   public static var defaultAllowTransform = true;

   public var kernelSize(default,null):Array<Int>;
   public var dilation(default,null):Array<Int>;
   public var strides(default,null):Array<Int>;
   public var filters(default,null):Int;
   public var useBias(default,null):Bool;
   public var activation(default,null):Int;
   public var padding(default,null):Int;
   public var weights(default,null):Tensor;
   public var bias(default,null):Tensor;
   public var allowTransform(default,null):Bool;
   public var inputChannels(default,null):Int;

   public var scales(default,null):Tensor;
   public var means(default,null):Tensor;
   public var vars(default,null):Tensor;


   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      if (name==null)
         name = "conv2d_" + (conv2DId++);


      kernelSize = config.kernelSize;
      dilation = config.dilation;
      strides = config.strides;
      filters = config.filters;
      useBias = config.useBias;
      inputChannels = 0;
      if (config.allowTransform==null)
         allowTransform = defaultAllowTransform;
      else
         allowTransform = config.allowTransform;

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

   public function setActivation(inActication:Int)
   {
      activation = inActication;
      layConv2DSetActication(handle,inActication);
   }

   public function setNormalization(inScales:Tensor, inMeans:Tensor, inVars:Tensor)
   {
      scales = inScales;
      means = inMeans;
      vars = inVars;
      if (handle!=null)
         layConv2DSetNorm(handle, scales, means, vars);
   }


   override public function setWeights(inWeights:Array<Tensor>)
   {
      weights = inWeights[0];
      var s = weights.shape;
      kernelSize = [ s[2], s[1] ];
      filters = s[0];
      inputChannels = s[3];

      bias = inWeights[1];
      release();

      handle = layCreateConv2D(strides, activation, padding, weights, null, bias, allowTransform);
      if (scales!=null)
         layConv2DSetNorm(handle, scales, means, vars);
   }


   //override public function toString() return 'Conv2D($name:$kernelSize x $filters $activation $weights $bias)';
   override public function toString()
   {
      var w = 0;
      var h = 0;
      if (kernelSize.length==2)
      {
         w= kernelSize[0];
         h= kernelSize[1];
      }
      else
         w = h = kernelSize[0];

      return 'Conv2D($name: $w x $h x $filters)';
   }



   static var layCreateConv2D = Loader.load("layCreateConv2D","oiiooobo");
   static var layConv2DSetNorm = Loader.load("layConv2DSetNorm","oooov");
   static var layConv2DSetActication = Loader.load("layConv2DSetActication","oiv");


}



