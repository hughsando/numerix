package numerix;

class MaxPool extends Layer
{
   static var id = 0;

   public var kernelSize(default,null):Array<Int>;
   public var strides(default,null):Array<Int>;
   public var padding(default,null):Int;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      if (name==null)
         name = "maxpool_" + (id++);


      kernelSize = config.kernelSize;
      strides = config.strides;

      var pad:String = config.padding;
      if (pad=="same")
         padding = Layer.PAD_SAME;
      else if (pad=="valid")
         padding = Layer.PAD_VALID;
      else
         throw 'Unknown padding $pad';

      handle = layCreateMaxPool(kernelSize, strides, padding);
   }


   override public function toString() return 'MaxPool($name:$kernelSize/$strides)';

   static var layCreateMaxPool = Loader.load("layCreateMaxPool","ooio");


}




