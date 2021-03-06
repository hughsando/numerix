package numerix;
import numerix.Padding;

class MaxPool extends Layer
{
   static var id = 0;

   public var kernelSize(default,null):Array<Int>;
   public var strides(default,null):Array<Int>;
   public var padding(default,null):Padding;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      if (name==null)
         name = "maxpool_" + (id++);


      kernelSize = config.kernelSize;
      strides = config.strides;

      var pad:String = config.padding;
      if (pad=="same")
         padding = PadSame;
      else if (pad=="valid")
         padding = PadValid;
      else if (pad=="custom")
         padding = PadCustom(config.pad);
      else
         throw 'Unknown padding $pad';

      handle = layCreateMaxPool(kernelSize, strides, Layer.encodePadding(padding) );
   }


   override public function toString() return 'MaxPool($name:$kernelSize/$strides)';

   static var layCreateMaxPool = Loader.load("layCreateMaxPool","ooio");


}




