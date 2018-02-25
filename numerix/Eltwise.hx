package numerix;

class Eltwise extends Layer
{
   public static inline var PROD = 0;
   public static inline var SUM = 1;
   public static inline var MAX = 2;
   public static var opNames  = ["product", "sum", "max"];

   static var id = 0;
   var op(default,null):Int;
   var opName(default,null):String;
   var cropIndex:Int;
   var cropX:Int;
   var cropY:Int;

   public function new(config:Dynamic, input0:Layer,input1:Layer)
   {
      super(config,input0,input1);
      if (name==null)
         name = "eltwise_" + (id++);

      if (config.op==null)
         op = SUM;
      else
      {
         op = config.op;
         if (op<0 || op>2)
            throw "Eltwise - unknown op " + config.op;
      }
      opName = opNames[op];

      cropIndex = 0;
      cropX = 0;
      cropY = 0;

      handle = layCreateEltwise(op);
   }

   public function setCropIndex(inIndex:Int, inDx:Int, inDy:Int)
   {
      cropIndex = inIndex;
      cropX = inDx;
      cropY = inDy;
      laySetCropIndex(handle, cropIndex, cropX, cropY );
   }

   override public function toString() return 'Eltwise($opName $name)';

   static var layCreateEltwise = Loader.load("layCreateEltwise","io");
   static var laySetCropIndex = Loader.load("laySetCropIndex","oiiiv");


}



