package numerix;

class Pack extends Layer
{
   public var stride:Int;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      var s:Null<Int> = config.stride;
      if (s==null)
         stride = 2;
      else
         stride = s;

      handle = layCreatePack(stride);
   }


   override public function toString() return 'Pack($name:$stride)';

   static var layCreatePack = Loader.load("layCreatePack","io");


}





