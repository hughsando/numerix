package numerix;

class Pack extends Layer
{
   static var packId = 0;

   public var stride:Int;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);
      if (name==null)
         name = "pack_" + (packId++);

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





