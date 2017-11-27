package numerix;

class Reorg extends Layer
{
   static var id = 0;

   public var stride:Int;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);
      if (name==null)
         name = "reorg_" + (id++);

      var s:Null<Int> = config.stride;
      if (s==null)
         stride = 2;
      else
         stride = s;

      handle = layCreateReorg(stride);
   }


   override public function toString() return 'Reorg($name:$stride)';

   static var layCreateReorg = Loader.load("layCreateReorg","io");


}





