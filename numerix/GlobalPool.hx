package numerix;

class GlobalPool extends Layer
{
   static var id = 0;
   var average:Bool;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      average = true;

      if (name==null)
         name = "globalpool_" + (id++);

      handle = layCreateGlobalPool(average);
   }

   override public function toString() return 'GlobalPool(' + (average?"Averge":"Max") + ')';


   static var layCreateGlobalPool = Loader.load("layCreateGlobalPool","bo");
}




