package numerix;

class Concat extends Layer
{
   static var concatId = 0;

   public function new(config:Dynamic, input0:Layer,input1:Layer)
   {
      super(config,input0,input1);
      if (name==null)
         name = "concat_" + (concatId++);

      handle = layCreateConcat();
   }


   override public function toString() return 'Concat($name)';

   static var layCreateConcat = Loader.load("layCreateConcat","o");


}






