package numerix;

class Concat extends Layer
{
   public function new(config:Dynamic, input0:Layer,input1:Layer)
   {
      super(config,input0,input1);

      handle = layCreateConcat();
   }


   override public function toString() return 'Ccncat($name)';

   static var layCreateConcat = Loader.load("layCreateConcat","o");


}






