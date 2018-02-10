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

   override public function setActivation(inActication:Int)
   {
      for( i in inputs)
         i.setActivation(inActication);
   }


   public function bypass(layer:Layer)
   {
      var slot = inputs.indexOf(layer);
      if (!inputs.remove(layer))
         throw "Concat bypass - layer not found " + layer;

      if (!layer.outputs.remove(this))
         throw "Concat bypass - bad outputs reference ";

      var output = outputs[0];
      unlink();
      return output;
   }


   override public function toString() return 'Concat($name)';

   static var layCreateConcat = Loader.load("layCreateConcat","o");


}






