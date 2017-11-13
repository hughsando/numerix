package numerix;

class Model
{
   var inputLayer:InputLayer;
   var outputLayer:Layer;
   var layers:Array<Layer>;

   public function new()
   {
      layers = [];
   }

   public function run(input:Tensor) : Tensor
   {
      if (input==null || outputLayer==null)
         trace("Incomplete model specification");
      inputLayer.set(input);
      return outputLayer.getOutput();
   }

}


