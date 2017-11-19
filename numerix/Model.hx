package numerix;

using StringTools;

class Model
{
   var inputLayer:InputLayer;
   var outputLayer:Layer;
   var layers:Array<Layer>;
   var width:Null<Int>;
   var height:Null<Int>;
   var channels:Null<Int>;

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

   public static function load(modelname:String) : Model
   {
      #if hxhdf5
      if (modelname.endsWith(".h5"))
         return new numerix.keras.Model(modelname);
      #else
      if (modelname.endsWith(".h5"))
         throw ".h5 files require hxhdf5 lib"
      #end

      if (modelname.endsWith(".cfg"))
         return new numerix.darknet.Model(modelname);

      throw "Could not deduce model type from filename";
      return null;
   }

}


