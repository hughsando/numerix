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
   var resizeBuffer:Tensor;

   public function new()
   {
      layers = [];
   }

   public function run(input:Tensor) : Tensor
   {
      if (outputLayer==null && layers.length>0)
         outputLayer = layers[layers.length-1];

      if (inputLayer==null || outputLayer==null)
         trace("Incomplete model specification");

      if (channels!=null)
      {
         var shape = input.shape;
         if (shape.length!=3)
            throw 'This model only supports images with 3 dimensions';
         if (shape[2]!=channels)
            throw 'This model only supports images with $channels channels';
      }
      if (width!=null && height!=null)
      {
         var shape = input.shape;
         if (shape.length!=3)
            throw "This model only supports images with 3 dimensions";
         var h = shape[0];
         var w = shape[1];
         if (width!=w || height!=h)
         {
            resizeBuffer = input.cropAndScale(width,height,resizeBuffer,true);
            input = resizeBuffer;
            trace(input);
         }
      }
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


