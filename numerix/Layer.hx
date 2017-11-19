package numerix;
import numerix.*;

class Layer
{
   static public inline var PAD_SAME = 0;
   static public inline var PAD_VALID = 1;

   static public inline var ACT_LINEAR = 0;
   static public inline var ACT_RELU = 1;
   static public inline var ACT_SIGMOID = 2;
   static public inline var ACT_LEAKY = 3;



   public var name:String;
   public var inputs:Array<Layer>;
   public var outputs:Array<Layer>;
   public var resultBuffer:Tensor;
   var handle:Dynamic;

   public function new(confg:Dynamic, ?input:Layer)
   {
      inputs = input==null ? [] : [input];
      if (input!=null)
         input.outputs.push(this);
      outputs = [];
      name = confg.name;
   }

   public function release()
   {
      layRelease(handle);
      handle = null;
   }

   public function setWeights(inWeights:Array<Tensor>)
   {
   }

   public function getOutput() : Tensor
   {
      if (handle!=null)
      {
         var src = inputs.length>0 ? inputs[0].getOutput() : null;
         resultBuffer = Tensor.fromHandle( layRun(handle, this, src) );
      }

      return resultBuffer;
   }

   public function toString() return 'Layer($name)';

   static var layRun = Loader.load("layRun","oooo");
   static var layRelease = Loader.load("layRelease","ov");
}


