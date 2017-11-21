package numerix;
import numerix.*;
import Sys.println;

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
   public var valid:Bool;
   public var resultBuffer:Tensor;
   var handle:Dynamic;

   public function new(confg:Dynamic, ?input0:Layer,?input1:Layer)
   {
      inputs = input0==null ? [] : [input0];
      if (input0!=null)
         input0.outputs.push(this);
      if (input1!=null)
      {
         inputs.push(input1);
         input1.outputs.push(this);
      }
      outputs = [];
      name = confg.name;
      valid = false;
   }

   public function release()
   {
      layRelease(handle);
      handle = null;
   }

   public function setWeights(inWeights:Array<Tensor>)
   {
   }

   public function invalidateAll()
   {
      if (valid)
      {
         valid = false;
         for(output in outputs)
            output.invalidateAll();
      }
   }

   public function getOutput() : Tensor
   {
      if (handle!=null && !valid)
      {
         if (inputs.length==1)
         {
            if (inputs[0]==null)
               throw "Bad input0 in " + this;
            var src = inputs[0].getOutput();
            if (src==null)
               throw "Bad output :" + this + " " + inputs;
            resultBuffer = Tensor.fromHandle( layRun(handle, this, src) );
         }
         else if (inputs.length==2)
         {
            if (inputs[0]==null || inputs[1]==null)
               throw "Bad inputs in " + this + " " + inputs;
            var src = [ inputs[0].getOutput(), inputs[1].getOutput() ];
            if (src[0]==null || src[1]==null)
               throw "Bad output: " + this + " " + src;
            resultBuffer = Tensor.fromHandle( layRun(handle, this, src) );
         }
         else
         {
            resultBuffer = Tensor.fromHandle( layRun(handle, this, null) );
         }
         valid = true;
      }

      println( resultBuffer + "=" + resultBuffer[0] + "..." + resultBuffer[resultBuffer.channels*resultBuffer.width] );

      return resultBuffer;
   }

   public function toString() return 'Layer($name)';

   static var layRun = Loader.load("layRun","oooo");
   static var layRelease = Loader.load("layRelease","ov");
}


