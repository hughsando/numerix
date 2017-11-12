package numerix.keras;
import numerix.*;

class Layer
{
   public var name:String;
   public var inputs:Array<Layer>;
   public var outputs:Array<Layer>;
   public var valid:Bool;
   public var validSize:Bool;

   public function new(confg:Dynamic, ?input:Layer)
   {
      inputs = input==null ? [] : [input];
      if (input!=null)
         input.outputs.push(this);
      outputs = [];
      name = confg.name;
      valid = false;
      validSize = false;
   }

   public function invalidate(inSameSize:Bool)
   {
      if (valid)
      {
         valid = false;
         if (!inSameSize)
            validSize = false;
         for(o in outputs)
            o.invalidate(inSameSize);
      }
   }

   public function getOutput() : Tensor
   {
      return null;
   }

   public function toString() return 'Layer($name)';

}


