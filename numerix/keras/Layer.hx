package numerix.keras;

class Layer
{
   public var inputs:Array<Layer>;
   public var outputs:Array<Layer>;
   public var name:String;

   public function new(confg:Dynamic, ?input:Layer)
   {
      inputs = input==null ? [] : [input];
      outputs = [];
      name = confg.name;
   }

   public function toString() return 'Layer($name)';

}


