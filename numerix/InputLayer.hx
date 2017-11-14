package numerix;

class InputLayer extends Layer
{
   var value:Tensor;
   var lastShape:Array<Int>;

   public function new()
   {
      super({name:"Input"});
      lastShape = [];
   }

   public function set(tensor:Tensor)
   {
      value = tensor;
   }

   override public function getOutput() : Tensor
   {
      return value;
   }

   override public function toString() return 'InputLayer($value)';

}



