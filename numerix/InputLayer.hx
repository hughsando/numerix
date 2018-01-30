package numerix;

class InputLayer extends Layer
{
   var lastShape:Array<Int>;

   public function new()
   {
      super({name:"Input"});
      lastShape = [];
   }

   public function set(tensor:Tensor)
   {
      resultBuffer = tensor;
      valid = true;
      invalidateAll();
   }

   override public function getOutput() : Tensor
   {
      return resultBuffer;
   }

   override public function toString() return 'InputLayer($resultBuffer)';

}



