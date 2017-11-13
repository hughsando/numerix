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
      var inShape = tensor.shape;
      var sameSize = inShape.length == lastShape.length;
      if (sameSize)
      {
         for(i in 0...inShape.length)
            if (inShape[i]!=lastShape[i])
            {
                sameSize = false;
                lastShape = inShape;
                break;
            }
      }
      for(o in outputs)
         o.invalidate(sameSize);
   }

   override public function getOutput() : Tensor
   {
      return value;
   }

   override public function toString() return 'InputLayer($value)';

}



