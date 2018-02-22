package numerix;

class Crop extends Layer
{
   static var id = 0;
   var offsetX(default,null):Int;
   var offsetY(default,null):Int;

   public function new(config:Dynamic, input0:Layer,input1:Layer)
   {
   trace("crop " + input0 + " " + input1);
      super(config,input0,input1);
      if (name==null)
         name = "crop_" + (id++);

      offsetX = 0;
      offsetY = 0;
      var offsets:Array<Int> = config.offset;
      if (offsets!=null && offsets.length>0)
      {
         offsetY = offsets[0];
         offsetX = offsets.length>1 ? offsets[1] : offsetY;
      }

      handle = layCreateCrop(offsetX,offsetY);
   }

   override public function setActivation(inActication:Int)
   {
      inputs[0].setActivation(inActication);
   }


   override public function toString() return 'Crop($name)';

   static var layCreateCrop = Loader.load("layCreateCrop","iio");


}


