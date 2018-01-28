package numerix;

class YoloRegions extends Layer
{
   static var coco_classes = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

   //static var  coco_ids = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90];

   static var id = 0;

   var classCount:Int;
   var classNames:Array<String>;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      if (name==null)
         name = "yolo_" + (id++);

      var anchors:String = config.anchors;
      var anchors = [for(a in anchors.split(",")) Std.parseFloat(a)];
      var num = Std.parseInt(config.num);
      classCount = Std.parseInt(config.classes);
      if (config.classNames==null)
         classNames = CocoClasses.names;
      else
         classNames = Std.string(config.classNames).split(",");

      handle = layCreateYolo(anchors,num,classCount);
   }

   
   override public function getBoxes() : Array< {x:Float, y:Float, w:Float, h:Float, prob:Float, classId:Int, className:String } >
   {
      var result = super.getBoxes();
      for(box in result)
         untyped box.className = classNames[ box.classId ];
      return cast result;
   }


   override public function toString() return 'Yolo($name:$classCount)';

   static var layCreateYolo = Loader.load("layCreateYolo","oiio");
}




