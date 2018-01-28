package numerix;

class Softmax extends Layer
{
   static var id = 0;

   public function new(config:Dynamic, input:Layer)
   {
      super(config,input);

      if (name==null)
         name = "softmax_" + (id++);

      handle = layCreateSoftmax();
   }

   override public function toString() return 'Softmax($name)';


   override public function getClasses() : Array< Classification >
   {
      var values = resultBuffer;
      if (values==null)
         return null;
      var size = values.elementCount;

      var classNames = size==1000 ? ImageNet.names : null;

      var result = new Array<Classification>();
      for(i in 0...size)
      {
         var r = values[i];
         if (classNames==null)
            result.push({ classId:i, prob:r });
         else
            result.push({ classId:i, prob:r, className:classNames[i] });
      }
      result.sort( function(a,b) return a.prob>b.prob?-1 : 1 );
      return result;
   }


   static var layCreateSoftmax = Loader.load("layCreateSoftmax","o");
}






