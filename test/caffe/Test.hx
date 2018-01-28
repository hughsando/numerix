import numerix.*;
import Sys.println;
using StringTools;

class Test
{
   public static function main()
   {
      numerix.Model.enableGpu(false);
      var modelname = "mynet.caffemodel";
      var model = numerix.Model.load(modelname);

      var img = Tensor.create(null, Nx.float32, [4,6,3]);

      var idx = 0;
      for(y in 0...4)
         for(x in 0...6)
            for(chan in 0...3)
               img.set(idx++, ( ((chan*9+12)%13) + ((y*39 + 6)%17) + ((x*37 + 8)%19)  ) * 0.1 );


      var result = model.run(img);

      trace(result);

      // convert to caffe order
      result = result.reorder([2,0,1]);
      trace(" -> " + result);

      result.print(100000);
   }
}

