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

      var O = 1;
      var I = 1;
      var H = 2;
      var W = 3;


      var img = Tensor.create(null, Nx.float32, [H,W,I]);

      /*
      var idx = 0;
      for(y in 0...H)
         for(x in 0...W)
            for(chan in 0...I)
               img.set(idx++, ( ((chan*9+12)%13) + ((y*39 + 6)%17) + ((x*37 + 8)%19)  ) * 0.1 );
      */
      //img.set( (1 * W + 1) * I, 1);
      img.set( 0, 1.0);

      trace(img);

      var result = model.run(img);

      trace(result);

      // convert to caffe order
      result = result.reorder([2,0,1]);
      trace(" -> " + result);

      result.print(100000);
   }
}

