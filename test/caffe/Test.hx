import numerix.*;
import Sys.println;
using StringTools;

class Test
{
   public static function main()
   {
      numerix.Model.enableGpu(false);

      var args = Sys.args();
      if (args.remove("-opencl"))
      {
         var platforms = numerix.opencl.ClCtx.platforms;
         // todo - 
         var platform = platforms[0];
         Sys.println("Using opencl platform " + platform.name);
         var devices = null;
         var devices = [ platform.devices[0] ];
         var ctx = new numerix.opencl.ClCtx(platform,devices);
         trace( ctx );
      }

      var modelname = "mynet.caffemodel";
      var model = numerix.Model.load(modelname);

      var I = 1;
      var H = 2;
      var W = 2;


      var img = Tensor.create( 0.0, Nx.float32, [H,W,I]);

      var idx = 0;
      //for(y in 0...H)
      //   for(x in 0...W)
      //      for(chan in 0...I)
      //         img.set(idx++, ( ((chan*9+12)%13) + ((y*39 + 6)%17) + ((x*37 + 8)%19)  ) * 0.1 );
      img.set( (0 * W + 1) * I, 1);
      //img.set( 0, 1.0);

      trace(img);

      var result = model.run(img,false);

      trace(result);

      // convert to caffe order
      result = result.reorder([2,0,1]);
      trace(" -> " + result);

      result.print(100000);
   }
}

