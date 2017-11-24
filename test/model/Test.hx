import numerix.*;
import Sys.println;
using StringTools;

class Test
{
   static function testImage()
   {
      var floats = [ 0,0,0, 0,0,0,   0,0,0, 0,0,0, 0,0,0,
                     0,0,0, 0,0,0,   0,0,0, 0,0,0, 0,0,0,
                     0,0,0, 1.0,0,0, 0,0,0, 0,0,0, 0,0,0,
                     0,0,0, 0,0,0,   0,0,0, 0,0,0, 0,0,0 ];

      return Nx.array( floats, Nx.float32, [4,5,3] );
   }

   public static function main()
   {
      var args = Sys.args();
      var modelname = args.shift();
      if (modelname==null)
      {
         Sys.println("Usage: Test modelname [imagename]");
         return;
      }

      var model = numerix.Model.load(modelname);

      var bmpname = args.shift();
      var val:Tensor = null;

      if (bmpname!=null && bmpname.endsWith(".dat"))
      {
         var bytes = sys.io.File.getBytes(bmpname);
         val = Tensor.fromBytes(bytes,Nx.float32, [3, model.height, model.width] );
         val = val.reorder([1,2,0],true);
      }
      else if (bmpname!=null)
      {
         //var scaling = NmeTools.TRANS_STD;
         var scaling = NmeTools.TRANS_UNIT_SCALE;
         val = NmeTools.loadImageF32( bmpname, scaling );
      }

      if (val!=null)
      {
         println( val + ": " + val.min + "..." + val.max + "=" + val[0] );
         println("Warmup...");

         var t0 = haxe.Timer.stamp();
         var result = model.run(val);
         println("Warmup Time : " + Std.int((haxe.Timer.stamp()-t0)*1000) + "ms");
         var t0 = haxe.Timer.stamp();
         var result = model.run(val);
         println("Time : " + Std.int((haxe.Timer.stamp()-t0)*1000) + "ms");

         var boxes = model.outputLayer.getBoxes();
         trace(boxes);
         println( result );
         println( result.min + "..." + result.max );
         Io.writeFile("result.nx", result);

         if (result.channels==3)
            NmeTools.saveImageF32(result, "result.png", NmeTools.TRANS_UNIT_SCALE );
      }
   }
}

