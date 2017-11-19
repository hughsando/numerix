import numerix.*;
import Sys.println;

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
      if (bmpname!=null)
      {
         var val = NmeTools.loadImageF32( bmpname, NmeTools.TRANS_STD );
         println( val + ": " + val.min + "..." + val.max );

         var t0 = haxe.Timer.stamp();
         var result = model.run(val);
         println("Time : " + Std.int((haxe.Timer.stamp()-t0)*1000) + "ms");
         println( result.min + "..." + result.max );
         Io.writeFile("result.nx", result);

         NmeTools.saveImageF32(result, "result.png", NmeTools.TRANS_UNIT_SCALE );
      }
   }
}

