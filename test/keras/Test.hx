import numerix.*;
import Sys.println;

class Test
{
   public static function main()
   {
      var args = Sys.args();
      var filename = args.shift();
      if (filename==null)
      {
         Sys.println("Usage: Test filename");
         return;
      }

      var model = new numerix.keras.Model(filename);

      var bmpname = args.shift();
      if (bmpname!=null)
      {
         var val = NmeTools.loadImageF32( bmpname, NmeTools.TRANS_STD );
         println( val + ": " + val.min + "..." + val.max );

         var result = model.run(val);
         println(result);
      }
   }
}

