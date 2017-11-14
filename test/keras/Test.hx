import numerix.*;
import nme.display.*;

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
            var bmp = BitmapData.loadFromBytes( nme.utils.ByteArray.readFile(bmpname) );
         trace(bmp.width + "x" + bmp.height);
      }
   }
}

