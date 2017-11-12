import numerix.*;

class Test
{
   public static function main()
   {
      var args = Sys.args();
      var filename = args.pop();
      if (filename==null)
      {
         Sys.println("Usage: Test filename");
         return;
      }

      var model = new numerix.keras.Model(filename);
   }
}

