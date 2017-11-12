import numerix.Nx;
using numerix.Hdf5Tools;

class Test
{
   public static function main()
   {
      var args = Sys.args();

      var t = Nx.tensor([ [[1,2,3], [4,5,6]], [[7,8.1,9],[10,11,12]] ], Nx.float32);
      Sys.print("t : " + t + " = ");
      t.print();

      var t = Nx.tensor(10, Nx.int16, [5,5]);
      t.print();

      if (args[0]!=null)
      {
         var file = hdf5.File.open(args[0]);
         if (args[1]==null)
         {
            Sys.println("Choose:\n" + file.getItemsRecurse()
              .filter( i -> i.type==DatasetItem ).map( i -> return i.path ).join("\n")
            );
         }
         else
         {
            var t = file.read(args[1]);
            t.print();
         }
      }
   }
}

