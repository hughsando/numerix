import numerix.Nx;
using numerix.Hdf5Tools;

class Test
{
   public static function main()
   {

      var t = Nx.tensor([ [for(i in 0...15) [for(j in 0...15) i*j]] ]);
      Sys.print("t : " + t + " = ");
      t.print();

      var t = Nx.tensor(10, Nx.int16, [5,5]);
      t.print();
   }
}

