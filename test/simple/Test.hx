import numerix.Nx;

class Test
{
   public static function main()
   {
      var t = Nx.tensor([ [[1,2,3], [4,5,6]], [[7,8.1,9],[10,11,12]] ]);
      Sys.print("t : " + t + " = ");
      t.print();
   }
}

