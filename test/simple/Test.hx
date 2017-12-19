import numerix.Nx;
import numerix.*;

class Test
{
   public static function main()
   {
      var t = Nx.tensor([ [for(i in 0...15) [for(j in 0...15) i*j]] ]);
      Sys.print("t : " + t + " = ");
      t.print();

      var t = Nx.tensor(10, Nx.int16, [5,5]);
      t.print();

      var idx = 0;
      var t = Nx.tensor([for(i in 0...3) [for(j in 0...3) idx++] ]);
      var trans = t.reorder([1,0]);
      Sys.println("Trans");
      trans.print();

      var idx = 0;
      var t = Nx.tensor([for(k in 0...3) [for(i in 0...3) [for(j in 0...3) idx++] ]]);
      Sys.println("Before");
      t.print();
      var trans = t.reorder([0,2,1]);
      Sys.println("After");
      trans.print();

      var reduced = trans.resizeAxis(0,4,1);
      Sys.println("Reduced");
      reduced.print();

      //testConv();
   }

   static function testConv()
   {
      Model.enableGpu(false);

      var model = new Model();
      var input = model.makeInputLayer();
      var cfg = { activation:'linear', kernelSize:[3,3], filters:4, padding:'same' };
      var conv2D = new Conv2D(cfg,input);
      model.addLayer( conv2D );

      var weight = Nx.zeros( [3,3,3,4] );
      weight.setAt(5, 1);

      conv2D.setWeights( [weight] );

      var input = Nx.zeros( [3,3,4] );
      input.setAt(16, 1);
      input.print();
      model.run(input).print();

   }
}

