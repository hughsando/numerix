import numerix.Nx;
import numerix.*;

class Test
{
   public static function main()
   {
      testConv();
   }

   public static function testCreate()
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
   }


   static function testConv()
   {
      Model.enableGpu(false);

      var model = new Model();
      var input = model.makeInputLayer();
      var cfg = { activation:'linear', kernelSize:[3,3], filters:4, padding:'same' };
      var conv2D = new Conv2D(cfg,input);
      model.addLayer( conv2D );

      var inDim = 8;
      var outDim = 8;
      var weight = Nx.zeros( [outDim,3,3,inDim] );
      for(o in 0...outDim)
         weight.setAt(inDim*(3+1)+ o*3*3*inDim, (o+1));

      //weight.setAt(16+3*3*4 + 1, 1);
      //weight.setAt(16+3*3*4*2 + 2, 1);
      //weight.setAt(16+3*3*4*3 + 3, 1);
      //for(i in 0...4*4*3*3)
      //   weight.setAt(i,i);

      conv2D.setWeights( [weight] );

      var input = Nx.zeros( [3,3,inDim] );
      input.setAt(inDim*(3+1), 1);
      //for(i in 0...3*3*inDim)
      //   input.setAt(i, i);
      input.print();
      var res = model.run(input);
      trace(res);
      res.print();
   }
}

