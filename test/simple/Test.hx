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

     for(size in 1...9)
     {
      for(inputs in 1...4)
         for(outputs in 1...4)
         {
            var inCount = inputs*4;
            var outCount = outputs*4;
            var res = [];

            var weight = Nx.zeros( [outCount,3,3,inCount] );
            //for(i in 0...weight.elementCount)
            //   weight.setAt(i, Math.random()-0.5 );
            weight[ (3+2)*inCount ] = 1;

            var input = Nx.zeros( [size,size,inCount] );
            for(i in 0...input.elementCount)
               input[i] = Math.random()-0.5;

            for(act in ['leaky', 'linear', 'relu'])
            {
               for(go in 0...2)
               {
                  var cfg = { activation:act, kernelSize:[3,3], filters:outCount, padding:'same',
                              allowTransform: go==1 };

                  var model = new Model();
                  var inputLayer = model.makeInputLayer();
                  var conv2D = new Conv2D(cfg,inputLayer);
                  model.addLayer( conv2D );

                  conv2D.setWeights( [weight] );

                  res[go] = model.run(input);
               }

               var ok = true;
               for(i in 0...res[0].elementCount)
                  if ( Math.abs(res[0][i]-res[1][i]) > 0.001 )
                  {
                     Sys.println("  err " + i + ":" + res[0][i] + "/" + res[1][i] );
                     ok = false;
                     Sys.println("no trans");
                     res[0].print();
                     Sys.println("trans");
                     res[1].print();
                     Sys.println('$size: $inCount $outCount $act ' + res[0].shape + " " + (ok?"ok":"bad"));
                     Sys.exit(-1);
                  }
               Sys.println('$inCount $outCount $act ' + res[0].shape + " " + (ok?"ok":"bad"));
            }
         }
      }
   }
}

