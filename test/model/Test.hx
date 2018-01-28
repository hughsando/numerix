import numerix.*;
import Sys.println;
using StringTools;

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

      var cpu = args.remove("-cpu");
      if (cpu)
         numerix.Model.enableGpu(false);
      var doTime = !args.remove("-notime");

      for(a in args)
      {
         if (a.startsWith("-opencl"))
         {
            args.remove(a);
            var parts = a.split("=");
            var platforms = numerix.opencl.ClCtx.platforms;
            if (parts.length!=2)
            {
               println("use -opencl=#.#, choose # from:");
               var pid = 0;
               for(p in platforms)
               {
                  println( " " + pid + "] " + p.name );
                  var did = 0;
                  for(d in p.devices)
                  {
                     println( "    " + pid + "." + did + "] " + d.name + "(" + d.type + "x" + d.computeUnits + ")" );
                     did ++;
                  }

                  pid++;
               }
               Sys.exit(0);
            }

            var val = parts[1];
            parts = val.split(".");
            var id = Std.parseInt(parts[0]);
            var platform = platforms[id];
            Sys.println("Using opencl platform " + platform.name);
            var devices = null;
            if (parts.length==2)
               devices = [ platform.devices[ Std.parseInt(parts[1]) ] ];
            var ctx = new numerix.opencl.ClCtx(platform,devices);
            trace( ctx );
         }
      }

      if (args.remove("-notrans"))
         numerix.Conv2D.defaultAllowTransform = false;

      var modelname = args.shift();
      if (modelname==null)
      {
         Sys.println("Usage: Test modelname [imagename]");
         return;
      }

      var model = numerix.Model.load(modelname);

      var bmpname = args.shift();
      var val:Tensor = null;

      if (bmpname!=null && bmpname.endsWith(".dat"))
      {
         var bytes = sys.io.File.getBytes(bmpname);
         if (bytes.length!=3*model.height*model.width)
            throw "Length of .dat file does not match specified model file size " + model.width + "x" + model.height + "x3";
         val = Tensor.fromBytes(bytes,Nx.float32, [3, model.height, model.width] );
         val = val.reorder([1,2,0],true);
      }
      #if nme
      else if (bmpname!=null)
      {
         //var scaling = NmeTools.TRANS_STD;
         var scaling = NmeTools.TRANS_UNIT_SCALE;
         val = NmeTools.loadImageF32( bmpname, scaling );
      }
      #end

      if (val!=null)
      {
         println( val + ": " + val.min + "..." + val.max + "=" + val[0] );
         println("Warmup...");

         var t0 = haxe.Timer.stamp();
         var result = model.run(val);
         println("Warmup Time : " + Std.int((haxe.Timer.stamp()-t0)*1000) + "ms");
         if (doTime)
         {
            var t0 = haxe.Timer.stamp();
            for(go in 0...10)
               result = model.run(val);
            var time = (haxe.Timer.stamp()-t0);
            Layer.showTimes = true;
            result = model.run(val);
            println("Time 10: " + Std.int(time*100) + "ms");
         }

         var boxes = model.outputLayer.getBoxes();
         if (boxes!=null &&boxes.length>0)
         {
            for(box in boxes)
               println(box.className + ":" + box.prob + "  " + box.w + "x" + box.h);
         }
         else
         {
            var classes = model.outputLayer.getClasses();
            if (classes!=null && classes.length>0)
            {
               for(c in 0...classes.length)
               {
                  if (c>4) break;
                  var cls = classes[c];
                  println(c + " " + cls);
               }
            }
            else
            {
               println( result );
               println( result.min + "..." + result.max );
            }
         }
         Io.writeFile("result.nx", result);

         #if nme
         if (result.channels==3)
            NmeTools.saveImageF32(result, "result.png", NmeTools.TRANS_UNIT_SCALE );
         #end
      }
   }
}

