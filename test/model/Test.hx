import numerix.*;
import Sys.println;
using StringTools;

class Test
{
   public static function main()
   {
      var args = Sys.args();

      var cpu = args.remove("-cpu");
      if (cpu)
         numerix.Model.enableGpu(false);
      var doTime = !args.remove("-notime");
      var allowResize = !args.remove("-noresize");
      var showResults = args.remove("-showresults");
      var loop = args.remove("-loop");

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

         // mean_value: 104
         // mean_value: 117
         // mean_value: 123

         //var scaling = NmeTools.TRANS_STD;
         //var scaling = NmeTools.TRANS_UNIT_SCALE;
         var scaling = NmeTools.TRANS_UNSCALED;
         val = NmeTools.loadImageF32( bmpname, scaling );
         var pixels = val.width * val.height;
         var idx = 0;
         for(p in 0...pixels)
         {
            val[idx] = val[idx]-104; idx++;
            val[idx] = val[idx]-117; idx++;
            val[idx] = val[idx]-123; idx++;
         }
      }
      #end

      if (val!=null)
      {
         println( val + ": " + val.min + "..." + val.max + "=" + val[0] );
         println("Warmup...");

         var t0 = haxe.Timer.stamp();
         var result = model.run(val,allowResize);
         println("Warmup Time : " + Std.int((haxe.Timer.stamp()-t0)*1000) + "ms");
         if (doTime)
         {
            if (!loop)
               Layer.enablePerLayerTiming();
            var max = loop ? 100000000 : 100;
            var t0 = haxe.Timer.stamp();
            for(go in 0...max)
            {
               if (go>0 && (go%100)==0)
               {
                  var time = (haxe.Timer.stamp()-t0);
                  println("Time : " + Std.int(time*10) + "ms");
                  t0 = haxe.Timer.stamp();
               }
               result = model.run(val,allowResize);
            }

            // Final pass displays layer times
            var time = (haxe.Timer.stamp()-t0);
            Layer.showTimes = true;
            result = model.run(val,allowResize);
            println("Time 100: " + Std.int(time*10) + "ms");
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
         if (showResults)
         {
            for(layer in model.layers)
            {
               var result = "?";
               var shape = [];
               if (layer.resultBuffer!=null)
               {
                  shape = layer.resultBuffer.shape;
                  if (shape.length==3)
                     result = Std.string(layer.resultBuffer.get(shape[0]-1,shape[1]-1,shape[2]-1));
                  else if (shape.length==2)
                     result = Std.string(layer.resultBuffer.get(shape[0]-1,shape[1]-1));
                  else
                     result = Std.string(layer.resultBuffer[shape[0]-1]);
               }
               println(layer.name + " " + shape + " = " + result);
            }
         }

         #if nme
         if (result.channels==3)
            NmeTools.saveImageF32(result, "result.png", NmeTools.TRANS_UNIT_SCALE );
         #end
      }
   }
}

