package numerix.darknet;

import Sys.println;
import haxe.io.Bytes;
import numerix.Tensor;
import numerix.Nx;
import cpp.NativeMath.idiv;

class Params
{
  public var w:Int;
  public var h:Int;
  public var channels:Int;
  public var layer:Layer;

  public function new(inW=0, inH=0, inC=0, inLayer=null)
  {
     w = inW;
     h = inH;
     channels = inC;
     layer = inLayer;
  }
}

class Model extends numerix.Model
{
   var transpose:Bool;
   var paramStack:Array<Params>;

   public function new(filename:String)
   {
      super();

      var cfg = sys.io.File.getContent(filename);
      var lines = cfg.split("\r").join("").split("\n");

      var sections = new Array<Dynamic>();
      var section:Dynamic = null;
      var iniMatch = ~/\[(.*)\]/;
      var eqMatch = ~/(.*)=(.*)/;

      for(line in lines)
      {
         if (line.substr(0,1)=="#")
         {
            // ignore
         }
         else if (iniMatch.match(line))
         {
            if (section!=null)
               sections.push(section);
            section = { class_name : iniMatch.matched(1) };
         }
         else if (eqMatch.match(line) && section!=null)
         {
            Reflect.setField(section, eqMatch.matched(1), eqMatch.matched(2) );
         }
      }

      if (section!=null)
         sections.push(section);

      var weightName = filename.substr(0,filename.length-3) + "weights";
      var weights:sys.io.FileInput = null;
      try {
         weights = sys.io.File.read(weightName);
         weights.bigEndian = false;
      }
      catch(e:Dynamic)
      {
         throw "Could not open associated weights file '"+weightName+"'";
      }
      var major = weights.readInt32();
      var minor = weights.readInt32();
      var revision = weights.readInt32();
      println('version: $major.$minor.$revision');
      if (major*10+minor>=2)
      {
         // Skip 64
         weights.readInt32();
         weights.readInt32();
      }
      else
      {
         var seen = weights.readInt32();
      }
      transpose = (major > 1000) || (minor > 1000);

      paramStack = [];
      var current:Params = null;
      for(section in sections)
      {
         current = createLayer(weights, section, current);
         paramStack.push(current);
         if (current.layer!=null)
            layers.push(current.layer);
      }

      weights.close();
      outputLayer = current.layer;
   }

   function createLayer(file:haxe.io.Input, config:Dynamic, params:Params) : Params
   {
      var name = config.class_name;
      switch(name)
      {
         case "net" :
            width = config.width;
            height = config.height;
            channels = config.channels;
            inputLayer = new InputLayer();
            return new Params(width, height, channels, inputLayer);

         case "convolutional" :
            var size:Int = config.size;
            config.kernelSize = [ size,size ];
            var stride:Int = config.stride;
            config.strides = [ stride,stride ];
            config.padding = config.pad==1 ? "same" : "valid";
            config.useBias = true;
            if (config.activation==null)
               config.activation = "logistic";

            var conv2D = new Conv2D(config, params.layer);
            var n:Int = config.filters;
            var w = params.w;
            var h = params.h;
            var c = params.channels;
            println('Convolutional $w,$h,$c');

            var buffer = Bytes.alloc(n*4);
            file.readBytes(buffer,0,n*4);
            var bias = Tensor.fromBytes(buffer,Nx.float32,[n]);

            if (config.batch_normalize)
            {
               // output scales, mean, variance
               file.readBytes(buffer,0,n*4);
               var scales = Tensor.fromBytes(buffer,Nx.float32,[n]);
               file.readBytes(buffer,0,n*4);
               var means = Tensor.fromBytes(buffer,Nx.float32,[n]);
               file.readBytes(buffer,0,n*4);
               var vars = Tensor.fromBytes(buffer,Nx.float32,[n]);

               conv2D.setNormalization(scales, means, vars);
            }
            var buffer = Bytes.alloc(n*c*size*size*4);
            file.readBytes(buffer,0,buffer.length);
            var weights = Tensor.fromBytes(buffer,Nx.float32,[n,c,size,size]);
            weights = weights.reorder([0,2,3,1],true);
            //println(" weight " + weights.min + "..." + weights.max );
            //println('Conv2D $weights $bias');

            conv2D.setWeights([weights,bias]);
            if (conv2D.padding==Layer.PAD_SAME)
            {
               return new Params(
                   idiv(params.w + stride-1,stride),
                   idiv(params.h + stride-1,stride),
                   n, conv2D );
            }
            else
            {
               return new Params(
                  idiv(params.w - size+1,stride),
                  idiv(params.h - size+1,stride),
                   n, conv2D );
            }

         case "maxpool" :
            var size:Int = config.size;
            config.kernelSize = [ size,size ];
            var stride:Int = config.stride;
            config.strides = [ stride,stride ];
            config.padding = 'same';
            println('Maxpool ${params.w},${params.h},${params.channels}');
            var maxPool = new MaxPool(config, params.layer);
            if (maxPool.padding==Layer.PAD_SAME)
            {
               return new Params(
                 idiv(params.w + stride-1,stride),
                 idiv(params.h + stride-1,stride),
                 params.channels, maxPool );
            }
            else
            {
               return new Params(
                 idiv(params.w - size+1 + stride-1,stride),
                 idiv(params.h - size+1 + stride-1,stride),
                 params.channels, maxPool );
            }

         case "route" :
            var lays = config.layers.split(",");
            var l = paramStack.length;
            if (lays.length==1)
            {
               var old:Params = untyped paramStack[ l + Std.parseInt(lays[0]) ];
               println('route1 ${old.w},${old.h},${old.channels}');
               return old;
            }

            var l0:Params = untyped  paramStack[ l + Std.parseInt(lays[0]) ];
            var l1:Params = untyped  paramStack[ l + Std.parseInt(lays[1]) ];
            println('Concat ${l0.w},${l0.w},${l0.channels+l1.channels}');
            var cc =  new Concat(config,l0.layer,l1.layer);
            return new Params(l0.w,l0.h,l0.channels + l1.channels, cc);

         case "reorg" :
            var stride = Std.parseInt(config.stride);
            var reorg =  new Pack(config, params.layer);
            var w = idiv(params.w, stride);
            var h = idiv(params.h, stride);
            return new Params(w,h,params.channels*stride*stride, reorg);

         case "region" :
            var regions = new YoloRegions(config,params.layer);
            return new Params(1,1,1, regions);



         default:
            throw 'Unknown layer type $name';
      }

      return null;
   }
}


