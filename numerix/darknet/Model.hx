package numerix.darknet;

import Sys.println;
import haxe.io.Bytes;
import numerix.Tensor;
import numerix.Nx;

class Params
{
  public var w:Int;
  public var h:Int;
  public var c:Int;
  public var layer:Layer;

  public function new() { }
}

class Model extends numerix.Model
{
   var transpose:Bool;

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

      var params = new Params();

      for(section in sections)
      {
         var layer = createLayer(weights, section, params);
         if (layer!=null)
         {
            params.layer = layer;
            layers.push(layer);
         }
      }

      weights.close();
   }

   function createLayer(file:haxe.io.Input, config:Dynamic, params:Params) : Layer
   {
      var name = config.class_name;
      switch(name)
      {
         case "net" :
            width = params.w = config.width;
            height = params.h = config.height;
            channels = params.c = config.channels;

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
            var c = params.c;
            println('Convolutional $w,$h,$c');

            var buffer = Bytes.alloc(n*4);
            file.readBytes(buffer,0,n*4);
            var bias = Tensor.fromBytes(buffer,Nx.float32,[n]);

            if (config.batch_normalize)
            {
               // scales, mean, variance
               file.readBytes(buffer,0,n*4);
               file.readBytes(buffer,0,n*4);
               file.readBytes(buffer,0,n*4);
            }
            var buffer = Bytes.alloc(n*c*size*size*4);
            file.readBytes(buffer,0,buffer.length);
            var weights = Tensor.fromBytes(buffer,Nx.float32,[n,c,size,size]);
            weights = weights.reorder([0,2,3,1],true);
            println(" weight " + weights.min + "..." + weights.max );
            println('Conv2D $weights $bias');

            conv2D.setWeights([weights,bias]);
            if (conv2D.padding==Layer.PAD_SAME)
            {
               params.w = cpp.NativeMath.idiv(params.w + stride-1,stride);
               params.h = cpp.NativeMath.idiv(params.h + stride-1,stride);
            }
            else
            {
               params.w = cpp.NativeMath.idiv(params.w - stride+1,stride);
               params.h = cpp.NativeMath.idiv(params.h - stride+1,stride);
            }
            params.c = n;
            println(' -> $w,$h,$c');

            return conv2D;

         /*
         case "Conv2D" :
            cfg.activation = cfg.activation;
            cfg.useBias = cfg.use_bias;
            var data = cfg.name;

            var conv2D = new Conv2D(cfg, prev);
            var weights = file.read('model_weights/$data/$data/kernel:0');
            weights = weights.reorder([3,0,1,2],true);
            var bias = conv2D.useBias ? file.read('model_weights/$data/$data/bias:0') : null;
            if (bias!=null)
               bias.setFlat();
            conv2D.setWeights([weights,bias]);
            return conv2D;

         case "SeparableConv2D" :
            var cfg = config.config;
            cfg.kernelSize = cfg.kernel_size;
            cfg.dilation = cfg.dilation_rate;
            cfg.activation = cfg.activation;
            cfg.useBias = cfg.use_bias;
            var data = cfg.name;

            var conv2D = new SeparableConv2D(file, cfg, prev);
            var dweights = file.read('model_weights/$data/$data/depthwise_kernel:0');
            dweights = dweights.reorder([3,2,0,1],true);
            var s = dweights.shape;
            dweights.setShape([s[1],s[2],s[3]]);
            trace(dweights);
            var pweights = file.read('model_weights/$data/$data/pointwise_kernel:0');
            pweights = pweights.reorder([0,1,3,2],true);
            var s = pweights.shape;
            pweights.setShape([s[2],s[3]]);
            trace(pweights);
            var bias = conv2D.useBias ? file.read('model_weights/$data/$data/bias:0') : null;
            if (bias!=null)
               bias.setFlat();
            conv2D.setWeights([dweights,pweights,bias]);
            return conv2D;

         */

         default:
            throw 'Unknown layer type $name';
      }
      return null;
   }
}


