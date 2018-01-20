package numerix.caffe;

import Sys.println;
import haxe.io.Bytes;
import numerix.Tensor;
import numerix.Nx;
import cpp.NativeMath.idiv;
using StringTools;

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

class Node
{
   public var inputs:Array<Dynamic>;
   public var outputs:Array<Dynamic>;

   public function new()
   {
      inputs = [];
      outputs = [];
   }
   public function toString()
   {
      var inName = inputs.map( function(x) return x.name );
      var outName = outputs.map( function(x) return x.name );
      return 'Node($inName,$outName)';
   }
}

class Model extends numerix.Model
{
   var transpose:Bool;
   var paramStack:Array<Params>;
   var reorgLayer:Reorg = null;
   var filename:String;
   var weightName:String;

   public function new(inFilename:String)
   {
      super();

      var binName = "";
      var txtName = "";

      if (inFilename.endsWith(".prototxt"))
      {
         txtName = inFilename;
         binName = inFilename.substr(0, inFilename.length - ".prototxt".length) + ".caffemodel";
      }
      else
      {
         binName = inFilename;
      }


      var nodeMap = new Map<String, Node>();
      function makeNode(name:String)
      {
         if (nodeMap.exists(name))
            return nodeMap.get(name);
         var node = new Node();
         nodeMap.set(name,node);
         return node;
      }

      var layers:Array<Dynamic> = caffeLoad(txtName, binName);
      var first:Dynamic = null;
      for(layer in layers)
      {
         var bottom:Array<String> = layer.bottom;
         if (bottom!=null)
            for(b in bottom)
               makeNode(b).outputs.push(layer);
         else
            first = layer;

         var top:Array<String> = layer.top;
         if (top!=null)
            for(t in top)
               makeNode(t).inputs.push(layer);
      }

      println('First $first');

      for(n in nodeMap.keys())
      {
         var node = nodeMap.get(n);
         if (node.outputs.length==0)
            println('OUT node ' + node);
         if (node.inputs.length==0)
            println('IN node ' + node);
      }

   }

   /*
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
            //println('Convolutional $w,$h,$c');

            var buffer = Bytes.alloc(n*4);
            checkData(file);
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
            //println('Maxpool ${params.w},${params.h},${params.channels}');
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
               //println('route1 ${old.w},${old.h},${old.channels}');
               return old;
            }

            var l0:Params = untyped  paramStack[ l + Std.parseInt(lays[0]) ];
            var l1:Params = untyped  paramStack[ l + Std.parseInt(lays[1]) ];
            //println('Concat ${l0.w},${l0.w},${l0.channels+l1.channels}');
            var cc =  new Concat(config,l0.layer,l1.layer);
            return new Params(l0.w,l0.h,l0.channels + l1.channels, cc);

         case "reorg" :
            var stride = Std.parseInt(config.stride);
            var reorg =  new Reorg(config, params.layer);
            var w = idiv(params.w, stride);
            var h = idiv(params.h, stride);
            reorgLayer = reorg;
            return new Params(w,h,params.channels*stride*stride, reorg);

         case "region" :
            var regions = new YoloRegions(config,params.layer);
            return new Params(1,1,1, regions);

         case "ncs" :
            var graphDir = haxe.io.Path.directory(filename);
            if (graphDir=="") graphDir = ".";

            var ncs = new Movidius(config, params.layer, graphDir);
            return new Params(ncs.outputWidth, ncs.outputHeight, ncs.outputChannels, ncs);


         default:
            throw 'Unknown layer type $name';
      }

      return null;
   }
   */

   static var caffeLoad = Loader.load("caffeLoad","sso");
}



