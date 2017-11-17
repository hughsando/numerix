package numerix.keras;
using numerix.Hdf5Tools;

import Sys.println;
import hdf5.*;

class Model extends numerix.Model
{
   public function new(filename:String)
   {
      super();

      var file = hdf5.File.open(filename);
      var attribs = haxe.Json.parse( file.attribs.model_config );
      var className = attribs.class_name;
      if (className!="Sequential")
         throw "Only sequential models supported";

      //Sys.println("Data:\n" + file.getItemsRecurse()
      //        .filter( i -> i.type==DatasetItem ).map( i -> return i.path ).join("\n"));

      try
      {
         inputLayer = new InputLayer();
         var prev:Layer = inputLayer;
         for(l in (attribs.config:Array<Dynamic>))
         {
            var layer = createLayer(file,l,prev);
            prev = layer;
            layers.push(layer);
         }
         outputLayer = prev;
      }
      catch(e:Dynamic)
      {
         println("Error in model.  layers:");
         for(l in (attribs.config:Array<Dynamic>))
            println("   " + l.config);
         println("datasets:");
         println("   " + Hdf5Tools.listDatasets(file).join("\n   "));
         file.close();
         throw(e);
      }
   }

   function createLayer(file:Group, config:Dynamic, prev:Layer) : Layer
   {
      var name = config.class_name;
      switch(name)
      {
         case "Conv2D" :
            var cfg = config.config;
            cfg.kernelSize = cfg.kernel_size;
            cfg.dilation = cfg.dilation_rate;
            cfg.activation = cfg.activation;
            cfg.useBias = cfg.use_bias;
            var data = cfg.name;

            var conv2D = new Conv2D(file, cfg, prev);
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



         default:
            throw 'Unknown layer type $name';
      }
      return null;
   }
}

