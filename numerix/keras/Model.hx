package numerix.keras;

using numerix.Hdf5Tools;
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

      inputLayer = new InputLayer();
      var prev:Layer = inputLayer;
      for(l in (attribs.config:Array<Dynamic>))
      {
         var layer = createLayer(file,l,prev);
         prev = layer;
         layers.push(layer);
      }
      outputLayer = prev;
      file.close();

      trace(layers);
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
            weights = weights.reorder([2,0,1,3],true);
            var bias = conv2D.useBias ? file.read('model_weights/$data/$data/bias:0') : null;
            conv2D.setWeights([weights,bias]);
            return conv2D;

         default:
            throw 'Unknown layer type $name';
      }
      return null;
   }
}

