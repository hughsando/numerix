package numerix.keras;
import hdf5.*;

class Model
{
   var className:String;
   var input:Layer;
   var output:Layer;
   var layers:Array<Layer>;

   public function new(filename:String)
   {
      var file = hdf5.File.open(filename);
      var attribs = haxe.Json.parse( file.attribs.model_config );
      className = attribs.class_name;
      if (className!="Sequential")
         throw "Only sequential models supported";

      //Sys.println("Data:\n" + file.getItemsRecurse()
      //        .filter( i -> i.type==DatasetItem ).map( i -> return i.path ).join("\n"));

      layers = [];
      var prev:Layer = null;
      for(l in (attribs.config:Array<Dynamic>))
      {
         var layer = createLayer(file,l,prev);
         if (input==null)
            input = layer;
         prev = layer;
         layers.push(layer);
      }
      output = prev;
      file.close();

      trace(layers);
   }

   function createLayer(file:Group, config:Dynamic, prev:Layer) : Layer
   {
      var name = config.class_name;
      switch(name)
      {
         case "Conv2D" : return new Conv2D(file, config.config, prev);
         default:
            throw 'Unknown layer type $name';
      }
      return null;
   }
}
