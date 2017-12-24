package numerix;

class Movidius extends Layer
{
   static var id = 0;
   public var outputWidth(default,null):Int;
   public var outputHeight(default,null):Int;
   public var outputChannels(default,null):Int;

   var graphName:String;

   public function new(config:Dynamic, input:Layer, inBaseDir="./")
   {
      super(config,input);

      outputWidth = config.width;
      outputHeight = config.height;
      outputChannels = config.channels;

      if (outputWidth==0 || outputHeight==0 || outputChannels==0)
         throw "Movidius - invalid shape $outputWidth, $outputHeight, $outputChannels";

      if (name==null)
         name = "movidius_" + (id++);

      var index:Int = 0;
      if (config.index!=null)
         index = config.index;

      var device = moviGetDeviceName(index);

      graphName = inBaseDir + "/" + config.graph;
      var data = sys.io.File.getBytes(graphName);

      handle = layCreateMovidius(device, data, [outputHeight,outputWidth,outputChannels] );
   }

   public static function getDeviceName(index:Int = 0)
   {
      return moviGetDeviceName(index);
   }


   override public function toString() return 'Movidius($graphName)';


   static var layCreateMovidius = Loader.load("layCreateMovidius","sooo");
   static var moviGetDeviceName = Loader.load("moviGetDeviceName","is");


}





