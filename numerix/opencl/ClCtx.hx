package numerix.opencl;

class ClCtx
{
   public static var platforms(get,null):Array<Platform>;

   var handle:Dynamic;
   var devices:Array<Dynamic>;

   public function new(platform:Platform, ?inDevices:Array<Device>)
   {
      devices = new Array<Dynamic>();
      if (inDevices==null)
         inDevices = [ platform.devices[0] ];
      for(d in inDevices)
         devices.push( d.id );

      handle = oclCreateContext(platform.id, devices);
   }

   public static function setCurrent(inCtx:ClCtx)
   {
      oclSetCurrent(inCtx.handle);
   }


   public function toString():String return handle;


   public static function get_platforms()
   {
      var platforms:Array<Dynamic> = oclGetPlatforms();
      for(platform in platforms)
         platform.extensions = platform.extensions.split(" ");
      return cast platforms;
   }

   static var oclGetPlatforms = Loader.load("oclGetPlatforms","o");
   static var oclCreateContext = Loader.load("oclCreateContext","ooo");
   static var oclSetCurrent = Loader.load("oclSetCurrent","ov");
}
