import nme.media.Camera;
import nme.display.*;
import nme.events.*;
import nme.text.*;
import numerix.*;
import nme.bare.Surface;
import nme.image.PixelFormat;
import nme.utils.ByteArray;
import cpp.vm.Thread;
using StringTools;
import Sys.println;

class Main extends Sprite
{
   var camera:Camera;
   var bitmap:Bitmap;
   var detector:Detector;
   var detectorBusy:Bool;
   var bmpTensor:Tensor;
   var overlay:Sprite;
   var timeField:TextField;
   var labelFormat:TextFormat;
   var colours:Array<Int>;
   var resultBmp:Bitmap;
   var mirror:Bool;
   var sync:Bool;
   var cpu:Bool;

   public function new()
   {
      var modelName = "";

      var args = Sys.args();
      mirror = args.remove("-mirror") || args.remove("-m");
      sync = args.remove("-sync");
      cpu = args.remove("-cpu");
      if (cpu)
         numerix.Model.enableGpu(false);

      for(a in args)
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
            numerix.Model.enableGpu(false);
         }


      if (args.length>0)
         modelName = args[0];
      else if (Sys.getEnv("MODEL")!=null)
         modelName = Sys.getEnv("MODEL");
      else
      {
         var msg = "Please set the model name with '... -args name' or 'setenv MODEL=name'";
         Sys.println("----");
         Sys.println(msg);
         Sys.println("----");
         Sys.exit(-1);
      }



      super();
      camera = Camera.getCamera();
      if (camera!=null)
         camera.addEventListener(Event.VIDEO_FRAME,onFrame);
      stage.addEventListener(Event.RESIZE, function(_) setBmpSize() );
      stage.addEventListener(Event.ENTER_FRAME, function(_) checkMessages() );

      labelFormat = new TextFormat();
      labelFormat.color = 0xff00ff;
      labelFormat.size = 14;
      labelFormat.bold = true;
      labelFormat.font = "_sans";

      detector = new Detector(this,modelName);

      var r = 91131;
      colours = [for(i in 0...80) { r = r*78123 + 123; r & 0xffffff; }  ];

      detectorBusy = true;
      detector.run( function() detectorBusy = false );
   }

   function checkMessages()
   {
      while(true)
      {
         var job = Thread.readMessage(false);
         if (job!=null)
            job();
         else
            break;
      }
   }

   function setBmpSize()
   {
      if (bitmap!=null)
      {
         var sw:Float = stage.stageWidth;
         var sh:Float = stage.stageHeight;
         var w = bitmap.bitmapData.width;
         var h = bitmap.bitmapData.height;
         if (w*sh > h*sw)
         {
            bitmap.width = sw;
            sh = h*sw/w;
            bitmap.height = sh;
            bitmap.x = 0;
            bitmap.y = (stage.stageHeight-sh)*0.5;
         }
         else
         {
            bitmap.height = sh;
            sw = w*sh/h;
            bitmap.width = sw;
            bitmap.y = 0;
            bitmap.x = (stage.stageWidth-sw)*0.5;
         }


         resultBmp.width = w/3;
         resultBmp.height = h/3;

         overlay.x = bitmap.x;
         overlay.y = bitmap.y;
         if (mirror)
         {
            bitmap.x += bitmap.width;
            bitmap.width = -bitmap.width;
         }
      }
   }

   function createLabel(text:String,colour:Int)
   {
      var field = new TextField();
      field.autoSize = TextFieldAutoSize.LEFT;
      labelFormat.color = colour;
      field.defaultTextFormat = labelFormat;
      field.text = text;
      return field;
   }

   public function onImage(bmp:BitmapData, time:Float)
   {
      detectorBusy = false;
      timeField.text = (Std.int(time*100000) * 0.01 ) + "ms";
      if (bmp!=null)
         resultBmp.bitmapData = bmp;
   }

   public function onBoxes(boxes:Array<Box>, time:Float)
   {
      if (boxes!=null)
      {
         detectorBusy = false;

         while(overlay.numChildren>0)
            overlay.removeChildAt( overlay.numChildren-1 );

         var gfx = overlay.graphics;
         gfx.clear();
         var sx = Math.abs(bitmap.width);
         var sy = bitmap.height;
         for(box in boxes)
         {
            if (mirror)
               box.x = 1 - box.x;

            var col =  colours[ box.classId ];
            if (box.className!=null)
            {
               var label = createLabel(box.className, col);
               overlay.addChild(label);
               label.x = (box.x -box.w*0.5) * sx;
               label.y = (box.y -box.h*0.5) * sy;
            }
            gfx.lineStyle(2, col );
            //trace(box.className);
            gfx.drawRect((box.x-box.w*0.5)*sx, (box.y-box.h*0.5)*sy, box.w*sx, box.h*sy);
         }
         timeField.text = (Std.int(time*100000) * 0.01 ) + "ms";
      }
      else
      {
         trace("Error : " + detector.error );
      }
   }

   public function onFrame(_)
   {
      if (camera!=null)
      {
         if (bitmap==null)
         {
            bitmap = new Bitmap( camera.bitmapData );
            bitmap.smoothing = true;
            addChild(bitmap);
            overlay = new Sprite();
            addChild(overlay);

            resultBmp = new Bitmap();
            addChild(resultBmp);
            resultBmp.bitmapData = new BitmapData(100,60,false,0xffff0000);

            timeField = new TextField();
            timeField.autoSize = TextFieldAutoSize.LEFT;
            var fmt = new TextFormat();
            fmt.size = 32;
            fmt.color = 0x00ffff;
            timeField.defaultTextFormat = fmt;
            addChild(timeField);
            setBmpSize();
         }
         if (!detectorBusy)
         {
            if (bmpTensor==null)
            {
               var w = bitmap.bitmapData.width;
               var h = bitmap.bitmapData.height;
               bmpTensor = Tensor.empty(Nx.float32, [h,w,3] );
             }

            //var trans = Surface.FLOAT_UNIT_SCALE;
            var trans = Surface.FLOAT_UNSCALED;
            camera.bitmapData.getFloats32( bmpTensor.data, 0, 0, PixelFormat.pfRGB, trans );

            var pixels = bmpTensor.width * bmpTensor.height;
            var idx = 0;
            for(p in 0...pixels)
            {
               bmpTensor[idx] = bmpTensor[idx]-106; idx++;
               bmpTensor[idx] = bmpTensor[idx]-114; idx++;
               bmpTensor[idx] = bmpTensor[idx]-129; idx++;
            }


            detectorBusy = true;
            if (sync)
               detector.runModelSync(bmpTensor, onImage /*onBoxes*/ );
            else
               detector.runModelAsync(bmpTensor, onImage /*onBoxes*/ );
         }
      }
   }

}



