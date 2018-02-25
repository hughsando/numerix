import numerix.*;
import nme.display.BitmapData;

import cpp.vm.Thread;

class Detector
{
   public var error:String;
   var owner:Main;
   var mainThread:Thread;
   var workThread:Thread;
   var model:Model;

   public function new(inOwner:Main, modelname:String)
   {
      owner = inOwner;
      mainThread = Thread.current();
      workThread = Thread.create( function() threadLoop(modelname) );
   }

   public function run( job:Void->Void )
   {
      workThread.sendMessage(job);
   }

   public function postMain( result:Void->Void )
   {
      mainThread.sendMessage(result);
   }

   function getImage(result:Tensor) : BitmapData
   {
      if (result.channels==1)
      {
         var shape = result.shape;
         var h = shape[0];
         var w = shape[1];
         var ints = new Array<Int>();
         for(idx in 0...w*h)
            ints[idx] = result[idx]>128 ? 0xffffffff : 0;
         var bmp = new nme.display.BitmapData(w,h,false);
         bmp.setVector(new nme.geom.Rectangle(0,0,w,h),ints);
         return bmp;
      }
      return null;
   }

   public function runModelAsync(src:Tensor, ?onImage:BitmapData->Float->Void, ?onBoxes:Array<Box>->Float->Void )
   {
      run( function() {
         try
         {
            var t0 = haxe.Timer.stamp();
            var result = model.run(src);
            var t = haxe.Timer.stamp() - t0;
            if (onBoxes!=null)
            {
               var boxes = model.outputLayer.getBoxes();
               postMain( function() onBoxes(boxes,t) );
            }
            if (onImage!=null)
            {
               var bmp = getImage(result);
               postMain( function() onImage(bmp,t) );
            }
         }
         catch(e:Dynamic)
         {
            trace("Error " + haxe.CallStack.exceptionStack().join("\n") );
            error = e;
            if (onImage!=null)
               postMain( function() onImage(null,0) );
         }
     } );
   }

   public function runModelSync(src:Tensor, ?onImage:BitmapData->Float->Void, ?onBoxes:Array<Box>->Float->Void )
   {
      if (error!=null && onImage!=null)
         onImage(null,0);

      try
      {
         if (model==null)
         {
            //Sys.println("waiting for model...");
            if (onImage!=null)
               onImage(null,0);
            return;
         }
         var t0 = haxe.Timer.stamp();
         var result = model.run(src);
         var t = haxe.Timer.stamp() - t0;
         if (onBoxes!=null)
         {
            var boxes = model.outputLayer.getBoxes();
            onBoxes(boxes,t);
         }
         if (onImage!=null)
         {
            onImage(getImage(result),t);
         }

      }
      catch(e:Dynamic)
      {
         trace("Error " + haxe.CallStack.exceptionStack().join("\n") );
         error = e;
         if (onImage!=null)
            onImage(null,0);
      }
   }


   function threadLoop(modelname)
   {
      try
      {
         model = numerix.Model.load(modelname);
         while(true)
         {
            var job = Thread.readMessage(true);
            if (job==null)
               break;

            job();
         }
      }
      catch(e:Dynamic)
      {
         error = e;
         trace("Error in model " + e);
      }
   }
}
