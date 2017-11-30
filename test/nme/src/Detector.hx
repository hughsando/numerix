import numerix.*;

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

   public function runModelAsync(src:Tensor, onComplete:Array<Box>->Float->Void )
   {
      run( function() {
         try
         {
            var t0 = haxe.Timer.stamp();
            model.run(src);
            var t = haxe.Timer.stamp() - t0;
            var boxes = model.outputLayer.getBoxes();
            postMain( function() onComplete(boxes,t) );
         }
         catch(e:Dynamic)
         {
            trace("Error " + haxe.CallStack.exceptionStack().join("\n") );
            error = e;
            postMain( function() onComplete(null,0) );
         }
     } );
   }

   function threadLoop(modelname)
   {
      try
      {
         model = numerix.Model.load(modelname);
         trace("Loaded");
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
