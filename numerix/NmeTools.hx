package numerix;
import nme.display.BitmapData;
import nme.bare.Surface;
import nme.image.PixelFormat;
import nme.utils.ByteArray;

class NmeTools
{
   public static inline var TRANS_UNSCALED     = Surface.FLOAT_UNSCALED;
   public static inline var TRANS_ZERO_MEAN    = Surface.FLOAT_ZERO_MEAN;
   public static inline var TRANS_128_MEAN     = Surface.FLOAT_128_MEAN;
   public static inline var TRANS_UNIT_SCALE   = Surface.FLOAT_UNIT_SCALE;
   public static inline var TRANS_STD_SCALE    = Surface.FLOAT_STD_SCALE;
   public static inline var TRANS_SWIZZLE_RGB  = Surface.FLOAT_SWIZZLE_RGB;
   public static inline var TRANS_STD          = Surface.FLOAT_STD_SCALE | Surface.FLOAT_ZERO_MEAN;

   public static function loadImageF32(filename:String, transform=0):Tensor
   {
      var bmp = BitmapData.loadFromBytes( nme.utils.ByteArray.readFile(filename) );
      var w = bmp.width;
      var h = bmp.height;
      var result = Tensor.empty(Nx.float32, [h,w,3] );
      bmp.getFloats32( result.data, 0, 0, PixelFormat.pfRGB, transform );
      bmp.dispose();
      return result;
   }

   public static function saveImageF32(tensor:Tensor, filename:String, transform=TRANS_UNIT_SCALE):Void
   {
      var shape = tensor.shape;
      var channels = 0;
      if (shape.length==2)
         channels = 1;
      else if (shape.length==3)
         channels = shape[2];
      if (channels!=1 && channels!=3)
          throw 'Could not write $channels channels';

      var h = shape[0];
      var w = shape[1];
      var bmp = channels==1 ? BitmapData.createGrey(w,h) : new BitmapData(w,h,false,0);
      bmp.setFloats32(tensor.data, 0, 0,channels==1 ? PixelFormat.pfLuma : PixelFormat.pfRGB,transform);
      var dot = filename.lastIndexOf('.');
      var data = bmp.encode( dot<0 ? "png" : filename.substr(dot+1) );
      data.writeFile(filename);
   }

}

