package numerix;

class Io
{
   static public inline var version = 1;

   public static function encode(value:Dynamic) : haxe.io.Bytes
   {
      var output = new haxe.io.BytesOutput();
      output.writeString("nxio");
      output.writeInt32(version);

      var extra = new Array<Tensor>();
      var stringData = haxe.Json.stringify( value,
         function(_, value) return replace(value,extra), " " );

      output.writeInt32(stringData.length);
      output.writeString(stringData);
      output.writeInt32(0);
      output.writeString("nxxd");
      output.writeInt32(extra.length);
      for(tensor in extra)
      {
         var flags = 0;
         output.writeInt32(flags);
         output.writeInt32(tensor.dataSize);
      }
      for(tensor in extra)
      {
         var buffer = tensor.getBytes();
         output.writeBytes(buffer, 0, buffer.length);
      }

      return output.getBytes();
   }

   static function replace(value:Dynamic, extra:Array<Tensor>)
   {
      if (value._hxcpp_kind == "Tensor")
      {
         var tensor = Tensor.fromHandle(value);
         var id = extra.indexOf(tensor);
         if (id<0)
         {
            id = extra.length;
            extra.push(tensor);
         }

         var result = {
            _tensor_:true,
            dataId:id,
            shape:tensor.shape,
            type:tensor.typename,
         };
         return result;
      }
      return value;
   }
}


