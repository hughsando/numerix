package numerix;

import haxe.io.Input;
import haxe.io.Bytes;
using Reflect;

class Io
{
   static public inline var version = 1;

   public static function decode(input:Input) : Dynamic
   {
      var magic = input.readString(4);
      input.bigEndian = false;
      if (magic!="nxio")
         throw "Bad magic";
      var ver = input.readInt32();
      var strLen = input.readInt32();
      var str = input.readString(strLen);
      var zero = input.readInt32();
      if (input.readString(4)!="nxxd")
         throw "Bad data magic";
      var count = input.readInt32();
      var extras = new Array<Bytes>();
      for(i in 0...count)
      {
         var flags = input.readInt32();
         var blen = input.readInt32();
         var bytes = Bytes.alloc(blen);
         input.readBytes(bytes, 0, blen);
         extras.push(bytes);
      }

      var json = haxe.Json.parse(str);
      json = remapTensors(json, extras);
      return json;
   }

   static function remapTensors(value:Dynamic, extras:Array<Bytes>)
   {
      if ( value.isObject())
      {
         if (value._tensor_ && value.hasField("dataId"))
         {
            var typename:String = value.type;
            var dataId:Int = value.dataId;
            var shape:Array<Int> = value.shape;
            return Tensor.fromBytes( extras[dataId], DataType.fromString(typename), shape);
         }

         if (Std.is(value,Array))
         {
            var len = value.length;
            for(i in 0...len)
               value[i] = remapTensors(value[i], extras);
         }
         else
            for(field in value.fields())
               value.setField(field, remapTensors( value.field(field), extras)  );
      }

      return value;
   }

   public static function decodeBytes(input:Bytes) : Dynamic
   {
      return decode(new haxe.io.BytesInput(input) );
   }

   public static function readFile(filename:String) : Dynamic
   {
      var file = sys.io.File.read(filename);
      var result = decode(file);
      file.close();
      return result;
   }


   public static function encode(value:Dynamic) : Bytes
   {
      var output = new haxe.io.BytesOutput();
      output.bigEndian = false;
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

   public static function writeFile(filename:String, data:Dynamic)
   {
      sys.io.File.saveBytes(filename, encode(data));
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


