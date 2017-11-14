package numerix;

class DataType
{
   public static inline var SignedInteger   = 0x01000000;
   public static inline var UnsignedInteger = 0x02000000;
   public static inline var Floating        = 0x04000000;
   public static inline var AsciiString     = 0x08000000;
   public static inline var Boolean         = 0x10000000;

   public static inline var BitsMask        = 0x000000ff;

   public static inline var Float32        = Floating | 32;
   public static inline var Float64        = Floating | 64;
   public static inline var UInt8          = UnsignedInteger | 8;
   public static inline var UInt16         = UnsignedInteger | 16;
   public static inline var UInt32         = UnsignedInteger | 32;
   public static inline var UInt64         = UnsignedInteger | 64;
   public static inline var Int8           = SignedInteger | 8;
   public static inline var Int16          = SignedInteger | 16;
   public static inline var Int32          = SignedInteger | 32;
   public static inline var Int64          = SignedInteger | 64;


   public static function string(type:Int):String
   {
      if ( (type & BitsMask) == BitsMask)
         return "Void";

      if ( (type & SignedInteger)>0)
         return "Int" + (type & BitsMask);
      if ( (type & UnsignedInteger)>0)
         return "UInt" + (type & BitsMask);
      if ( (type & Floating)>0)
         return "Float" + (type & BitsMask);
      if ( (type & AsciiString)>0)
         return "AsciiString";
      if ( (type & Boolean)>0)
         return "Boolean";

      return "Bad Type";
   }

   public static function fromString(type:String):Int
   {
      return switch(type)
      {
         case "Float32" : Float32;
         case "Float64" : Float64;
         case "UInt8" : UInt8;
         case "UInt16" : UInt16;
         case "UInt32" : UInt32;
         case "UInt64" : UInt64;
         case "Int8" : Int8;
         case "Int16" : Int16;
         case "Int32" : Int32;
         case "Int64" : Int64;
         case "Boolean" : Boolean;
         case "AsciiString" : AsciiString;
         default:
             throw "Unknown type " + type;
             0;
      }
   }

   public static function size(type:Int):Int
   {
      if ( (type & (SignedInteger|UnsignedInteger|Floating)) > 0 )
         return (type & BitsMask) >> 3;

      return -1;
   }

}


