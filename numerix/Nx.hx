package numerix;

class Nx
{
   public static inline var float32 = DataType.Float32;
   public static inline var float64 = DataType.Float64;
   public static inline var int8 = DataType.Int8;
   public static inline var int16 = DataType.Int16;
   public static inline var int32 = DataType.Int32;
   public static inline var int64 = DataType.Int64;
   public static inline var uint8 = DataType.UInt8;
   public static inline var uint16 = DataType.UInt16;
   public static inline var uint32 = DataType.UInt32;
   public static inline var uint64 = DataType.UInt64;


   public static function array(arrayLike:Dynamic, dataType=-1, ?inShape:Array<Int>)
   {
      return new Tensor(arrayLike, dataType, inShape);
   }

   public static function tensor(arrayLike:Dynamic, dataType=-1, ?inShape:Array<Int>) : Tensor
   {
      return new Tensor(arrayLike, dataType, inShape);
   }
}


