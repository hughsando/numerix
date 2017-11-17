package numerix;

abstract Tensor(Dynamic)
{
   public var shape(get,never):Array<Int>;
   public var elementCount(get,never):Int;
   public var elementSize(get,never):Int;
   public var dataSize(get,never):Int;
   public var type(get,never):Int;
   public var typename(get,never):String;
   public var data(get,never):Dynamic;
   public var min(get,never):Float;
   public var max(get,never):Float;

   function new(inHandle:Dynamic)
   {
      this = inHandle;
      if (this!=null)
         this._hxcpp_toString = handleToString;
   }

   public static function create(inArrayLike:Dynamic,inStoreType = -1, ?inShape:Array<Int>)
   {
      return new Tensor( tdFromDynamic(inArrayLike, inStoreType, inShape) );
   }

   public static function empty(inStoreType, inShape:Array<Int>)
   {
      return create(null, inStoreType, inShape);
   }

   public static function fromBytes(inBytes:haxe.io.Bytes,inStoreType:Int, inShape:Array<Int>)
   {
      return create(inBytes, inStoreType, inShape);
   }


   public static function fromHandle(inHandle:Dynamic) return new Tensor(inHandle);

   public function release()
   {
      tdRelease(this);
   }

   public function print(maxElems = 12)
   {
      tdPrint(this, maxElems);
   }

   /*
   public function getMin(axis:Int) : Tensor
   {
      var result:Dynamic = tdGetMinAxis(this,axis);
      return fromHandle(result);
   }
   */

   function get_min() : Float
   {
      var result:Dynamic = tdGetMin(this);
      return result;
   }

   /*
   public function getMax(axis:Int) : Tensor
   {
      var result:Dynamic = tdGetMaxAxis(this,axis);
      return fromHandle(result);
   }
   */

   function get_max() : Float
   {
      var result:Dynamic = tdGetMax(this);
      return result;
   }



   public function reorder(order:Array<Int>, inRelease = false):Tensor
   {
      var result = new Tensor(tdReorder(this, order));
      if (inRelease)
         release();
      return result;
   }

   public function setFlat():Void
   {
      tdSetFlat(this);
   }

   public function setShape(inShape:Array<Int>):Void
   {
      tdSetShape(this,inShape);
   }


   public function getBytes():haxe.io.Bytes
   {
      var size = dataSize;
      if (size<1)
         return haxe.io.Bytes.alloc(0);

      var result = haxe.io.Bytes.alloc(size);

      tdFillData(this,result);

      return result;
   }

   public function get_shape() : Array<Int>
   {
      var shape = new Array<Int>();
      for(i in 0...tdGetDimCount(this))
         shape.push(tdGetDimAt(this,i));
      return shape;
   }

   public function get_elementCount() : Int
   {
      return tdGetElementCount(this);
   }

   public function get_elementSize() : Int
   {
      return DataType.size( tdGetType(this) );
   }

   public function get_dataSize() : Int
   {
      return tdGetElementCount(this) * DataType.size( tdGetType(this) );
   }


   public function get_typename() : String
   {
      return DataType.string( tdGetType(this) );
   }

   public function get_type() : Int
   {
      return tdGetType(this);
   }

   public function toString() : String
   {
      return handleToString(this);
   }

   public function get_data() : Dynamic
   {
      return tdGetData(this);
   }



   static function handleToString(handle:Dynamic)
   {
      if (handle==null)
         return "Tensor(null)";
      var shape = new Array<Int>();
      for(i in 0...tdGetDimCount(handle))
         shape.push(tdGetDimAt(handle,i));
      return "Tensor(" + shape + " x " + DataType.string(tdGetType(handle)) + ")";
   }

   static var tdFromDynamic = Loader.load("tdFromDynamic","oioo");
   static var tdGetDimCount = Loader.load("tdGetDimCount","oi");
   static var tdGetDimAt = Loader.load("tdGetDimAt","oii");
   static var tdGetElementCount = Loader.load("tdGetElementCount","oi");
   static var tdGetType = Loader.load("tdGetType","oi");
   static var tdGetData = Loader.load("tdGetData","oo");
   static var tdFillData = Loader.load("tdFillData","oov");
   static var tdRelease = Loader.load("tdRelease","ov");
   static var tdPrint = Loader.load("tdPrint","oiv");
   static var tdReorder = Loader.load("tdReorder","ooo");
   static var tdGetMin = Loader.load("tdGetMin","od");
   static var tdGetMax = Loader.load("tdGetMax","od");
   static var tdSetFlat = Loader.load("tdSetFlat","ov");
   static var tdSetShape = Loader.load("tdSetShape","oov");
   //static var tdGetMinAxis = Loader.load("tdGetMin","ooo");
   //static var tdGetMaxAxis = Loader.load("tdGetMax","oio");

}

