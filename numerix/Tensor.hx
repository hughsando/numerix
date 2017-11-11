package numerix;

abstract Tensor(Dynamic)
{
   public var shape(get,never):Array<Int>;
   public var elementCount(get,never):Int;
   public var elementSize(get,never):Int;
   public var type(get,never):Int;
   public var typename(get,never):String;
   //public var data(get,null):cpp.Pointer<cpp.Void>;

   public function new(inArrayLike:Dynamic,inStoreType = -1, ?inShape:Array<Int>)
   {
      this = tdFromDynamic(inArrayLike, inStoreType, inShape);
      if (this!=null)
         this._hxcpp_toString = handleToString;
   }

   inline public function release()
   {
      tdRelease(this);
      this = null;
   }

   public function print(maxElems = 12)
   {
      tdPrint(this, maxElems);
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



   static function handleToString(handle:Dynamic)
   {
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
   static var tdRelease = Loader.load("tdRelease","ov");
   static var tdPrint = Loader.load("tdPrint","oiv");

}

