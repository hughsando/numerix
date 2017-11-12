package numerix;

class Hdf5Tools
{
   public static function read(group:hdf5.Group, inPath:String) : Tensor
   {
      var dataset = group.openDataset(inPath);
      var tensor = new Tensor(null, dataset.type, dataset.shape);
      dataset.fillData(tensor.data, tensor.type);
      dataset.close();
      return tensor;
   }
}

