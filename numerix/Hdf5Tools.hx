package numerix;

import hdf5.ItemType;

class Hdf5Tools
{
   public static function read(group:hdf5.Group, inPath:String) : Tensor
   {
      try {
         var dataset = group.openDataset(inPath);
         var tensor = Tensor.create(null, dataset.type, dataset.shape);
         dataset.fillData(tensor.data, tensor.type);
         dataset.close();
         return tensor;
      }
      catch(e:Dynamic)
      {
         trace('Could not open dataset in $inPath from: ' + listDatasets(group).join("\n"));
         throw(e);
      }
   }

   public static function listDatasets(group:hdf5.Group) : Array<String>
   {
      return group.getItemsRecurse().filter(function(i) return i.type==DatasetItem)
                .map(function(i) return i.path);
   }

}

