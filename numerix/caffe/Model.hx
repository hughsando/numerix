package numerix.caffe;

import Sys.println;
import haxe.io.Bytes;
import numerix.Tensor;
import numerix.Nx;
import cpp.NativeMath.idiv;
using StringTools;



class Model extends numerix.Model
{
   var transpose:Bool;
   var reorgLayer:Reorg = null;
   var filename:String;
   var weightName:String;

   public function new(inFilename:String)
   {
      super();

      var binName = "";
      var txtName = "";

      if (inFilename.endsWith(".prototxt"))
      {
         txtName = inFilename;
         binName = inFilename.substr(0, inFilename.length - ".prototxt".length) + ".caffemodel";
      }
      else
      {
         binName = inFilename;
      }


      var layers:Array<Dynamic> = caffeLoad(txtName, binName);

      // BOTTOM = input layer name(s)
      // TOP = set output name

      var layerMap = new Map<String, Layer>();


      for(layer in layers)
      {
         var top:Array<String> = layer.top;
         if (top!=null && top.length==0)
            throw "caffe layer without top:" + top;
         var outName = top[0];


         var inputLayers:Array<Layer> = layer.bottom==null ? [] :
              layer.bottom.map( layerMap.get );

         var lastLayer:Layer = null;
         switch(layer.type)
         {
            case "Input":
               var lay = new InputLayer();
               if (inputLayer==null)
               {
                  inputLayer = lay;
                  var shape:Array<Int> = layer.input_dim;
                  if (shape!=null)
                  {
                      switch(shape.length)
                      {
                         case 4:
                            height = shape[2];
                            width = shape[3];
                         case 3:
                            height = shape[1];
                            width = shape[2];
                         default:
                            throw "Could not use input shape " + shape;
                      }
                  }
               }
               layerMap.set(outName,lastLayer = lay);

            case "Convolution":
               var cfg:Dynamic = { };

               var padSize:Int = 0;
               var pad:Array<Int> = layer.pad;
               if (pad!=null)
               {
                  if (pad.length!=0)
                  {
                     if (pad.length!=1 && pad[0]!=pad[1])
                        throw "Convolution.pad - expected padding to be the same";
                     padSize = pad[0];
                  }
               }
               else if (layer.pad_w !=null)
               {
                  if (layer.pad_w!=layer.pad_h)
                     throw "Convolution - expected pad_w = pad_h";
                  padSize = layer.pad_w;
               }

               cfg.outputs = layer.filters;
               cfg.dilation = layer.dilation;

               var strides = layer.stride;
               if (layer.stride_w!=null && layer.stride_h!=null)
                   strides = [ Std.int(layer.stride_w), Std.int(layer.stride_h) ];
               cfg.strides = strides;

               var size:Array<Int> = layer.size;
               if (layer.kernel_w!=null && layer.kernel_h!=null)
                   size = [ Std.int(layer.kernel_w), Std.int(layer.kernel_h) ];
               cfg.kernelSize = size;

               if (padSize==0)
                  cfg.padding='valid';
               else if (padSize== Std.int( (size[0]-1)/2 ))
                  cfg.padding='same';
               else
                  throw 'Convolution - could not calculate padding - $padSize / $size' + ((size[0]-1)/2);

               if (inputLayers.length!=1 || inputLayers[0]==null)
                  throw "Convolution - expected 1 input layer";

               var lay = new Conv2D(cfg, inputLayers[0]);
               layerMap.set(outName,lastLayer = lay);

               var weights:Array<Dynamic> = layer.weights;
               var tensors:Array<Tensor> = weights.map( Tensor.fromHandle );
               if (tensors[0]!=null)
                  tensors[0] = tensors[0].reorder([0,2,3,1]);
               lay.setWeights(tensors);

               //trace(tensors[0]);
               //tensors[0].print();
               //trace(tensors[1]);
               //tensors[1].print();


            case "ReLU":
               var conv:Conv2D = cast inputLayers[0];
               if (conv==null)
                  throw "ReLU must follow a Convolution layer";
               conv.setActivation(Layer.ACT_RELU);
               layerMap.set(outName,lastLayer = conv);


            case "Dropout":
               if (inputLayers.length!=1)
                  throw "Dropout - expected single input";
               layerMap.set(outName,lastLayer = inputLayers[0]);


            case "Split":
               if (inputLayers.length!=1)
                  throw "Split - expected single input";
               var lay = inputLayers[0];
               for(t in top)
                  layerMap.set(t,lay);
               lastLayer = lay;


            case "Pooling":
               if (layer.global_pooling==null)
               {
                  var cfg:Dynamic = { };

                  var padSize:Int = 0;
                  var pad:Null<Int> = layer.pad;
                  if (pad!=null)
                  {
                     padSize = pad;
                  }
                  else if (layer.pad_w !=null)
                  {
                     if (layer.pad_w!=layer.pad_h)
                        throw "Pooling - expected pad_w = pad_h";
                     padSize = layer.pad_w;
                     trace(padSize);
                  }

                  var strides = layer.stride;
                  if (layer.stride!=null)
                  {
                     var s:Int = layer.stride;
                     strides = [s,s];
                  }
                  else if (layer.stride_w!=null && layer.stride_h!=null)
                      strides = [ Std.int(layer.stride_w), Std.int(layer.stride_h) ];

                  if (strides==null)
                      throw "Pooling - could not find stride";
                  cfg.strides = strides;

                  var size:Array<Int> = layer.size;
                  if (layer.size!=null)
                  {
                     var ks:Int = layer.size;
                     size = [ks,ks];
                  }
                  else if (layer.kernel_w!=null && layer.kernel_h!=null)
                      size = [ Std.int(layer.kernel_w), Std.int(layer.kernel_h) ];
    
                  if (size==null)
                     throw "Pooling - could not find kernel size";
                  cfg.kernelSize = size;


                  if (padSize==0)
                     cfg.padding='valid';
                  else if (padSize== Std.int( (size[0]+strides[0]-1)/2 ))
                     cfg.padding='same';
                  else
                     throw 'Pooling - could not calculate padding - $padSize $strides / $size';

                  if (layer.method!=0)
                     println("todo - pool type : " + layer.method );

                  var lay = new MaxPool(cfg, inputLayers[0]);
                  layerMap.set(outName,lastLayer = lay);
               }
               else
               {
                  if (layer.method!=1)
                     println("todo - global pool type : " + layer.method );

                  var lay = new GlobalPool(layer, inputLayers[0]);
                  layerMap.set(outName,lastLayer = lay);
               }


            case "Concat":
               if (inputLayers.length!=2)
                  throw "Concat - expected 2 inputs";
               var lay = new Concat(layer, inputLayers[0],inputLayers[1]);
               layerMap.set(outName,lastLayer = lay);



            case "Softmax":
               if (inputLayers.length!=1)
                  throw "Softmax - expected single input";
               var lay = new Softmax(layer, inputLayers[0]);
               layerMap.set(outName,lastLayer = lay);


            default:
               throw "Unknown layer type " + layer.type;
         }

         if (lastLayer!=null && (layers.length==0 || layers[layers.length-1]!=layers) )
            addLayer(lastLayer);
      }
   }


   static var caffeLoad = Loader.load("caffeLoad","sso");
}



