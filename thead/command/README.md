<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

## HHB Command line tools

### 1. Introduction
HHB is a set of function related command line tools, its usage is similar to gcc.

The commands that HHB supports are shown as following table.

| args     | notes|
| :------  | :------|
|-c|Generate code params and json file until codegen. Corresponding to gcc -c  |
|-S|Generate tvm ir until optimize. Corresponding to gcc -S.   |
|-E|Generate tvm ir until import. Corresponding to gcc -E.   |
|-v|Output the full command line to stderr.   |
|--version|Print version number.   |
|-q|Generate tvm ir until quantize. |
|-O0/O1/O2/O3|Set opt_level while build relay ir. opt_level=0/1/2/3.   |
|-p|Generate profile which will predict performance. |
|--deploy| Generate binary codes which will executed on the specific target. |
|--simulate|Replace the specified target with X86 llvm, and dump the reference results of each layer, which is equivalent to the behavior simulator.|
|||
|--save-temps|Save temp files if '--save-temps' is set|
|-o|Place output into specific file.|
|--no-quantize|If set, don't quantize the model.|
|--tf-pb|The path of Tensorflow model file(.pb).|
|--caffe-proto|The path of Caffe model file(.proto file).|
|--caffe-blobs|The path of Caffe model file(.caffemodel).|
|--input-name|Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple valuesare separated by semicolon(;).|
|--input-shape|Set the shape of input nodes. Multiple shapes areseparated by semicolon(;) and the dims between shape areseparated by space.|
|--output-name|Set the name of output nodes. Multiple shapes areseparated by semicolon(;).|
|--calibrate-mode|Set calibrate mode when quantizing model(global_scaleor kl_divergence).|
|--quantized-type|Select the algorithm of quantization (asym or sym).|
|--weight-scale|Set the mode of weight scale (power2 or max).|
|--dataset|Provide with dataset for for calibration in quantization step.Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt.|
|--input-dataset|Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt.|
|--target|Set target device which codes generated will run. The defualt value is 'llvm'|
|--main-gen|Generate main c++ code according to template.|
|--input-normal|divide number for inputs normalization.|
|--input-mean|Set the mean value of input while preprocess the input data.|

### 2. How to use

#### 2.1 import model

##### 2.1.1 import tensorflow
```Python
python hhb.py -E --tf-pb alexnet.pb --input-name Placeholder --input-shape '1 227 227 3' --output-name Softmax
```
##### 2.1.2 import caffe
```Python
python hhb.py -E --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel
```

#### 2.2 quantize model
##### 2.2.1 only '-q'
```Python
python hhb.py -q --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --dataset dataset.txt
```

#### 2.3 optimize model
##### 2.3.1 with quantization
```Python
python hhb.py -S --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --dataset 0.jpg
```

##### 2.3.2 without quantization
```Python
python hhb.py -S --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --no-quantize
```

#### 2.4 generate profile

#### 2.5 codegen
##### 2.5.1 with quantization
```Python
python hhb.py -c --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --dataset 0.jpg
```

##### 2.5.2 without quantization
```Python
python hhb.py -c --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --no-quantize
```

#### 2.6 deploy model
```Python
python hhb.py --deploy --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --no-quantize
```

#### 2.7 simulate
##### 2.7.1 with quantization
```Python
python hhb.py --simulate --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --dataset 0.jpg
```

##### 2.7.2 without quantization
```Python
python hhb.py --simulate --caffe-proto lenet.prototxt --caffe-blobs lenet.caffemodel --no-quantize --dataset 0.jpg
```

##### 2.8 Preprocess the input data
```Python
python hhb.py --simulate --caffe-proto mobilenetv1.prototxt --caffe-blobs mobilenetv1.caffemodel --dataset n01440764_188.JPEG --calibrate-mode global_scale --input-normal 127.5 --input-mean '104.0 117.0 124.0'
```

##### 2.9 Specify the input of model
```Python
python hhb.py --simulate --caffe-proto mobilenetv1.prototxt --caffe-blobs mobilenetv1.caffemodel --dataset n01440764_188.JPEG --calibrate-mode global_scale --input-normal 127.5 --input-mean '104.0 117.0 124.0' --input-dataset n01491361_405.JPEG
```

#### 2.10 Makefile
Execute hhb.py with '--deploy'
```bash
make
if specifying --dataset while executing hhb.py, then
./lib/c_runtime data.tenosr
else:
./lib/c_runtime (the input of model is generated randomly.)
```

#### 2.11 images preprocess api

```Python
sys.path.append("${tvm_path}/thead/command/")
from model_evaluation import ModelZoom

model = ModelZoom(model_name='inceptionv1', tf_pb=pb_path, target_framework='tensorflow')
img_gen = model.image_dataset_generator(img_path)
img = next(img_gen)
```
- tf_pb: the path of tensorflow model
- target_framework: the platform that you will run the model.
                    support for tensorflow caffe and hhb
