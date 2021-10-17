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
HHB is a set of function related command line tools, its usage is similar to gcc. All argumnets are listed as following:

| Arguments              | Note                                                         |
| ---------------------- | ------------------------------------------------------------ |
| optional arguments:    |                                                              |
| -v, --verbose          | Increase verbosity                                           |
| -h, --help             | Show this help information                                    |
| --version              | Print the version and exit                                   |
| -f, --model-file       | Path to the input model file, can pass multi files           |
| -o , --output          | The directory that holds the outputs.                        |
| -E                     | Convert model into relay ir.                                 |
| -Q                     | Quantize the relay ir.                                       |
| -C                     | Codegen the model.                                           |
| --simulate             | Simulate model on x86 device.                                |
| --generate-config      | Generate config file for HHB                                 |
| --no-quantize          | If set, don't quantize the model.                            |
| --save-temps           | Save temp files.                                             |
| -in, --input-name      | Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple values are separated by semicolon(;). |
| -is, --input-shape     | Set the shape of input nodes. Multiple shapes are separated by semicolon(;) and the dims between shape are separated by space. |
| -on, --output-name     | Set the name of output nodes. Multiple shapes are separated by semicolon(;). |
| --model-format         | Specify input model format: {keras,onnx,pb,tflite,pytorch,caffe} |
| --board                | Set target device, default is anole: {anole,c860,x86_ref}    |
| --opt-level            | Specify the optimization level, default is 3: {-1,0,1,2,3}   |
| -cd, --calibrate-dataset    | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| --num-bit-activation   | The bit number that quantizes activation layer, default is 32. {16,32} |
| --dtype-input          | The dtype of quantized input layer, default is uint. {int,uint} |
| --dtype-weight         | The dtype of quantized constant parameters, default is uint. {int,uint} |
| --calibrate-mode       | How to calibrate while doing quantization, default is maxmin. {maxmin,global_scale,kl_divergence,kl_divergence_tsing} |
| --quantized-type       | Select the algorithm of quantization, default is asym. {asym,sym} |
| --weight-scale         | Select the mothod that quantizes weight value, default is max. {max,power2} |
| --fuse-relu            | Fuse the convolutioon and relu layer.                        |
| --channel-quantization | Do quantizetion across channel.                              |
| --broadcast-quantization | Broadcast quantization parameters for special ops. |
| -m, --data-mean        | Set the mean value of input, multiple values are separated by space, default is 0. |
| -s, --data-scale       | Divide number for inputs normalization, default is 1.        |
| -r, --data-resize      | Resize base size for input image to resize.                  |
| --pixel-format         | The pixel format of input data, defalut is RGB:{RGB,BGR}   |
| --config-file          | Configue more complex parameters for executing the model.    |
| -sd, --simulate-dat    | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| --postprocess          | Set the mode of postprocess: 'top5' show top5 of output; 'save' save output to file.'save_and_top5' show top5 and save output to file. {top5,save,save_and_top5} |
| Subcommand             |                                                              |
| import                 | Import a model into relay ir                                 |
| quantize               | Quantize the imported model                                  |
| codegen                | Codegen the imported model                                   |
| simulate               | Simulate the imported model                                  |
| show                   | Show the imported model                                      |

#### 1.1 import command

| Arguments             | Note                                                         |
| --------------------- | ------------------------------------------------------------ |
| positional arguments: |                                                              |
| FILE                  | Path to the input model file, can pass multi files           |
| optional arguments:   |                                                              |
| -h, --help            | Show this helo infomation                                    |
| -in, --input-name          | Set the name of input node. If '--input-name'is None, default value is 'Placeholder'. Multiple values are separated by semicolon(;). |
| -is, --input-shape         | Set the shape of input nodes. Multiple shapes are separated by semicolon(;) and the dims between shape are separated by space. |
| -on, --output-name         | Set the name of output nodes. Multiple shapes are separated by semicolon(;). |
| --model-format        | Specify input model format: {keras,onnx,pb,tflite,pytorch,caffe} |
| --board               | Set target device, default is anole: {anole,c860,x86_ref}    |
| --opt-level           | Specify the optimization level, default is 3: {-1,0,1,2,3}   |
| --config-file         | Configue more complex parameters for executing the model.    |
| -o , --output         | The directory that holds the relay ir.                       |

#### 1.2 quantize command
| Arguments              | Note                                                         |
| ---------------------- | ------------------------------------------------------------ |
| positional arguments: |                                                              |
| FILE                  | Path to the input model file, can pass multi files           |
| optional arguments:   |                                                              |
| -h, --help            | Show this helo infomation                                    |
| -cd, --calibrate-dataset    | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| --num-bit-activation   | The bit number that quantizes activation layer, default is 32. {16,32} |
| --dtype-input          | The dtype of quantized input layer, default is uint. {int,uint} |
| --dtype-weight         | The dtype of quantized constant parameters, default is uint. {int,uint} |
| --calibrate-mode       | How to calibrate while doing quantization, default is maxmin. {maxmin,global_scale,kl_divergence,kl_divergence_tsing} |
| --quantized-type       | Select the algorithm of quantization, default is asym. {asym,sym} |
| --weight-scale         | Select the mothod that quantizes weight value, default is max. {max,power2} |
| --fuse-relu            | Fuse the convolutioon and relu layer.                        |
| --channel-quantization | Do quantizetion across channel.                              |
| --broadcast-quantization | Broadcast quantization parameters for special ops. |
| -m, --data-mean        | Set the mean value of input, multiple values are separated by space, default is 0. |
| -s, --data-scale      | Divide number for inputs normalization, default is 1.        |
| -r, --data-resize     | Resize base size for input image to resize.                  |
| --pixel-format        | The pixel format of input data, defalut is RGB:{RGB,BGR}            |
| --board               | Set target device, default is anole: {anole,c860,x86_ref}    |
| --opt-level           | Specify the optimization level, default is 3: {-1,0,1,2,3}   |
| -o , --output         | The directory that holds the quantized relay ir.              |
| --config-file         | Configue more complex parameters for executing the model.    |

#### 1.3 codegen command

| Arguments              | Note                                                         |
| ---------------------- | ------------------------------------------------------------ |
| positional arguments: |                                                              |
| FILE                  | Path to the input model file, can pass multi files           |
| optional arguments:   |                                                              |
| -h, --help            | Show this helo infomation                                    |
| --board               | Set target device, default is anole: {anole,c860,x86_ref}    |
| --opt-level           | Specify the optimization level, default is 3: {-1,0,1,2,3}   |
| --config-file         | Configue more complex parameters for executing the model.    |
| -o , --output         | The directory that holds the codegen files.                  |

#### 1.4 simulate command

| Arguments              | Note                                                         |
| ---------------------- | ------------------------------------------------------------ |
| positional arguments: |                                                              |
| FILE                  | Path to the input model file, can pass multi files           |
| optional arguments:   |                                                              |
| -h, --help            | Show this helo infomation                                    |
| -sd, --simulate-dat    | Provide with dataset for the input of model in reference step. Support dir or .npz .jpg .png .JPEG or .txt in which there are path of images. Note: only one image path in one line if .txt. |
| -m, --data-mean       | Set the mean value of input, multiple values are separated by space, default is 0. |
| -s, --data-scale      | Divide number for inputs normalization, default is 1.        |
| -r, --data-resize     | Resize base size for input image to resize.                  |
| --pixel-format        | The pixel format of input data, defalut is RGB:{RGB,BGR}            |
| --config-file         | Configue more complex parameters for executing the model.    |
| -o , --output         | Directory to the model file                                  |


### 2. How to use

HHB support two kind of modes: single-stage mode and multi-stages mode. And both modes can parser all command parameters from specify file.

#### 2.1 single-stage mode

For example, you can import, quantize and simulate the specified model by a single command as follows:

```Python
python hhb.py --simulate \
	-v -v -v \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 0.017 \
	--data-resize 256 \
	--calibrate-dataset quant.txt \
	--simulate-data n01440764_188.JPEG \
	--opt-level 3 \
	--board x86_ref \
	--model-file mobilenetv1.prototxt \
	mobilenetv1.caffemodel \
	--postprocess top5 \
```



#### 2.2 multi-stages mode

In this mode, you can compile model by executing multiply sub command.

##### 2.2.1 import model

```Python
python hhb.py import alexnet.prototxt alexnet.caffemodel -o model.relay --opt-level -1
```

##### 2.2.2 quantize model

```Python
python hhb.py quantize \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 1 \
	--data-resize 256 \
	--calibrate-dataset quant.txt \
    -o model_qnn \
	model.relay
```

##### 2.2.3 codegen

```Python
python hhb.py codegen \
	--board x86_ref \
	-o quant_codegen \
	model_qnn \
```

##### 2.2.4 simulate

```Python
python hhb.py simulate \
	--simulate-data /lhome/fern/aone/hhb/tests/thead/images/n01440764_188.JPEG \
	--data-mean "103.94,116.98,123.68" \
	--data-scale 1 \
	--data-resize 256 \
	--postprocess top5 \
	-o output \
	quant_codegen \
```


#### 2.3 using config file

You can generate a template config file by:

```bash
python hhb.py --generate-config -o config.yaml
```

change the part parameters...

The use the config file by:

```Bash
python hhb.py --config-file config.yaml --file mobilenetv1.prototxt mobilenetv1.caffemodel
```
