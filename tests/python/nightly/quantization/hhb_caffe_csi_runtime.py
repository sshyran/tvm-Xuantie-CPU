# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import numpy as np
from tvm.relay.frontend import caffe_pb2
from PIL import Image
import tvm
from tvm import relay
import importlib
from google.protobuf import text_format
import time
import logging
from tvm.contrib import graph_runtime
from tvm.contrib import hhb_runtime
from tvm.relay import quantize as qtz
import os
import sys
from tvm import runtime

logging.basicConfig(level=20)

model_path = "../../../../../ai-bench/net/caffe/"
image_path = "../../../../../ai-bench/config_file/"
model_map = {
    "lenet": [
        model_path + "lenet/lenet.prototxt",
        model_path + "lenet/lenet.caffemodel",
        (28, 28),
        image_path + "1.jpg",
        (63.5),
        (63.5),
    ],
    "alexnet": [
        model_path + "alexnet/alexnet.prototxt",
        model_path + "alexnet/alexnet.caffemodel",
        (227, 227),
        image_path + "cat.jpg",
        (1),
        (0, 0, 0),
    ],
    "resnet50": [
        model_path + "resnet/resnet50.prototxt",
        model_path + "resnet/resnet50.caffemodel",
        (224, 224),
        image_path + "cat.jpg",
        (1),
        (0, 0, 0),
    ],
    "inceptionv1": [
        model_path + "inception/inceptionv1.prototxt",
        model_path + "inception/inceptionv1.caffemodel",
        (224, 224),
        image_path + "cat.jpg",
        (1),
        (0, 0, 0),
    ],
    "inceptionv3": [
        model_path + "inception/inceptionv3.prototxt",
        model_path + "inception/inceptionv3.caffemodel",
        (299, 299),
        image_path + "cat.jpg",
        (58.8),
        (103.94, 116.98, 123.68),
    ],
    "inceptionv4": [
        model_path + "inception/inceptionv4.prototxt",
        model_path + "inception/inceptionv4.caffemodel",
        (299, 299),
        image_path + "cat.jpg",
        (58.8),
        (103.94, 116.98, 123.68),
    ],
    "mobilenetv1": [
        model_path + "mobilenet/mobilenetv1.prototxt",
        model_path + "mobilenet/mobilenetv1.caffemodel",
        (224, 224),
        image_path + "cat.jpg",
        (58.8),
        (103.94, 116.98, 123.68),
    ],
    "mobilenetv2": [
        model_path + "mobilenet/mobilenetv2.prototxt",
        model_path + "mobilenet/mobilenetv2.caffemodel",
        (224, 224),
        image_path + "cat.jpg",
        (58.8),
        (103.94, 116.98, 123.68),
    ],
    "squeezenet_v1": [
        model_path + "squeezenet/squeezenet_v1.0.prototxt",
        model_path + "squeezenet/squeezenet_v1.0.caffemodel",
        (227, 227),
        image_path + "cat.jpg",
        (1),
        (104, 117, 124),
    ],
    "ssd": [
        model_path + "ssd/ssdvgg16.prototxt",
        model_path + "ssd/ssdvgg16.caffemodel",
        (300, 300),
        image_path + "cat.jpg",
        (127.5),
        (123, 117, 104),
    ],
    "ssdmobilenetv1": [
        model_path + "ssd/ssdmobilenetv1.prototxt",
        model_path + "ssd/ssdmobilenetv1.caffemodel",
        (300, 300),
        image_path + "cat.jpg",
        (127.5),
        (104, 117, 124),
    ],
    "ssdmobilenetv2": [
        model_path + "ssd/ssdmobilenetv2.prototxt",
        model_path + "ssd/ssdmobilenetv2.caffemodel",
        (300, 300),
        image_path + "cat.jpg",
        (127.5),
        (104, 117, 124),
    ],
    "vgg16": [
        model_path + "vgg/vgg16.prototxt",
        model_path + "vgg/vgg16.caffemodel",
        (224, 224),
        image_path + "cat.jpg",
        (1),
        (0, 0, 0),
    ],
    "rfcn": [
        model_path + "rfcn/resnet101_rfcn_voc.prototxt",
        model_path + "rfcn/resnet101_rfcn_voc.caffemodel",
        (224, 224),
        image_path + "cat.jpg",
        (127.5),
        (104, 117, 124),
    ],
}


def image_preprocess(model_name, image_path, image_shape, norm, mean_value):
    sys.path.append("../../../../thead/command")
    from model_evaluation import DataPreprocess

    dp = DataPreprocess()

    if model_name != "lenet":
        dp.load_image(image_path)
    else:
        dp.load_image(image_path, True)

    dp.img_resize(resize_shape=image_shape)
    dp.sub_mean(mean_value)
    dp.data_scale(1.0 / norm)
    dataset = np.asarray(dp.get_data())
    dataset = np.transpose(dataset, (0, 3, 1, 2))

    return dataset


def get_top5(prediction):
    """return top5 index and value of input array"""
    length = np.prod(prediction.size)
    pre = np.reshape(prediction, [length])
    ind = np.argsort(pre)

    ind = ind[length - 5 :]
    value = pre[ind]

    ind = ind[::-1]
    value = value[::-1]
    res_str = ""
    logging.info("============ top5 ===========")
    for (i, v) in zip(ind, value):
        logging.info("{}:{}".format(i, v))
        res_str = res_str + "{}:{}".format(i, v) + "\n"
    return res_str


def read_model(model_name, input, proto_file, blob_file):

    init_net = caffe_pb2.NetParameter()
    predict_net = caffe_pb2.NetParameter()

    # load model
    with open(proto_file, "r") as f:
        text_format.Merge(f.read(), predict_net)
    # load blob
    with open(blob_file, "rb") as f:
        init_net.ParseFromString(f.read())

    shape_dict = {"data": input.shape}
    dtype_dict = {"data": "float32"}

    if model_name == "rfcn":
        shape_dict = {"data": [1, 3, 224, 224], "im_info": [1, 3]}
        dtype_dict = {"data": "float32", "im_info": "float32"}
    mod, params, _ = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)
    return mod, params


def run_quant_tvm(model_name, input, mod, params, quant=False):
    if model_name == "rfcn":
        img_list = {"data": input, "im_info": np.asarray([[224, 224, 1]])}
    else:
        img_list = {"data": input}
    if quant:
        #        with relay.build_config(opt_level=3):
        #            mod = relay.quantize.prerequisite_optimize(mod, params=params)
        qconfig = qtz.qconfig(
            skip_conv_layers=[-1],
            nbit_input=8,
            weight_scale="max",
            quantized_type="asym",
            nbit_weight=8,
            dtype_input="uint8",
            dtype_weight="uint8",
            dtype_activation="int32",
            debug_enabled_ops=None,
            calibrate_mode="global_scale",
            do_simulation=False,
        )
        with qconfig:
            mod = qtz.quantize_hhb(mod, params=params, dataset=[img_list])

    target = "llvm"
    layout = "NCHW"
    ctx = tvm.cpu(0)

    tvm_output = []
    if quant:
        func = mod["main"]
        func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
        mod["main"] = func

        with relay.build_config(opt_level=2):
            lib, params = relay.build_hhb(mod, target=target, params=params)
        func = mod["main"]
        params = func.params

        m = tvm.contrib.hhb_runtime.create(lib, mod, ctx)
        m.set_params("model.params")
        m.run(**img_list)
        for i in range(m.get_num_outputs()):
            tvm_output.append(m.get_output(i).asnumpy())
    else:
        with relay.build_config(opt_level=2):
            graph, lib, params = relay.build(mod, target=target, params=params)

        dtype = "float32"
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input("data", tvm.nd.array(input.astype(dtype)))
        if model_name == "rfcn":
            im_info = np.array([224, 224, 1], dtype=np.float32)
            m.set_input("im_info", tvm.nd.array(im_info))
        m.set_input(**params)
        m.run()
        for i in range(m.get_num_outputs()):
            tvm_output.append(m.get_output(i).asnumpy())

    return tvm_output


def test_quantization_tvm(
    model_name,
    image=None,
    image_shape=(224, 224),
    prototxt=None,
    caffemodel=None,
    quant=True,
    norm=(127.5),
    mean_value=(104, 117, 124),
):

    image = image_preprocess(model_name, image, image_shape, norm, mean_value)

    mod, params = read_model(model_name, image, proto_file=prototxt, blob_file=caffemodel)

    tvm_output = run_quant_tvm(model_name, input=image, mod=mod, params=params, quant=False)
    if quant:
        q_output = run_quant_tvm(model_name, input=image, mod=mod, params=params, quant=True)
        return tvm_output, q_output


def test_single():
    models = [
        "mobilenetv1",
        "mobilenetv2",
        "alexnet",
        "resnet50",
        "inceptionv1",
        "inceptionv3",
        "inceptionv4",
        "vgg16",
        "squeezenet_v1",
        "ssdmobilenetv1",
        "ssdmobilenetv2",
        "rfcn",
        "lenet",
    ]

    for model_name in models:
        prototxt = model_map[model_name][0]
        caffemodel = model_map[model_name][1]
        image_shape = model_map[model_name][2]
        image = model_map[model_name][3]
        norm = model_map[model_name][4]
        mean_value = model_map[model_name][5]
        quant = True
        tvm_out, q_tvm_out = test_quantization_tvm(
            model_name, image, image_shape, prototxt, caffemodel, quant, norm, mean_value
        )

        np.set_printoptions(threshold=np.inf)
        #    print("==================origin output====================")
        #    print(tvm_out)
        #    print("==================quantization output==============")
        #    print(q_tvm_out)
        print("==================origin mean std max==============")
        for float_out in tvm_out:
            print(np.mean(float_out))
            print(np.std(float_out))
            get_top5(np.array(float_out))

        print("==================quantization mean std max=========")
        for quant_out in q_tvm_out:
            print(np.mean(quant_out))
            print(np.std(quant_out))
            get_top5(np.array(quant_out))


if __name__ == "__main__":
    test_single()
