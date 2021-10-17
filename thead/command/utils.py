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
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=missing-function-docstring, no-else-return, inconsistent-return-statements
# pylint: disable=logging-format-interpolation
""" Utils for HHB Command line tools """
import logging
import os
import sys
import pickle
from collections import namedtuple
import time
import shutil


import numpy as np
from google.protobuf import text_format

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# pylint: disable=wrong-import-position
import tvm
import tvm.relay.frontend.caffe_pb2 as pb
from tvm import relay
from tvm.contrib import graph_runtime, hhb_runtime
import tvm.relay.testing.tf as tf_testing
from tvm.relay import quantize as qtz
from tvm.contrib import hhb_runtime
from model_evaluation import DataPreprocess, ModelKind, DataLayout

# pylint: enable=wrong-import-position


def hhb_version():
    __version__ = "1.0.0"
    __build_time__ = "20200901"
    return "HHB version: " + __version__ + ", build " + __build_time__


def hhb_exit(error_msg):
    logging.error(error_msg)
    sys.exit(1)


class PreprocessParams(object):
    def __init__(self):
        self.mean = list()
        self.scale = None
        self.resize = list()
        self.crop = list()
        self.is_rgb = True
        self.disable = False


class WorkspaceArgs(object):
    """Set the parameter that will be used in current workspace."""

    def __init__(self, args):
        """Initialize the parameters of current workspace.

        Parameters
        ----------
        args : argparse.Namespace

        """
        self.args = args
        self.opt_level = self._get_opt_level()
        self.input_name = self._get_input_names()
        self.input_shape = self._get_input_shapes()
        self.output_name = self._get_output_names()
        self.output_shape = list()
        self.data_layout = None
        self.preprocess = PreprocessParams()
        self.input_filenames = None
        self.model_name = self._get_model_name()

    def _get_model_name(self):
        model_kind = get_model_kind(self.args)
        if model_kind == ModelKind.TENSORFLOW:
            model_name = os.path.basename(self.args.tf_pb.strip())
            if "." in model_name:
                model_name = model_name.split(".")[0]
        elif model_kind == ModelKind.CAFFE:
            model_name = os.path.basename(self.args.caffe_proto.strip())
            if "." in model_name:
                model_name = model_name.split(".")[0]
        else:
            model_name = "model"
        return model_name

    def _get_opt_level(self):
        return 2

    def _get_input_names(self):
        model_kind = get_model_kind(self.args)
        input_names = None
        if model_kind == ModelKind.TENSORFLOW:
            # parse input name of model
            if self.args.input_name is None:
                input_names = ["Placeholder"]
            else:
                input_names = self.args.input_name.strip().split(";")
        elif model_kind == ModelKind.CAFFE:
            if self.args.input_name is None:
                input_names = None
            else:
                input_names = self.args.input_name.strip().split(";")
        return input_names

    def _get_input_shapes(self):
        input_shapes = None
        if self.args.input_shape is not None:
            shapes_tmp = self.args.input_shape.strip().split(";")
            shapes_tmp = [ishape.strip().split(" ") for ishape in shapes_tmp]
            input_shapes = list()
            for st in shapes_tmp:
                input_shapes.append(list(map(int, st)))
        return input_shapes

    def _get_output_names(self):
        output_names = None
        if self.args.output_name is not None:
            output_names = self.args.output_name.strip().split(";")
            output_names = [item.strip() for item in output_names]
        return output_names

    def print_cmd_args(self):
        """Update the self.args according to the other attrs and print."""
        self.args.opt_level = self.opt_level
        self.args.input_name = self.input_name
        self.args.input_shape = self.input_shape
        self.args.output_name = self.output_name
        print(self.args)


def get_model_kind(args):
    """Obtain the kind of model framework.

    Parameters
    ----------
    args: argparse.Namespace
    """
    if args.tf_pb is not None:
        return ModelKind.TENSORFLOW
    if args.caffe_proto is not None:
        return ModelKind.CAFFE
    return ModelKind.NOMODEL


def print_top5(value, output_name):
    if not isinstance(value, np.ndarray):
        hhb_exit("Unsupport for {}, please input ndarray".format(type(value)))
    len_t = np.prod(value.size)
    pre = np.reshape(value, [len_t])
    ind = np.argsort(pre)
    ind = ind[len_t - 5 :]
    value = pre[ind]
    ind = ind[::-1]
    value = value[::-1]
    print("============ {} top5: ===========".format(output_name))
    for (i, v) in zip(ind, value):
        print("{}:{}".format(i, v))


def import_caffe_model(wksp_args):
    proto_file = wksp_args.args.caffe_proto
    blob_file = wksp_args.args.caffe_blobs
    init_net = pb.NetParameter()
    predict_net = pb.NetParameter()
    # load proto
    with open(proto_file, "r") as f:
        text_format.Merge(f.read(), predict_net)
    # load blobs
    with open(blob_file, "rb") as f:
        init_net.ParseFromString(f.read())
    shape_dict = dict()
    dtype_dict = dict()

    input_name_list = wksp_args.input_name
    input_shape_list = wksp_args.input_shape
    if input_name_list is not None or input_shape_list is not None:
        assert input_name_list
        assert input_shape_list
        assert len(input_name_list) == len(
            input_shape_list
        ), "The length of \
                input_name must be equal to that of input_shape."

        for idx, name in enumerate(input_name_list):
            shape_dict[name] = input_shape_list[idx]
            dtype_dict[name] = "float32"
    else:
        # get default shape and dtype of input layer
        if len(predict_net.input) != 0:  # old caffe
            input_name = list(predict_net.input)
            for idx, iname in enumerate(input_name):
                shape_dict[iname] = list(predict_net.input_shape[idx].dim)
                dtype_dict[iname] = "float32"
        else:  # new caffe
            for layer in predict_net.layer:
                if layer.type == "Input":
                    iname = layer.top[0]
                    shape_dict[iname] = list(layer.input_param.shape[0].dim)
                    dtype_dict[iname] = "float32"
        wksp_args.input_name = list(shape_dict.keys())
        wksp_args.input_shape = list(shape_dict.values())
    logging.debug(shape_dict)
    logging.debug(dtype_dict)

    # get tvm relay ir
    mod, params, model_outputs = relay.frontend.from_caffe(
        init_net, predict_net, shape_dict, dtype_dict
    )
    wksp_args.output_name = model_outputs
    return mod, params, model_outputs


def import_tf_model(wksp_args):
    assert wksp_args.args.tf_pb
    pb_file = wksp_args.args.tf_pb
    input_names = wksp_args.input_name
    input_shapes = wksp_args.input_shape

    if len(input_names) != len(input_shapes):
        hhb_exit(
            "The number of input name(--input-name) shoud be equal to"
            "that of input shape(--input-shape), but len(--input-name)"
            "={} and len(--input-shpae)={}".format(len(input_names), len(input_shapes))
        )
    input_dict = dict()
    for idx, name in enumerate(input_names):
        input_dict[name] = list(map(int, input_shapes[idx]))
    output_names = wksp_args.output_name

    with tf_compat_v1.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        # Add shapes to the graph.
        with tf_compat_v1.Session() as sess:
            graph_def = tf_testing.AddShapesToGraphDef(sess, output_names)

    mod, params = relay.frontend.from_tensorflow(
        graph_def, layout="NCHW", shape=input_dict, outputs=output_names
    )
    return mod, params


def quantize_relay_ir(mod, params, cfg, dataset=None):
    qconfig = qtz.qconfig(
        nbit_input=cfg.nbit_input,
        nbit_weight=cfg.nbit_weight,
        nbit_activation=cfg.nbit_activation,
        dtype_input=cfg.dtype_input,
        dtype_weight=cfg.dtype_weight,
        dtype_activation=cfg.dtype_activation,
        calibrate_mode=cfg.calibrate_mode,
        global_scale=cfg.global_scale,
        weight_scale=cfg.weight_scale,
        quantized_type=cfg.quantized_type,
        skip_conv_layers=[-1],
        do_simulation=cfg.do_simulation,
        round_for_shift=cfg.round_for_shift,
        debug_enabled_ops=cfg.debug_enabled_ops,
        rounding=cfg.rounding,
        fuse_relu=cfg.fuse_relu,
    )
    with qconfig:
        logging.debug("current quantize config")
        logging.debug(qtz.current_qconfig())

        qfunc = qtz.quantize_hhb(mod, params=params, dataset=dataset)
    return qfunc


def build_relay_ir(mod, params, target, opt_level, params_file):
    func = mod["main"]
    func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
    mod["main"] = func
    with relay.build_config(opt_level=opt_level):
        lib, params = relay.build_hhb(
            mod=mod, target=target, params=params, params_path=params_file
        )
    return lib, params


def optimize_relay_ir(mod, params, target, opt_level):
    with relay.build_config(opt_level=opt_level):
        mod, params = relay.optimize(mod=mod, target=target, params=params)
    return mod, params


def serialize_object(data, file):
    with open(file, "wb") as f:
        f.write(pickle.dumps(data))


def deserialize_object(data, file):
    with open(file, "rb") as f:
        data = pickle.loads(f.read())
    return data


def save_to_file(data, file):
    with open(file, "w") as f:
        f.write(data)


def dump_tvm_build(lib, params, lib_file, params_file):
    lib.save(lib_file)


def get_quant_config(args):
    Config = namedtuple(
        "Config",
        [
            "nbit_input",
            "nbit_weight",
            "nbit_activation",
            "dtype_input",
            "dtype_weight",
            "dtype_activation",
            "calibrate_mode",
            "global_scale",
            "weight_scale",
            "quantized_type",
            "skip_conv_layers",
            "do_simulation",
            "round_for_shift",
            "debug_enabled_ops",
            "rounding",
            "fuse_relu",
        ],
    )
    quant_config = [
        Config(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=32,
            dtype_input="int8",
            dtype_weight="int8",
            dtype_activation="int32",
            fuse_relu=True,
            calibrate_mode="global_scale",
            global_scale=8.0,
            weight_scale="max",
            quantized_type="sym",
            skip_conv_layers=[-1],
            do_simulation=False,
            round_for_shift=True,
            debug_enabled_ops=None,
            rounding="UPWARD",
        ),
        Config(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=32,
            dtype_input="uint8",
            dtype_weight="uint8",
            dtype_activation="int32",
            fuse_relu=True,
            calibrate_mode="global_scale",
            global_scale=8.0,
            weight_scale="max",
            quantized_type="asym",
            skip_conv_layers=[-1],
            do_simulation=False,
            round_for_shift=True,
            debug_enabled_ops=None,
            rounding="UPWARD",
        ),
        Config(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=32,
            dtype_input="uint8",
            dtype_weight="uint8",
            dtype_activation="int32",
            fuse_relu=False,
            calibrate_mode="global_scale",
            global_scale=8.0,
            weight_scale="max",
            quantized_type="asym",
            skip_conv_layers=[-1],
            do_simulation=False,
            round_for_shift=True,
            debug_enabled_ops=None,
            rounding="UPWARD",
        ),
    ]
    if args.quantized_type == "sym":
        current_quant_config = quant_config[0]
    elif args.quantized_type == "asym":
        current_quant_config = quant_config[1]

    if args.board == "anole":
        current_quant_config = quant_config[2]

    return current_quant_config


def get_data_layout(shape):
    assert shape
    assert len(shape) == 4, "The dim of data should be 4."
    curr_shape = shape[1:]
    if 1 in curr_shape:
        index = curr_shape.index(1)
    elif 3 in curr_shape:
        index = curr_shape.index(3)
    else:
        hhb_exit("The dim of data channel is not 1 or 3.")
    if index == 0:
        return DataLayout.NCHW
    elif index == 2:
        return DataLayout.NHWC
    else:
        hhb_exit("Unable to infer data layout.")


def load_data(dataset_path, wksp_args):
    input_shape = wksp_args.input_shape[0]
    data_layout = get_data_layout(input_shape)
    wksp_args.data_layout = data_layout
    gray = False
    if 1 in input_shape[1:]:
        gray = True
        wksp_args.preprocess.is_rgb = False

    data_h_w = tuple()
    if data_layout == DataLayout.NCHW:
        data_h_w = (input_shape[2], input_shape[3])
    else:
        data_h_w = (input_shape[1], input_shape[2])

    wksp_args.preprocess.crop = list(data_h_w)

    mean_value_str = wksp_args.args.input_mean.split(" ")
    mean_value = list()
    mean_value = [float(i) for i in mean_value_str if i]

    if len(mean_value) == 1:
        wksp_args.preprocess.mean = list(mean_value) * 3
    else:
        wksp_args.preprocess.mean = list(mean_value)

    dp = DataPreprocess()
    dataset_path = dataset_path.strip()
    suffix = dataset_path.split(".")[-1].lower()
    if suffix in ("jpg", "png", "jpeg"):
        dp.load_image(dataset_path, gray=gray)
    else:
        dp.load_dataset(dataset_path, gray=gray)

    wksp_args.input_filenames = dp.origin_filenames

    if wksp_args.args.resize_base:
        dp.img_resize(resize_shape=int(wksp_args.args.resize_base))
        dp.img_crop(crop_shape=data_h_w)

        wksp_args.preprocess.resize = [int(wksp_args.args.resize_base), 0]
    else:
        dp.img_resize(resize_shape=data_h_w)
        wksp_args.preprocess.resize = list(data_h_w)

    dp.sub_mean(mean_val=mean_value)
    dp.data_scale(1.0 / wksp_args.args.input_normal)

    wksp_args.preprocess.scale = 1.0 / wksp_args.args.input_normal

    if data_layout == DataLayout.NCHW:
        model_kind = get_model_kind(wksp_args.args)
        if model_kind == ModelKind.TENSORFLOW and input_shape[1] == 3:
            dp.channel_swap((2, 1, 0))
    dp.data_expand_dim()
    dp.data_transpose((0, 3, 1, 2))
    dataset = dp.get_data()
    if len(dataset) == 0:
        hhb_exit("No valid images")
    else:
        dataset = np.concatenate(dataset, axis=0)
    return dataset


def generate_batch_dataset(dataset, wksp_args):
    num = dataset.shape[0]
    batch_size = wksp_args.input_shape[0][0]
    input_name = wksp_args.input_name[0]
    data_list = list()

    epoch = num // batch_size
    curr_dataset = dataset[: batch_size * epoch, :, :, :]
    split_dataset = np.split(curr_dataset, epoch, axis=0)

    for item in split_dataset:
        data_list.append({input_name: item})
    return data_list


def get_input_data(dataset_path, wksp_args):
    dataset = None
    if wksp_args.args.dataset or wksp_args.args.input_dataset:
        input_name_list = wksp_args.input_name
        suffix = dataset_path.split(".")[-1]
        if suffix == "npz":
            npz_data = np.load(dataset_path)
            for i_name in input_name_list:
                if i_name not in npz_data.keys():
                    hhb_exit(
                        "The input data {} is not in the imported .npz file, "
                        "Please rebuild the input file."
                    )
            dataset = [npz_data]
            wksp_args.preprocess.disable = True
        else:
            if len(input_name_list) > 1:
                hhb_exit(
                    "The number of model input is more than 1, please "
                    "use .npz file by '--dataset'(numpy data format) instead."
                )
            dataset = load_data(dataset_path, wksp_args)
            dataset = generate_batch_dataset(dataset, wksp_args)
    else:
        hhb_exit("Please set '--dataset' or '--input-dataset'.")
    return dataset


def get_target_device(wksp):
    if wksp.args.simulate:
        wksp.args.target = "llvm"
        return wksp.args.target

    if wksp.args.board == "anole":
        wksp.args.target = (
            "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard " "-device=anole --system-lib"
        )
    elif wksp.args.board == "c860":
        wksp.args.target = (
            "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard " "-device=c860 --system-lib"
        )
    elif wksp.args.board == "x86_ref":
        wksp.args.target = "llvm --system-lib"

    return wksp.args.target


def _check_option_no_quantize(cmd_args):
    if cmd_args.no_quantize and cmd_args.Q:
        hhb_exit("Can't set '--no-quantize' while setting '-Q'.")
    if cmd_args.no_quantize and cmd_args.quantized_type:
        hhb_exit("Can't set '--no-quantize' while setting '--quantized-type'.")


def _check_option_target(cmd_args):
    if cmd_args.target and cmd_args.simulate:
        tmp_target = "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard -device=c860 --system-lib"
        if not cmd_args.target == tmp_target:
            if not cmd_args.target == "llvm":
                hhb_exit("Only support '--target=llvm' while setting '--simulate'.")


def _check_option_quantized_type(cmd_args):
    if cmd_args.quantized_type and not cmd_args.quantized_type == "asym":
        hhb_exit("Only support '--quantized_type=asym'.")

    if not cmd_args.no_quantize:
        cmd_args.quantized_type = "asym"


def check_cmd_args(cmd_args):
    kind = get_model_kind(cmd_args)
    if kind == ModelKind.NOMODEL:
        hhb_exit(
            "Please either set Tensorflow model by '--tf-pb' or "
            "set Caffe model by '--caffe-proto' and '--caffe-blobs'."
        )

    _check_option_no_quantize(cmd_args)
    _check_option_target(cmd_args)
    _check_option_quantized_type(cmd_args)


def import_model(wksp_args):
    cmd_args = wksp_args.args
    model_kind = get_model_kind(cmd_args)
    if model_kind == ModelKind.CAFFE:
        if cmd_args.caffe_proto is None or cmd_args.caffe_blobs is None:
            hhb_exit("Need model files(--caffe-proto and --caffe-blobs)")
        mod, params, model_outputs = import_caffe_model(wksp_args)
    elif model_kind == ModelKind.TENSORFLOW:
        mod, params = import_tf_model(wksp_args)
        model_outputs = None
    # update the output_shape of model
    if hasattr(mod["main"].ret_type, "fields"):
        for i in mod["main"].ret_type.fields:
            wksp_args.output_shape.append(i.shape)
    else:
        wksp_args.output_shape.append(mod["main"].ret_type.shape)
    return mod, params, model_outputs


def quantize_model(mod, params, wksp_args):
    if not wksp_args.args.dataset:
        hhb_exit("Please set --dataset")
    dataset = get_input_data(wksp_args.args.dataset, wksp_args)

    cfg = get_quant_config(wksp_args.args)
    qfunc = quantize_relay_ir(mod, params, cfg, dataset=dataset)
    return qfunc


def optimize_model(mod, params, wksp_args):
    target = get_target_device(wksp_args)
    opt_level = wksp_args.opt_level
    mod, params = optimize_relay_ir(mod, params, target, opt_level)
    return mod, params


def codegen_model(mod, params, wksp_args, params_file):
    target = get_target_device(wksp_args)
    opt_level = wksp_args.opt_level
    lib, params = build_relay_ir(mod, params, target, opt_level, params_file)
    return lib, params


def preprocess_define(code_str, wksp_args):
    preprocess_params_code = ""
    preprocess_params_code += (
        "#define RESIZE_HEIGHT" + "       " + str(wksp_args.preprocess.resize[0]) + "\n"
    )
    preprocess_params_code += (
        "#define RESIZE_WIDTH" + "        " + str(wksp_args.preprocess.resize[1]) + "\n"
    )
    preprocess_params_code += (
        "#define CROP_HEGHT" + "          " + str(wksp_args.preprocess.crop[0]) + "\n"
    )
    preprocess_params_code += (
        "#define CROP_WIDTH" + "          " + str(wksp_args.preprocess.crop[1]) + "\n"
    )
    preprocess_params_code += (
        "#define R_MEAN" + "              " + str(wksp_args.preprocess.mean[2]) + "\n"
    )
    preprocess_params_code += (
        "#define G_MEAN" + "              " + str(wksp_args.preprocess.mean[1]) + "\n"
    )
    preprocess_params_code += (
        "#define B_MEAN" + "              " + str(wksp_args.preprocess.mean[0]) + "\n"
    )
    preprocess_params_code += (
        "#define SCALE" + "               " + str(wksp_args.preprocess.scale) + "\n"
    )

    preprocess_params_code += """
void preprocess(struct image_data *img, int is_rgb, int to_bgr)
{
    uint32_t new_height, new_width;
    uint32_t min_side;
    if (is_rgb) {
        im2rgb(img);
    }
    if (RESIZE_WIDTH == 0) {
        min_side = MIN(img->shape[0], img->shape[1]);
        new_height = (uint32_t) (img->shape[0] * (((float)RESIZE_HEIGHT) / (float)min_side));
        new_width = (uint32_t) (img->shape[1] * (((float)RESIZE_HEIGHT) / (float)min_side));
        imresize(img, new_height, new_width);
    } else {
        imresize(img, RESIZE_HEIGHT, RESIZE_WIDTH);
    }
    data_crop(img, CROP_HEGHT, CROP_WIDTH);
    sub_mean(img, R_MEAN, G_MEAN, B_MEAN);
    data_scale(img, SCALE);
    if(to_bgr) {
        imrgb2bgr(img);
    }
    imhwc2chw(img);
}
    """

    code_str = code_str.replace("#_preprocess_define_#", preprocess_params_code)
    return code_str


def main_codegen(wksp_args, main_cc_file, model_outputs):
    if hasattr(sys, "_MEIPASS"):
        execute_path = os.path.dirname(os.path.realpath(sys.executable))
    else:
        execute_path, _ = os.path.split(os.path.abspath(__file__))

    if wksp_args.args.board == "anole":
        template_file = "config/anole.tp"
    else:
        template_file = "config/thead.tp"

    with open(os.path.join(execute_path, template_file), "r") as f:
        code_str = f.read()
    process_c_path = os.path.join(execute_path, "config", "process", "src", "process.c")
    process_c = _file_path(wksp_args.args.o, "process.c")
    process_h_path = os.path.join(execute_path, "config", "process", "include", "process.h")
    process_h = _file_path(wksp_args.args.o, "process.h")
    shutil.copy(process_h_path, process_h)
    shutil.copy(process_c_path, process_c)

    input_num_code = len(wksp_args.input_name)
    code_str = code_str.replace("#_input_num#", str(input_num_code))
    output_num_code = len(wksp_args.output_name)
    code_str = code_str.replace("#_output_num#", str(output_num_code))
    input_csinn_code = ""
    for i in range(input_num_code):
        input_csinn_code += "void *data" + str(i) + ", "
    code_str = code_str.replace("#_anole_csinn_args#", input_csinn_code)

    input_csinn_code = ""
    for i in range(input_num_code + output_num_code):
        input_csinn_code += "void *data" + str(i) + ", "
    code_str = code_str.replace("#_thead_csinn_args#", input_csinn_code)

    run_csinn_stats_anole = ""
    run_csinn_stats_thead = ""

    for i in range(input_num_code):
        run_csinn_stats_anole += "input[" + str(i) + "], "
        run_csinn_stats_thead += "input[" + str(i) + "], "

    for i in range(output_num_code):
        run_csinn_stats_thead += "output[" + str(i) + "], "

    is_rgb = int(wksp_args.preprocess.is_rgb)
    to_bgr = 0
    if get_model_kind(wksp_args.args) != ModelKind.TENSORFLOW:
        to_bgr = 1

    _is_rgb = str(is_rgb)
    _to_bgr = str(to_bgr)

    code_str = code_str.replace("#_is_rgb#", _is_rgb)
    code_str = code_str.replace("#_to_bgr#", _to_bgr)
    code_str = code_str.replace("#_anole_value_pass#", run_csinn_stats_anole)
    code_str = code_str.replace("#_thead_value_pass#", run_csinn_stats_thead)

    input_size_code = ""
    for ishape in wksp_args.input_shape:
        input_shape_str = list(map(str, ishape))
        input_shape_str = " * ".join(input_shape_str)
        input_size_code += input_shape_str + ", "
    code_str = code_str.replace("#_input_size_define#", input_size_code)

    output_size_code = ""
    for ishape in wksp_args.output_shape:
        output_shape_str = list(map(str, ishape))
        output_shape_str = " * ".join(output_shape_str)
        output_size_code += output_shape_str + ", "
    code_str = code_str.replace("#_output_size_define#", output_size_code)

    disable_preprocess = wksp_args.preprocess.disable
    if disable_preprocess:
        code_str = code_str.replace("#_preprocess_define_#", "#define preprocess(a, b, c)")
    else:
        code_str = preprocess_define(code_str, wksp_args)

    code_str = code_str.replace("#_hhb_version_#", hhb_version())
    code_str = code_str.replace("#_model_name_define#", wksp_args.model_name)
    save_to_file(code_str, main_cc_file)


def _ensure_dir(directory):
    """Create a directory if not exists

    Parameters
    ----------

    directory : str
        File path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def _file_path(output_dir, filename):
    if output_dir is None:
        output_dir = "hhb_out"
    _ensure_dir(output_dir)
    return os.path.join(output_dir, filename)


def command_E(wksp_args):
    """Execute script with '-E', which will import model into relay ir.

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    output_file = _file_path(wksp_args.args.o, "model.E")

    mod, _, _ = import_model(wksp_args)

    save_to_file(mod["main"].astext(show_meta_data=False), output_file)
    logging.info("Save the relay ir of the imported model into {}".format(output_file))


def command_Q(wksp_args):
    """Execute script with '-Q', which will quantize relay ir.

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    Q_file = _file_path(wksp_args.args.o, "model.Q")
    # import model
    mod, params, _ = import_model(wksp_args)
    if wksp_args.args.save_temps:
        E_file = _file_path(wksp_args.args.o, "model.E")
        save_to_file(mod.astext(show_meta_data=False), E_file)
        logging.info("Save the relay ir of the imported model into {}".format(E_file))
    # quantize model
    qfunc = quantize_model(mod, params, wksp_args)
    # save file
    save_to_file(qfunc.astext(show_meta_data=False), Q_file)
    logging.info("Save the relay ir that is quantized into {}".format(Q_file))


def command_S(wksp_args):
    """Execute script with '-S', which will optimize relay ir.

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    S_file = _file_path(wksp_args.args.o, "model.S")
    # import model
    mod, params, _ = import_model(wksp_args)
    if wksp_args.args.save_temps:
        E_file = _file_path(wksp_args.args.o, "model.E")
        save_to_file(mod.astext(show_meta_data=False), E_file)
        logging.info("Save the relay ir of the imported model into {}".format(E_file))
    # quantize model
    if not wksp_args.args.no_quantize:
        mod = quantize_model(mod, params, wksp_args)
        params = None
        if wksp_args.args.save_temps:
            Q_file = _file_path(wksp_args.args.o, "model.Q")
            save_to_file(mod.astext(show_meta_data=False), Q_file)
            logging.info("Save the relay ir that is quantized into {}".format(Q_file))
    # optimize model
    mod, params = optimize_model(mod, params, wksp_args)
    # save file
    save_to_file(mod.astext(show_meta_data=False), S_file)
    logging.info("Save the relay ir of the optimized model into {}".format(S_file))
    return mod, params


def command_C(wksp_args):
    """Execute script with '-C', which will build model into (lib, params).

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    lib_name = "model.c"
    if wksp_args.args.no_quantize:
        lib_name = "model.o"
    lib_file = _file_path(wksp_args.args.o, lib_name)
    params_file = _file_path(wksp_args.args.o, "model.params")
    # import model
    mod, params, _ = import_model(wksp_args)
    if wksp_args.args.save_temps:
        E_file = _file_path(wksp_args.args.o, "model.E")
        save_to_file(mod.astext(show_meta_data=False), E_file)
        logging.info("Save the relay ir of the imported model into {}".format(E_file))
    # quantize model
    if not wksp_args.args.no_quantize:
        mod = quantize_model(mod, params, wksp_args)
        params = None
        if wksp_args.args.save_temps:
            Q_file = _file_path(wksp_args.args.o, "model.Q")
            save_to_file(mod.astext(show_meta_data=False), Q_file)
            logging.info("Save the relay ir that is quantized into {}".format(Q_file))
    # optimize model
    mod_S, _ = optimize_model(mod, params, wksp_args)
    if wksp_args.args.save_temps:
        S_file = _file_path(wksp_args.args.o, "model.S")
        save_to_file(mod_S.astext(show_meta_data=False), S_file)
        logging.info("Save the relay ir of the optimized model into {}".format(S_file))
    # codegen model
    if wksp_args.args.no_quantize:
        with relay.build_config(opt_level=2):
            _, lib, params = relay.build(mod, wksp_args.args.target, params=params)
        with open(params_file, "wb") as f:
            f.write(relay.save_param_dict(params))
    else:
        lib, params = codegen_model(mod, params, wksp_args, params_file)
    dump_tvm_build(lib, params, lib_file, params_file)
    logging.info("Save lib -> {}, params -> {}.".format(lib_file, params_file))


def command_simulate(wksp_args):
    """Execute script with '--simulate', and dump the outputs of each layer.

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    # import model
    mod, params, _ = import_model(wksp_args)
    if wksp_args.args.save_temps:
        E_file = _file_path(wksp_args.args.o, "model.E")
        save_to_file(mod.astext(show_meta_data=False), E_file)
        logging.info("Save the relay ir of the imported model into {}".format(E_file))

    if wksp_args.args.o:
        dump_dir = wksp_args.args.o
    else:
        dump_dir = "./dump_data"
        curr_time_str = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        dump_dir = os.path.join(dump_dir, curr_time_str)
    _ensure_dir(dump_dir)
    wksp_args.args.target = "llvm"
    ctx = tvm.cpu(0)

    if wksp_args.args.input_dataset is None:
        wksp_args.args.input_dataset = wksp_args.args.dataset
        logging.info("The dataset that used to calibrate is used as the input of model.")

    if wksp_args.args.no_quantize:
        with relay.build_config(opt_level=2):
            graph, lib, params = relay.build(mod, target=wksp_args.args.target, params=params)
        m = graph_runtime.create(graph, lib, ctx)
        m.set_input(**params)
    else:
        params_file = _file_path(wksp_args.args.o, "model.params")
        mod = quantize_model(mod, params, wksp_args)
        lib, params = codegen_model(mod, params, wksp_args, params_file)
        m = hhb_runtime.create(lib, mod, ctx, output_dir=dump_dir)
        m.set_params(params_file)

    data_list = get_input_data(wksp_args.args.input_dataset, wksp_args)
    for idx, data in enumerate(data_list):
        m.run(**data)
        for i in range(m.get_num_outputs()):
            output = m.get_output(i).asnumpy()
            out = np.reshape(output, [np.prod(output.size)])
            try:
                output_save_name = wksp_args.output_name[i].replace("/", "_")
            except:
                output_save_name = wksp_args.output_name[i - 1].replace("/", "_") + str(i)
            if wksp_args.args.postprocess == "top5":
                print_top5(out, output_save_name)
            elif wksp_args.args.postprocess == "save":
                np.savetxt(
                    dump_dir + "/output_" + str(idx) + "_" + output_save_name,
                    out,
                    delimiter="\n",
                    newline="\n",
                )
            else:
                print_top5(out, output_save_name)
                np.savetxt(
                    dump_dir + "/output_" + str(idx) + "_" + output_save_name,
                    out,
                    delimiter="\n",
                    newline="\n",
                )


def command_deploy(wksp_args):
    """Execute script with '--deploy', and generate binary codes which will executed
         on the specific target.

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    if wksp_args.args.no_quantize:
        hhb_exit("Can't set '--no-quantize' while converting model to deploy.")

    lib_file = _file_path(wksp_args.args.o, "model.c")
    params_file = _file_path(wksp_args.args.o, "model.params")
    main_cc_file = _file_path(wksp_args.args.o, "main.c")

    # import model
    mod, params, model_outputs = import_model(wksp_args)
    if wksp_args.args.save_temps:
        E_file = _file_path(wksp_args.args.o, "model.E")
        save_to_file(mod.astext(show_meta_data=False), E_file)
        logging.info("Save the relay ir of the imported model into {}".format(E_file))
    # quantize model
    if not wksp_args.args.no_quantize:
        mod = quantize_model(mod, params, wksp_args)
        params = None
        if wksp_args.args.save_temps:
            Q_file = _file_path(wksp_args.args.o, "model.Q")
            save_to_file(mod.astext(show_meta_data=False), Q_file)
            logging.info("Save the relay ir that is quantized into {}".format(Q_file))
    # optimize model
    mod_S, _ = optimize_model(mod, params, wksp_args)
    if wksp_args.args.save_temps:
        S_file = _file_path(wksp_args.args.o, "model.S")
        save_to_file(mod_S.astext(show_meta_data=False), S_file)
        logging.info("Save the relay ir of the optimized model into {}".format(S_file))
    # codegen model
    lib, params = codegen_model(mod, params, wksp_args, params_file)
    # save generated model files
    dump_tvm_build(lib, params, lib_file, params_file)
    logging.info("Save lib -> {}, params -> {}.".format(lib_file, params_file))

    main_codegen(wksp_args, main_cc_file, model_outputs)

    if wksp_args.args.input_dataset is None:
        wksp_args.args.input_dataset = wksp_args.args.dataset
        logging.info("The dataset that used to calibrate is used as the input of model.")
    data_list = get_input_data(wksp_args.args.input_dataset, wksp_args)
    data_count = 0
    for data in data_list:
        for k, v in data.items():
            safe_k = k.replace("/", "_")
            v.tofile(_file_path(wksp_args.args.o, safe_k + ".{}.tensor".format(data_count)), "\n")
            v.tofile(_file_path(wksp_args.args.o, safe_k + ".{}.bin".format(data_count)))
            data_count += 1


def command_generate_input(wksp_args):
    """Generate input data after preprocessing.

    Parameters
    ----------
    wksp_args : WorkspaceArgs
    """
    if not wksp_args.args.input_dataset:
        hhb_exit("Please set --input-dataset")
    if not wksp_args.args.input_shape:
        hhb_exit("Please set --input-shape")

    ipath = wksp_args.args.input_dataset
    if os.path.isfile(ipath) and ipath.split(".")[-1] == "txt":
        with open(ipath, "r") as f:
            files = f.readlines()
        for input_file in files:
            dataset = load_data(input_file, wksp_args)
            num = dataset.shape[0]
            split_dataset = np.split(dataset, num, axis=0)
            for idx, filename in enumerate(wksp_args.input_filenames):
                i_path = _file_path(wksp_args.args.o, filename.split(".")[0] + ".tensor")
                split_dataset[idx].tofile(i_path, sep="\n")
    else:
        dataset = load_data(ipath, wksp_args)
        num = dataset.shape[0]
        split_dataset = np.split(dataset, num, axis=0)
        for idx, filename in enumerate(wksp_args.input_filenames):
            i_path = _file_path(wksp_args.args.o, filename.split(".")[0] + ".tensor")
            split_dataset[idx].tofile(i_path, sep="\n")
