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
import importlib
import tarfile
import argparse
import yaml
import math
import os

import numpy as np

import tvm
from tvm.contrib import util
from tvm.ir.tensor_type import TensorType
from tvm.ir.type import TupleType


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")
ALL_MODULES_FOR_REGISTER = ["importer", "quantizer", "codegen", "simulate"]
ALL_SUBCOMMAND = ["import", "quantize", "codegen", "simulate"]
HHB_REGISTERED_PARSER = []


def hhb_version():
    """Version information"""
    __version__ = "1.2.10"
    __build_time__ = "20201231"
    return "HHB version: " + __version__ + ", build " + __build_time__


class HHBException(Exception):
    """HHB Exception"""


def hhb_register_parse(make_subparser):
    """
    Utility function to register a subparser for HHB.

    Functions decorated with `hhb_register_parse` will be invoked
    with a parameter containing the subparser instance they need to add itself to,
    as a parser.

    Example
    -------

        @hhb_register_parse
        def _example_parser(main_subparser):
            subparser = main_subparser.add_parser('example', help='...')
            ...

    """
    HHB_REGISTERED_PARSER.append(make_subparser)
    return make_subparser


def import_module_for_register():
    """Dynamic importing libraries"""
    for m in ALL_MODULES_FOR_REGISTER:
        importlib.import_module(m)


def parse_node_name(name):
    """
    Parse the name string which is obtained from argparse

    The name may be include multi name which is separated by semicolon(;), and this
    function will convert name to list.

    Parameters
    ----------
    name : str
        The name string

    Returns
    -------
    name_list : list[str]
        The name list
    """
    if not name:
        return list()
    name_list = name.strip().split(";")
    name_list = list([n for n in name_list if n])
    name_list = [n.strip() for n in name_list]
    return list(name_list)


def parse_node_shape(shape):
    """
    Parse the shape string which is obtained from argparse

    There may be include multi shapes which is separated by semicolon(;), and this
    function will convert shape to list.

    Parameters
    ----------
    shape : str
        The shape string

    Returns
    -------
    shape_list : list[list[int]]
        The shape list
    """
    if not shape:
        return list()
    if "," in shape:
        shape = shape.replace(",", " ")
    shape_list = []
    shape_str_list = shape.strip().split(";")
    shape_str_list = list([n for n in shape_str_list if n])
    for shape_str in shape_str_list:
        tmp_list = shape_str.strip().split(" ")
        tmp_list = [int(i) for i in tmp_list]
        shape_list.append(tmp_list)
    return shape_list


class HHBModel(object):
    """Denote the HHB models"""

    RELAY = 0
    QNN = 1
    CODEGEN = 2

    TYPE2NAME = {RELAY: "relay", QNN: "qnn", CODEGEN: "codegen"}
    NAME2TYPE = {v: k for k, v in TYPE2NAME.items()}

    @staticmethod
    def guess_model(file_path):
        """Get the type of model accorrding to the tar file"""
        filenames = os.listdir(file_path)

        model_type = None
        for name, value in HHBModel.NAME2TYPE.items():
            if name + ".txt" in filenames:
                model_type = value
                break
        return model_type


def get_target(board):
    """Get the target info accorrding to the board type."""
    if board == "anole":
        target = "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard " "-device=anole"
    elif board == "c860":
        target = "llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard " "-device=c860"
    elif board == "x86_ref":
        target = "llvm"

    return target


def match_mod_params(mod, params):
    """ The params of module's main function match the params dict."""
    if not params:
        return mod, params
    var_name_list = []
    for arg in mod["main"].params:
        if arg.name_hint not in var_name_list:
            var_name_list.append(arg.name_hint)
    params_new = {}
    flag = False
    for k in params.keys():
        if k not in var_name_list and ("v" + k) in var_name_list:
            flag = True
            break

    if flag:
        logger.debug("mod does not match params, and try to update the params dict...")
        for k, v in params.items():
            params_new["v" + k] = v
        params = params_new

    return mod, params


def get_input_info_from_relay(mod, params):
    input_name_list = []
    input_shape_list = []
    input_dtype_list = []

    for arg in mod["main"].params:
        if (not params) or arg.name_hint not in params.keys():
            input_name_list.append(str(arg.name_hint))
            input_shape_list.append(list(map(int, arg.checked_type.shape)))
            input_dtype_list.append(str(arg.checked_type.dtype))

    return input_name_list, input_shape_list, input_dtype_list


def get_output_info_from_relay(mod):
    output_shape_list = []
    output_dtype_list = []

    if isinstance(mod["main"].ret_type, TupleType):
        for item in mod["main"].ret_type.fields:
            output_shape_list.append(list(map(int, item.shape)))
            output_dtype_list.append(str(item.dtype))
    elif isinstance(mod["main"].ret_type, TensorType):
        output_shape_list.append(list(map(int, mod["main"].ret_type.shape)))
        output_dtype_list.append(str(mod["main"].ret_type.dtype))
    else:
        raise HHBException("unsupport for {}".format(type(mod["main"].ret_type)))
    return output_shape_list, output_dtype_list


def save_relay_module(mod, params, module_path, model_type):
    """Saving the relay module into tar file."""
    output_dir = ensure_dir(module_path)
    mod_relay_name = HHBModel.TYPE2NAME[model_type] + ".txt"
    mod_params_name = HHBModel.TYPE2NAME[model_type] + ".params"

    mod_relay_path = os.path.join(output_dir, mod_relay_name)
    mod_params_path = os.path.join(output_dir, mod_params_name)

    with open(mod_relay_path, "w") as f:
        logger.debug("exporting relay to {}".format(f.name))
        f.write(mod.astext())

    if model_type != HHBModel.QNN:
        with open(mod_params_path, "wb") as f:
            logger.debug("exporting params to {}".format(f.name))
            f.write(tvm.relay.save_param_dict(params))


def save_quantize_config(config_dict, output_path):
    """Append the quantiztion configure file in the tar file"""
    config_file = os.path.join(output_path, "quantize_config.yaml")
    logger.debug("exporting config parameters into {}".format(config_file))
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)


def add_preprocess_argument(parser, is_subcommand=True):
    """ All preprocess parameters"""
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "-m",
        "--data-mean",
        type=str,
        default="0",
        metavar="",
        dest="data_mean" + suffix,
        help="Set the mean value of input, multiple values are separated by space, "
        "default is 0.",
    )
    parser.add_argument(
        "-s",
        "--data-scale",
        type=float,
        default="1",
        metavar="",
        dest="data_scale" + suffix,
        help="Divide number for inputs normalization, default is 1.",
    )
    parser.add_argument(
        "-r",
        "--data-resize",
        type=int,
        default=None,
        metavar="",
        dest="data_resize" + suffix,
        help="Resize base size for input image to resize.",
    )
    parser.add_argument(
        "--pixel-format",
        choices=["RGB", "BGR"],
        default="BGR",
        dest="pixel_format" + suffix,
        help="The pixel format of input data, defalut is RGB",
    )


def add_import_argument(parser, is_subcommand=True):
    """All parameters needed by 'import' subcommand"""
    import frontends

    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "-in",
        "--input-name",
        type=str,
        default=None,
        metavar="",
        dest="input_name" + suffix,
        help="Set the name of input node. If '--input-name'is None, "
        "default value is 'Placeholder'. Multiple values "
        "are separated by semicolon(;).",
    )
    parser.add_argument(
        "-is",
        "--input-shape",
        type=str,
        default=None,
        metavar="",
        dest="input_shape" + suffix,
        help="Set the shape of input nodes. Multiple shapes are separated "
        "by semicolon(;) and the dims between shape are separated "
        "by space.",
    )
    parser.add_argument(
        "-on",
        "--output-name",
        type=str,
        metavar="",
        dest="output_name" + suffix,
        help="Set the name of output nodes. Multiple shapes are " "separated by semicolon(;).",
    )
    parser.add_argument(
        "--model-format",
        choices=frontends.get_frontend_names(),
        dest="model_format" + suffix,
        help="Specify input model format",
    )


def add_optimize_argument(parser, is_subcommand=True):
    """All parameters needed by optimization"""
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "--board",
        default="anole",
        choices=["anole", "c860", "x86_ref"],
        dest="board" + suffix,
        help="Set target device, default is anole.",
    )
    parser.add_argument(
        "--opt-level",
        choices=[-1, 0, 1, 2, 3],
        default=3,
        type=int,
        dest="opt_level" + suffix,
        # help="Specify the optimization level, default is 3.",
        help=argparse.SUPPRESS,
    )


def add_quantize_argument(parser, is_subcommand=True):
    """All parameters needed by 'quantize' subcommand"""
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "-cd",
        "--calibrate-dataset",
        type=str,
        default=None,
        metavar="",
        dest="calibrate_dataset" + suffix,
        help="Provide with dataset for the input of model in reference step. "
        "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
        "of images. Note: only one image path in one line if .txt.",
    )
    parser.add_argument(
        "--num-bit-activation",
        type=int,
        choices=[16, 32],
        default=32,
        dest="num_bit_activation" + suffix,
        # help="The bit number that quantizes activation layer, default is 32.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype-input",
        choices=["int", "uint"],
        default="uint",
        dest="dtype_input" + suffix,
        # help="The dtype of quantized input layer, default is uint.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--dtype-weight",
        choices=["int", "uint"],
        default="uint",
        dest="dtype_weight" + suffix,
        # help="The dtype of quantized constant parameters, default is uint.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--calibrate-mode",
        choices=["maxmin", "global_scale", "kl_divergence", "kl_divergence_tsing"],
        default="maxmin",
        dest="calibrate_mode" + suffix,
        # help="How to calibrate while doing quantization, default is maxmin.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--quantized-type",
        choices=["asym", "sym"],
        default="asym",
        dest="quantized_type" + suffix,
        # help="Select the algorithm of quantization, default is asym.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--weight-scale",
        choices=["max", "power2"],
        default="max",
        dest="weight_scale" + suffix,
        # help="Select the mothod that quantizes weight value, default is max.",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fuse-relu",
        action="store_true",
        dest="fuse_relu" + suffix,
        help="Fuse the convolutioon and relu layer.",
    )
    parser.add_argument(
        "--channel-quantization",
        action="store_true",
        dest="channel_quantization" + suffix,
        help="Do quantizetion across channel.",
    )
    parser.add_argument(
        "--broadcast-quantization",
        action="store_true",
        dest="broadcast_quantization" + suffix,
        help="Broadcast quantization parameters for special ops.",
    )


def add_simulate_argument(parser, is_subcommand=True):
    """All parameters needed by 'simulate' subcommand"""
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "-sd",
        "--simulate-data",
        type=str,
        default=None,
        dest="simulate_data" + suffix,
        metavar="",
        help="Provide with dataset for the input of model in reference step. "
        "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
        "of images. Note: only one image path in one line if .txt.",
    )


def add_postprocess_argument(parser, is_subcommand=True):
    """All postprocess parameters"""
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "--postprocess",
        type=str,
        default="top5",
        choices=["top5", "save", "save_and_top5"],
        dest="postprocess" + suffix,
        help="Set the mode of postprocess: "
        "'top5' show top5 of output; "
        "'save' save output to file;"
        "'save_and_top5' show top5 and save output to file."
        " Default is top5",
    )


def add_main_argument(parser):
    """All commands that are compatible with previous version."""
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-E", action="store_true", help="Convert model into relay ir.")
    group.add_argument("-Q", action="store_true", help="Quantize the relay ir.")
    group.add_argument("-C", action="store_true", help="codegen the model.")
    group.add_argument("--simulate", action="store_true", help="Simulate model on x86 device.")
    group.add_argument(
        "--generate-config", action="store_true", help="Generate config file for HHB"
    )
    parser.add_argument(
        "--no-quantize", action="store_true", help="If set, don't quantize the model."
    )
    parser.add_argument("--save-temps", action="store_true", help="Save temp files.")


def add_common_argument(parser, is_subcommand=True):
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        metavar="",
        dest="config_file" + suffix,
        help="Configue more complex parameters for executing the model.",
    )


def add_codegen_argument(parser, is_subcommand=True):
    if not isinstance(parser, argparse.ArgumentParser):
        raise HHBException("invalid parser:{}".format(parser))
    suffix = ""
    if not is_subcommand:
        suffix = "_main"
    parser.add_argument(
        "--disable-binary-graph",
        action="store_true",
        dest="disable_binary_graph" + suffix,
        # help="Do not generate binary graph for anole.",
        help=argparse.SUPPRESS,
    )


def wraper_for_parameters(func, parser, is_subcommand, params_name, params_dict):

    before_args = vars(parser.parse_known_args()[0])
    if is_subcommand is None:
        func(parser)
    else:
        func(parser, is_subcommand)
    after_args = vars(parser.parse_known_args()[0])
    params_dict[params_name] = {
        key: value for key, value in after_args.items() if key not in before_args
    }


def print_top5(value, output_name):
    """Print the top5 infomation"""
    if not isinstance(value, np.ndarray):
        raise HHBException("Unsupport for {}, please input ndarray".format(type(value)))
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


def save_to_file(data, file):
    """Write the data into file"""
    with open(file, "w") as f:
        f.write(data)


def generate_config_file(parameter_dict, output_path):
    if not isinstance(parameter_dict, dict):
        raise HHBException("should be dict, but get {}".format(type(parameter_dict)))
    for name, value in parameter_dict.items():
        new_value = dict()
        for k, v in value.items():
            if k.endswith("_main"):
                new_value[k.split("_main")[0]] = v
            else:
                new_value[k] = v
        parameter_dict[name] = new_value

    logger.debug("save the parameters info into %s", output_path)
    if "generate_config" in parameter_dict["main"]:
        parameter_dict["main"]["generate_config"] = False
    with open(output_path, "w") as f:
        yaml.dump(parameter_dict, f, default_flow_style=False)


def read_params_from_file(config_file, is_subcommand=True):
    with open(config_file, "r") as f:
        params_dict = yaml.load(f.read())

    new_params_dict = {}
    for group_name, group_value in params_dict.items():
        if is_subcommand:
            new_params_dict.update(group_value)
        else:
            if group_name == "main":
                new_params_dict.update(group_value)
            else:
                for k, v in group_value.items():
                    new_params_dict[k + "_main"] = v

    return new_params_dict


def get_set_arguments(parser, commands=""):
    """Get the arguments that are set in command line."""
    args_before = vars(parser.parse_args(commands))
    args_after = vars(parser.parse_args())

    results = []
    for k, v in args_before.items():
        if isinstance(v, float):
            if not math.isclose(v, args_after[k], rel_tol=1e-5):
                results.append(k)
            continue

        if k in args_after and (v != args_after[k] or v is not args_after[k]):
            results.append(k)

    return results


def update_cmd_params(config_path, parser, args, is_subcommand, commands=""):
    """update command line parameters from config file."""
    config_dict = read_params_from_file(config_path, is_subcommand=is_subcommand)

    args_set_by_cmd = get_set_arguments(parser, commands=commands)
    logger.debug(
        "the arguments is set in command line and will not be overwritted: %s",
        str(args_set_by_cmd),
    )
    for name, value in config_dict.items():
        if hasattr(args, name) and name not in args_set_by_cmd:
            args.__dict__[name] = value


def ensure_dir(directory):
    """Create a directory if not exists

    Parameters
    ----------

    directory : str
        File path to create
    """
    if directory is None:
        directory = "hhb_out"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def get_file_path(output_dir, filename):
    output_dir = ensure_dir(output_dir)
    return os.path.join(output_dir, filename)


def check_cmd_arguments(argv):
    """Check whether there are any other arguments before subcommand"""
    for sub in ALL_SUBCOMMAND:
        if sub in argv:
            idx = argv.index(sub)
            if idx != 0:
                raise HHBException("do not allow any arguments before subcommand...")


def check_subcommand(argv):
    """Check whether including subcommand"""
    res = False
    for sub in ALL_SUBCOMMAND:
        if sub in argv:
            res = True
            break
    return res
