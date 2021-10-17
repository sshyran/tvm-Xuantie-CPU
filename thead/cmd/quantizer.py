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
# pylint: disable=unnecessary-comprehension
"""
Optimize the imported model.
"""
import logging
import tarfile
import tempfile
import os
from collections import namedtuple

import tvm
from tvm.relay import quantize as qtz

from utils import hhb_register_parse
from utils import HHBException
from utils import HHBModel
from utils import get_input_info_from_relay
from utils import match_mod_params
from utils import add_preprocess_argument
from utils import add_quantize_argument
from utils import add_common_argument
from utils import add_optimize_argument
from utils import save_relay_module
from utils import save_quantize_config
from model_evaluation import PreprocessParams
from model_evaluation import DatasetLoader


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


@hhb_register_parse
def add_quantize_parser(subparsers):
    """ Include parser for 'quantize' subcommand """

    parser = subparsers.add_parser("quantize", help="Quantize the imported model")
    parser.set_defaults(func=driver_quantize)

    add_preprocess_argument(parser)
    add_quantize_argument(parser)
    add_optimize_argument(parser)
    add_common_argument(parser)

    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        default="model_qnn",
        help="The directory that holds the quantized relay ir.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("FILE", help="Directory to the model file")


def driver_quantize(args):
    """Driver quantize command"""
    if not os.path.exists(args.FILE) or not os.path.isdir(args.FILE):
        raise HHBException("The directory is not exists: {}".format(args.FILE))
    mod, params = get_model(args.FILE)
    input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
    preprocess_params = PreprocessParams(
        mean=args.data_mean,
        scale=args.data_scale,
        resize_base=args.data_resize,
        pixel_format=args.pixel_format,
    )
    input_shape_dict = {name: shape for name, shape in zip(input_name_list, input_shape_list)}
    logger.debug("get calibrate dataset from %s", args.calibrate_dataset)
    dl = DatasetLoader(args.calibrate_dataset, input_shape_dict, preprocess_params)
    dataset = dl.get_data()
    dataset_list = []
    for d in dataset:
        dataset_list.append(d)
    qconfig, qnn_config_dict = get_quantize_config(
        args.board,
        args.num_bit_activation,
        args.dtype_input,
        args.dtype_weight,
        args.calibrate_mode,
        args.quantized_type,
        args.weight_scale,
        args.fuse_relu,
        args.channel_quantization,
        args.broadcast_quantization,
    )

    qfunc = quantize_model(mod, params, qconfig, dataset_list)
    save_relay_module(qfunc, None, args.output, HHBModel.QNN)

    all_config_dict = {"preprocess": preprocess_params.__dict__, "qnn_config": qnn_config_dict}
    save_quantize_config(all_config_dict, args.output)


def quantize_model(mod, params, qconfig, dataset=None):
    """Quantize the imported relay module.

    Parameters
    ----------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    qconfig : tvm.relay.quantize.QConfig
        The config parameter for quantization
    dataset : data generator
        The dict of input_name(str) to numpy.ndarray

    Returns
    -------
    qfunc : Function
        The graph after quantization
    """
    if not isinstance(qconfig, qtz.QConfig):
        raise HHBException("Invalid qconfig(QConfig):{}".format(qconfig))
    with qconfig:
        logger.debug("current quantize config:")
        logger.debug(qtz.current_qconfig())

        qfunc = qtz.quantize_hhb(mod, params, dataset=dataset)
    return qfunc


def get_model(model_path):
    """Get module from file.

    Parameters
    ----------
    model_path : str
        The path of model to be quantized

    Returns
    -------
    mod : tvm.IRModule
        The relay module
    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    """
    model_type = HHBModel.guess_model(model_path)
    if model_type is None or model_type != HHBModel.RELAY:
        raise HHBException(
            "invalid module:{}, please get valid relay module "
            "by executing 'import' subcommand.".format(model_path)
        )
    mode_type_str = HHBModel.TYPE2NAME[model_type]
    with open(os.path.join(model_path, mode_type_str + ".txt"), "r") as f:
        mod = tvm.parser.fromtext(f.read())
    with open(os.path.join(model_path, mode_type_str + ".params"), "rb") as f:
        params = tvm.relay.load_param_dict(f.read())
    mod, params = match_mod_params(mod, params)
    return mod, params


def get_quantize_config(
    board,
    num_bit_activation,
    dtype_input,
    dtype_weight,
    calibrate_mode,
    quantized_type,
    weight_scale,
    fuse_relu,
    channel_quantization,
    broadcast_quantization,
):
    """ Generate the config parameters for quantization. """
    quant_config_t = namedtuple(
        "quant_config",
        [
            "nbit_input",
            "nbit_weight",
            "nbit_activation",
            "dtype_input",
            "dtype_weight",
            "dtype_activation",
            "calibrate_mode",
            "quantized_type",
            "weight_scale",
            "fuse_relu",
            "channel_quantization",
            "broadcast_quantization",
        ],
    )

    if board == "anole":
        quant_config = quant_config_t(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=num_bit_activation,
            dtype_input="uint8",
            dtype_weight="uint8",
            dtype_activation="int32",
            calibrate_mode=calibrate_mode,
            quantized_type=quantized_type,
            weight_scale="max",
            fuse_relu=False,
            channel_quantization=channel_quantization,
            broadcast_quantization=True,
        )
    else:
        quant_config = quant_config_t(
            nbit_input=8,
            nbit_weight=8,
            nbit_activation=num_bit_activation,
            dtype_input=dtype_input + "8",
            dtype_weight=dtype_weight + "8",
            dtype_activation="int" + str(num_bit_activation),
            calibrate_mode=calibrate_mode,
            quantized_type=quantized_type,
            weight_scale=weight_scale,
            fuse_relu=fuse_relu,
            channel_quantization=channel_quantization,
            broadcast_quantization=broadcast_quantization,
        )
    config_dict = {
        "nbit_input": quant_config.nbit_input,
        "nbit_weight": quant_config.nbit_weight,
        "nbit_activation": quant_config.nbit_activation,
        "dtype_input": quant_config.dtype_input,
        "dtype_weight": quant_config.dtype_weight,
        "dtype_activation": quant_config.dtype_activation,
        "calibrate_mode": quant_config.calibrate_mode,
        "quantized_type": quant_config.quantized_type,
        "weight_scale": quant_config.weight_scale,
        "fuse_relu": quant_config.fuse_relu,
        "channel_quantization": quant_config.channel_quantization,
        "broadcast_quantization": quant_config.broadcast_quantization,
    }
    qconfig = qtz.qconfig(**config_dict)
    return qconfig, config_dict
