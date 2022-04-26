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
"""Manage quantization"""
import logging
from collections import namedtuple

import tvm
from tvm.relay import quantize as qtz

from .common import argument_filter_helper
from .common import ALL_ARGUMENTS_INFO
from .common import AttributeDict
from .common import HHBException
from .common import hhb_exit


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


def get_quantize_config(quantize_config: AttributeDict):
    if not isinstance(quantize_config, AttributeDict):
        raise HHBException("Need AttributeDidct object but get {}".format(type(quantize_config)))
    config_dict = {
        "nbit_input": quantize_config.num_bit_input,
        "nbit_weight": quantize_config.num_bit_weight,
        "nbit_activation": quantize_config.num_bit_activation,
        "dtype_input": quantize_config.dtype_input,
        "dtype_weight": quantize_config.dtype_weight,
        "dtype_activation": quantize_config.dtype_activation,
        "calibrate_mode": quantize_config.calibrate_mode,
        "activate_quantized_type": quantize_config.activate_quantized_type,
        "weight_quantized_type": quantize_config.weight_quantized_type,
        "weight_scale": quantize_config.weight_scale,
        "fuse_relu": quantize_config.fuse_relu,
        "fuse_clip": quantize_config.fuse_clip,
        "fuse_conv_relu": quantize_config.fuse_conv_relu,
        "fuse_reshape_dense": quantize_config.fuse_reshape_dense,
        "channel_quantization": quantize_config.channel_quantization,
        "broadcast_quantization": quantize_config.broadcast_quantization,
        "channel_quantization_ratio_threshold": quantize_config.channel_quantization_ratio_threshold,
        "fuse_mul_before_conv": quantize_config.fuse_mul_before_conv,
        "fuse_mul_after_conv": quantize_config.fuse_mul_after_conv,
        "fuse_add_before_conv": quantize_config.fuse_add_before_conv,
        "fuse_add_after_conv": quantize_config.fuse_add_after_conv,
        "layout": quantize_config.target_layout,
        "quantization_scheme": quantize_config.quantization_scheme,
        "fuse_zp2bias": quantize_config.fuse_zp2bias,
        "use_custom_fusion": quantize_config.use_custom_fusion,
        "convert_to_relay": quantize_config.convert_to_relay,
        "hybrid_quantization_scheme": quantize_config.hybrid_quantization_scheme,
        "hybrid_layer_name": quantize_config.hybrid_layer_name,
        "h_sram_size": quantize_config.h_sram_size,
        "h_max_groups": quantize_config.h_max_groups,
        "h_max_out_channel": quantize_config.h_max_out_channel,
        "h_max_kernel_size": quantize_config.h_max_kernel_size,
        "h_contain_weight": quantize_config.h_contain_weight,
    }

    return config_dict


@argument_filter_helper
def collect_quantization_config(filtered_args, extra=None):
    """add quantize_config item for hold quantization info"""
    unexpected_params = ["calibrate_dataset"]
    all_true_quantize_params = [
        k for k in ALL_ARGUMENTS_INFO["quantize"] if k not in unexpected_params
    ]
    filtered_args.quantize_config = AttributeDict()
    for k in all_true_quantize_params:
        filtered_args.quantize_config[k] = filtered_args[k]


@argument_filter_helper
def set_quantize_params_by_board(filtered_args, extra=None):
    if not hasattr(filtered_args, "board"):
        raise HHBException("There is no board args in filtered_args\n")
    if not hasattr(filtered_args, "quantize_config"):
        raise HHBException("Please execute 'collect_quantization_config' filter first.\n")

    if filtered_args.board == "anole":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            "num_bit_activation": 32,
            "dtype_input": "uint8",
            "dtype_weight": "uint8",
            "dtype_activation": "int32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_clip": True,
            "fuse_relu": False,
            "fuse_reshape_dense": True,
            "fuse_mul_add_to_conv": True,
            "channel_quantization": False,
            "broadcast_quantization": True,
        }
        if not filtered_args.quantize_config.quantization_scheme in ("unset", "uint8_asym"):
            raise HHBException("Anole only support uint8_asym\n")
        if filtered_args.quantize_config.channel_quantization:
            hhb_exit("Anole unsupport channel quantization.")
    elif filtered_args.board == "light":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            "num_bit_activation": 32,
            "dtype_input": "float32",
            "dtype_weight": "float32",
            "dtype_activation": "float32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            "broadcast_quantization": True,
            "h_contain_weight": False,
        }
        if extra.model_save == "save_only":
            # 1M
            new_values["h_sram_size"] = (
                2 ** 20
                if not filtered_args.quantize_config.h_sram_size
                else filtered_args.quantize_config.h_sram_size
            )
            new_values["h_max_groups"] = (
                16
                if not filtered_args.quantize_config.h_max_groups
                else filtered_args.quantize_config.h_max_groups
            )
        if filtered_args.quantize_config.channel_quantization:
            if extra.model_save == "save_and_run":
                hhb_exit("Light unsupport channel quantization on save_and_run mode.")
            if filtered_args.quantize_config.quantization_scheme != "int8_asym_w_sym":
                hhb_exit(
                    "Light channel quantization only support with int8_asym_w_sym quantization scheme."
                )
        if filtered_args.quantize_config.channel_quantization:
            new_values["calibrate_mode"] = "maxmin"
            new_values["weight_scale"] = "maxmin"
        if filtered_args.quantize_config.quantization_scheme == "unset":
            new_values["quantization_scheme"] = "int8_sym"
            filtered_args.quantize_config.quantization_scheme = "unset"
        elif filtered_args.quantize_config.quantization_scheme == "int8_sym":
            new_values["quantization_scheme"] = "int8_sym"
            filtered_args.quantize_config.quantization_scheme = "unset"
        elif filtered_args.quantize_config.quantization_scheme == "int8_asym_w_sym":
            new_values["quantization_scheme"] = "int8_asym_w_sym"
        elif filtered_args.quantize_config.quantization_scheme == "int8_original":
            new_values["quantization_scheme"] = "int8_original"
        elif filtered_args.quantize_config.quantization_scheme == "uint8_asym":
            new_values["quantization_scheme"] = "uint8_asym"
        elif filtered_args.quantize_config.quantization_scheme == "int8_asym":
            new_values["quantization_scheme"] = "int8_asym"
            new_values["calibrate_mode"] = "pow2"
            new_values["weight_scale"] = "pow2"
        elif filtered_args.quantize_config.quantization_scheme == "int16_sym":
            new_values["quantization_scheme"] = "int16_sym"
            new_values["num_bit_input"] = 16
            new_values["num_bit_weight"] = 16
            filtered_args.quantize_config.quantization_scheme = "unset"
        else:
            raise HHBException(
                f"Unsupport quantization scheme '{filtered_args.quantize_config.quantization_scheme}' on light\n"
            )
        if filtered_args.quantize_config.quantization_scheme in ("float16", "bfloat16"):
            raise HHBException("Light unsupport float16 or bfloat16\n")

    elif filtered_args.board == "hlight":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "float32",
            "dtype_weight": "float32",
            "dtype_activation": "float32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            "broadcast_quantization": True,
            "h_contain_weight": False,
        }
        if extra.model_save == "save_only":
            new_values["h_sram_size"] = (2 ** 20,)  # 1M
            new_values["h_max_groups"] = 16
        if filtered_args.quantize_config.channel_quantization:
            hhb_exit("HLight unsupport channel quantization.")
    elif filtered_args.board == "asp":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "int8",
            "dtype_weight": "int8",
            "dtype_activation": "int32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": True,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            "broadcast_quantization": True,
        }
        # ASP only support NHWC
        filtered_args.quantize_config.target_layout = "NHWC"
    elif filtered_args.board == "i805":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "uint8",
            "dtype_weight": "uint8",
            "dtype_activation": "int32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
        if filtered_args.quantize_config.channel_quantization:
            hhb_exit("i805 unsupport channel quantization.")
    elif filtered_args.board == "c906":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "float32",
            "dtype_weight": "float32",
            "dtype_activation": "float32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "c908":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "float32",
            "dtype_weight": "float32",
            "dtype_activation": "float32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }
    elif filtered_args.board == "ch8601":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "float32",
            "dtype_weight": "float32",
            "dtype_activation": "float32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": False,
            "channel_quantization": True,
            # "broadcast_quantization": False,
        }
        if not filtered_args.quantize_config.quantization_scheme in ("unset", "int8_sym"):
            raise HHBException("CH8601 only support int8_sym\n")
        if filtered_args.quantize_config.channel_quantization:
            hhb_exit("CH8601 unsupport channel quantization.")
    elif filtered_args.board == "dp1k":
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "float32",
            "dtype_weight": "float32",
            "dtype_activation": "float32",
            "calibrate_mode": "maxmin",
            "weight_quantized_type": "sym",
            "activate_quantized_type": "sym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": False,
            "channel_quantization": True,
            # "broadcast_quantization": False,
        }
        if not filtered_args.quantize_config.quantization_scheme in ("unset", "int8_sym"):
            raise HHBException("DP1K only support int8_sym\n")
    else:
        new_values = {
            "num_bit_input": 8,
            "num_bit_weight": 8,
            # "num_bit_activation": 32,
            "dtype_input": "uint8",
            "dtype_weight": "uint8",
            "dtype_activation": "int32",
            # "calibrate_mode": "maxmin",
            "weight_quantized_type": "asym",
            "activate_quantized_type": "asym",
            "weight_scale": "maxmin",
            "fuse_conv_relu": False,
            "fuse_relu": False,
            # "fuse_reshape": False,
            "fuse_mul_add_to_conv": True,
            # "channel_quantization": False,
            # "broadcast_quantization": False,
        }

    if filtered_args.board != "x86_ref":
        if (
            filtered_args.quantize_config.hybrid_quantization_scheme != "unset"
            or filtered_args.quantize_config.hybrid_layer_name is not None
        ):
            raise HHBException("Only 'x86_ref' target support for hybrid-quantization.\n")

    if filtered_args.quantize_config.channel_quantization:
        if filtered_args.quantize_config.broadcast_quantization:
            if (
                filtered_args.board not in ("light", "x86_ref")
                or filtered_args.quantize_config.quantization_scheme != "int8_asym_w_sym"
            ):
                raise HHBException(
                    "--broadcast-quantization can't be set while board is not light/x86_ref with "
                    "int8_asym_w_sym quantization and --channel-quantization is set.\n"
                )

    if filtered_args.quantize_config.quantization_scheme == "unset":
        if filtered_args.board == "unset":
            raise HHBException("Unset --board and --quantization-scheme.\n")
    elif filtered_args.quantize_config.quantization_scheme in ["int4_asym_w_sym"]:
        new_values["num_bit_input"] = 4
        new_values["num_bit_weight"] = 4
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int4"
        new_values["dtype_weight"] = "int4"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "sym"
        if filtered_args.target_layout == "NCHW":
            raise HHBException("Unsupport target_layout=NCHW for int4.\n")
    elif filtered_args.quantize_config.quantization_scheme == "uint8_asym":
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "uint8"
        new_values["dtype_weight"] = "uint8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "asym"
    elif filtered_args.quantize_config.quantization_scheme in ["int8_sym", "int8_original"]:
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int8"
        new_values["dtype_weight"] = "int8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme in ["int8_asym_w_sym"]:
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int8"
        new_values["dtype_weight"] = "int8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "sym"
        # new_values["channel_quantization"] = True
    elif filtered_args.quantize_config.quantization_scheme == "int8_asym":
        new_values["num_bit_input"] = 8
        new_values["num_bit_weight"] = 8
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int8"
        new_values["dtype_weight"] = "int8"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "asym"
        new_values["weight_quantized_type"] = "asym"
    elif filtered_args.quantize_config.quantization_scheme == "int16_sym":
        new_values["num_bit_input"] = 16
        new_values["num_bit_weight"] = 16
        new_values["num_bit_activation"] = 32
        new_values["dtype_input"] = "int16"
        new_values["dtype_weight"] = "int16"
        new_values["dtype_activation"] = "int32"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme == "float16":
        new_values["num_bit_input"] = 16
        new_values["num_bit_weight"] = 16
        new_values["num_bit_activation"] = 16
        new_values["dtype_input"] = "float16"
        new_values["dtype_weight"] = "float16"
        new_values["dtype_activation"] = "float16"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    elif filtered_args.quantize_config.quantization_scheme == "bfloat16":
        new_values["num_bit_input"] = 16
        new_values["num_bit_weight"] = 16
        new_values["num_bit_activation"] = 16
        new_values["dtype_input"] = "bfloat16"
        new_values["dtype_weight"] = "bfloat16"
        new_values["dtype_activation"] = "bfloat16"
        new_values["activate_quantized_type"] = "sym"
        new_values["weight_quantized_type"] = "sym"
    else:
        raise HHBException("Unsupport quantization scheme.\n")

    if filtered_args.quantize_config.num_bit_input != 0:
        new_values["num_bit_input"] = filtered_args.quantize_config.num_bit_input

    if filtered_args.quantize_config.num_bit_weight != 0:
        new_values["num_bit_weight"] = filtered_args.quantize_config.num_bit_weight

    if filtered_args.quantize_config.num_bit_activation != 0:
        new_values["num_bit_activation"] = filtered_args.quantize_config.num_bit_activation

    if filtered_args.quantize_config.dtype_input != "unset":
        new_values["dtype_input"] = filtered_args.quantize_config.dtype_input + str(
            filtered_args.quantize_config.num_bit_input
        )

    if filtered_args.quantize_config.dtype_weight != "unset":
        new_values["dtype_weight"] = filtered_args.quantize_config.dtype_weight + str(
            filtered_args.quantize_config.num_bit_weight
        )

    if filtered_args.quantize_config.dtype_activation != "unset":
        new_values["dtype_activation"] = filtered_args.quantize_config.dtype_activation + str(
            filtered_args.quantize_config.num_bit_activation
        )

    if filtered_args.quantize_config.weight_quantized_type != "unset":
        new_values["weight_quantized_type"] = filtered_args.quantize_config.weight_quantized_type

    if filtered_args.quantize_config.activate_quantized_type != "unset":
        new_values[
            "activate_quantized_type"
        ] = filtered_args.quantize_config.activate_quantized_type

    filtered_args.quantize_config.update(new_values)


def quantize_model(mod, params, qconfig, dataset=None, target="x86_ref"):
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
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.csinn.options": qconfig}):
        logger.debug("current quantize config:")
        logger.debug(qconfig)
        qfunc = qtz.quantize_hhb(mod, params, dataset=dataset, target=target)
    return qfunc
