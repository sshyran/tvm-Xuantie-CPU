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
# pylint: disable=consider-using-enumerate, no-else-return, unused-variable
# pylint: disable=inconsistent-return-statements, logging-not-lazy, arguments-differ
"""Find scales for quantization on the dataset."""
from __future__ import absolute_import
import logging
import numpy as np
import tvm
from tvm import relay
import multiprocessing as mp
from tvm.ir import IRModule
from tvm.ir.tensor_type import TensorType
from tvm.contrib.debugger import debug_runtime
from multiprocessing import Pool

from .quantize import current_qconfig as qconfig
from .asy_kl_divergence import _find_scale_by_asy_kl
from .kl_divergence import _find_scale_by_kl
from ..expr import Var, Call, TupleGetItem, Constant, Tuple, const
from ..frontend.common import infer_shape as _infer_shape


def _find_minmax(stats):
    min_value = np.min(stats)
    max_value = np.max(stats)
    return min_value, max_value


def _find_minmax_channel(data, axis=(1, 2, 3)):
    """By default, params are collected along the 1,2,3 dimensions."""
    mins = np.min(data, axis=axis)
    maxs = np.max(data, axis=axis)
    return mins, maxs


def _asym_quantize(max_value, min_value, bits=8):
    max_value = max(max_value, 0.0)
    min_value = min(min_value, 0.0)
    valid_range = 2 ** bits - 1
    scale = (max_value - min_value) / valid_range

    if scale == 0:
        scale = abs(max_value)
    if scale == 0:
        scale = 1.0
    zp = min(valid_range, max(0, round(0 - min_value / scale)))
    return scale, int(zp)


def _sym_quantize(max_value, bits=8):
    valid_range = 2 ** bits - 1
    scale = 2 * max_value / valid_range

    if scale == 0:
        scale = 1.0
    return scale, 0


def _asym_channel_quant_params(mins, maxs, bits=8):
    max_value = np.maximum(maxs, 0.0)
    min_value = np.minimum(mins, 0.0)
    valid_range = 2 ** bits - 1
    scales = (max_value - min_value) / valid_range
    scales[scales == 0] = np.abs(max_value[scales == 0])
    scales[scales == 0] = 1.0
    zps = np.minimum(valid_range, np.maximum(0, np.round(0 - min_value / scales)))
    return scales, zps.astype(np.int32)


def _sym_channel_quant_params(mins, maxs, bits=8):
    max_value = np.maximum(np.abs(maxs), np.abs(mins))
    valid_range = 2 ** bits - 1
    scales = 2 * max_value / valid_range
    scales[scales == 0] = np.abs(max_value[scales == 0])
    scales[scales == 0] = 1.0
    zps = np.zeros_like(scales)
    return scales, zps.astype(np.int32)


def get_weight_params_per_channel(weight_val):
    bits = qconfig().nbit_weight
    mins, maxs = _find_minmax_channel(weight_val)
    scales, zps = _asym_channel_quant_params(mins, maxs, bits)
    return scales, zps


def get_weight_params(weight_val):
    if qconfig().quantized_type == "asym":
        min_val, max_val = _find_minmax(weight_val)
        weight_scale, weight_zp = _asym_quantize(max_val, min_val, qconfig().nbit_weight)
    elif qconfig().quantized_type == "sym":
        min_val, max_val = _find_minmax(weight_val)
        max_val = max(abs(min_val), abs(max_val))
        weight_scale, weight_zp = _sym_quantize(max_val, qconfig().nbit_weight)
    return [weight_scale, weight_zp]


def get_quant_params(outs):
    datas = np.array(outs)
    use_kl = qconfig().calibrate_mode == "kl_divergence"
    quantized_type = qconfig().quantized_type

    if use_kl:
        if quantized_type == "asym":
            min_value, max_value = _find_scale_by_asy_kl(datas)
            scale, zero_point = _asym_quantize(max_value, min_value)
        elif quantized_type == "sym":
            max_value = _find_scale_by_kl(datas)
            scale, zero_point = _sym_quantize(max_value)
        else:
            logging.error(f"Don't support use kl divergence with quantized_type '{quantized_type}'")
    else:
        min_value, max_value = _find_minmax(datas)
        if quantized_type == "asym":
            scale, zero_point = _asym_quantize(max_value, min_value, qconfig().nbit_input)
        elif quantized_type == "sym":
            max_value = max(abs(max_value), abs(min_value))
            scale, zero_point = _sym_quantize(max_value, qconfig().nbit_input)
        else:
            logging.error(f"Don't support quantized_type '{quantized_type}'")

    return [scale, int(zero_point)]


def calibration(module, dataset, num_workers=32):
    """Calibration: normal scale for uint8 asymmetric quantization,
        only use max and min value, to calculate scale and zero point.

    Parameters
    ---------
    module: Module
        The original module.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: dict
        The nodes append quantization information

    """
    # from .quantize_hhb import _bind_params

    def infer_value(expr):
        mod = IRModule.from_expr(expr)
        exc = tvm.relay.create_executor("debug", mod=mod, ctx=tvm.cpu(), target="llvm")
        result = exc.evaluate()
        return result()

    class GetLayerCount(relay.ExprVisitor):
        def __init__(self):
            self.memo_map = {}
            self.layer_count = {}

        def enter_dict(self, hash_call):
            if hash_call in self.layer_count:
                self.layer_count[hash_call] += 1
            else:
                self.layer_count[hash_call] = 0

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            for i, arg in enumerate(call.args):
                if isinstance(arg, Tuple):
                    len_tuple = len(arg)
                    for j in range(len_tuple):
                        self.enter_dict(hash(arg.fields[j]))
                else:
                    self.enter_dict(hash(arg))

    class Calibration(relay.ExprVisitor):
        def __init__(self, inputs, pool, layer_count):
            self.memo_map = {}
            self.outs_map = {}
            self.quant_params = {}
            self.inputs = inputs
            self.input_count = len(self.inputs)
            self.pool = pool
            self.layer_count = layer_count

        def clear_mem(self, call):
            hash_call = hash(call)
            if self.layer_count[hash_call] == 0:
                del self.outs_map[call]
                self.layer_count[hash_call] -= 1
            elif self.layer_count[hash_call] > 0:
                self.layer_count[hash_call] -= 1

        def _get_quant_params(self, call, data, kind):
            """
            kind:
                0: weights
                1: outs
            """
            if kind:
                s = pool.apply_async(get_quant_params, args=(data,))
            else:
                s = pool.apply_async(get_weight_params, args=(data.data.asnumpy(),))

            self.quant_params[call].append(s)

        def visit_var(self, var):
            quant_data = []
            new_args = []
            hash_call = hash(var)
            self.quant_params[hash_call] = []
            for in_data in self.inputs:
                data = in_data[var.name_hint]
                new_args.append(const(data))
                quant_data.append(data)
            self.outs_map[var] = new_args
            self._get_quant_params(hash_call, quant_data, 1)

        def visit_call(self, call):
            assert call.op.name != "nn.batch_normal"
            _ = [self.visit(arg) for arg in call.args]
            new_args = [[] for arg in call.args]
            hash_call = hash(call)
            self.quant_params[hash_call] = []
            for i, arg in enumerate(call.args):
                quant_data = []

                if isinstance(arg, Constant):
                    new_args[i] = [arg for j in range(self.input_count)]
                    self._get_quant_params(hash_call, arg, 0)

                elif isinstance(arg, Call) or isinstance(arg, TupleGetItem) or isinstance(arg, Var):
                    if arg in self.outs_map:
                        arg_val_list = self.outs_map[arg]
                        self.clear_mem(arg)
                    else:
                        raise "can't find input."
                    new_args[i] = arg_val_list
                    self.quant_params[hash_call].append(self.quant_params[hash(arg)][-1])

                elif isinstance(arg, Tuple):
                    len_tuple = len(arg)
                    field_val_lists = [[] for x in range(len_tuple)]
                    for j in range(len_tuple):
                        if arg.fields[j] in self.outs_map:
                            tuple_val_list = self.outs_map[arg.fields[j]]
                            self.clear_mem(arg.fields[j])
                        else:
                            raise "can't find input."
                        field_val_lists[j] = tuple_val_list
                        self.quant_params[hash_call].append(
                            self.quant_params[hash(arg.fields[j])][-1]
                        )
                    for j in range(self.input_count):
                        new_tuple = Tuple([x[j] for x in field_val_lists])
                        new_args[i].append(new_tuple)

            self.outs_map[call] = []
            quant_data = []
            mo_flag = False
            for i in range(self.input_count):
                args = [x[i] for x in new_args]
                new_call = Call(call.op, args, call.attrs)
                value = infer_value(new_call)
                if isinstance(value, tvm.nd.NDArray):
                    self.outs_map[call].append(const(value))
                    quant_data.append(value.asnumpy())
                else:
                    mo_flag = True
                    self.outs_map[call].append(value)
                    quant_data = [[] for _ in value]
                    for j, x in enumerate(value):
                        data = x.asnumpy()
                        quant_data[j].append(data)
            if mo_flag:
                for data in quant_data:
                    self._get_quant_params(hash_call, data, 1)
            else:
                self._get_quant_params(hash_call, quant_data, 1)

        def visit_tuple_getitem(self, t):
            self.visit(t.tuple_value)
            hash_call = hash(t)
            if t.tuple_value in self.outs_map:
                tuple_value = self.outs_map[t.tuple_value]
            else:
                raise "tuple getitem not find input."
            self.outs_map[t] = []
            quant_data = []
            for i in range(self.input_count):
                data = tuple_value[i][t.index]
                self.outs_map[t].append(const(data))
                quant_data.append(data.asnumpy())
            self.quant_params[hash_call] = []
            self._get_quant_params(hash_call, quant_data, 1)

    optimizer = GetLayerCount()
    optimizer.visit(module["main"])
    layer_count = optimizer.layer_count

    with Pool(num_workers) as pool:
        get_out = Calibration(dataset, pool, layer_count)
        get_out.visit(module["main"])
        pool.close()
        pool.join()

    quant_params = {}
    for key in get_out.quant_params:
        quant_params[key] = []
        for s in get_out.quant_params[key]:
            quant_params[key].append(s.get())
    del get_out.outs_map

    return quant_params


class csi_op:
    """ All qnn csi ops """

    def __init__(self):
        self.conv_handle = {
            "qnn.csi.conv2d": relay.qnn.op.csi_conv2d,
            "qnn.csi.conv2d_relu": relay.qnn.op.csi_conv2d_relu,
            "qnn.csi.conv2d_relu6": relay.qnn.op.csi_conv2d_relu6,
            "qnn.csi.deconv2d": relay.qnn.op.csi_deconv2d,
            "qnn.csi.deconv3d": relay.qnn.op.csi_deconv3d,
            "qnn.csi.conv3d": relay.qnn.op.csi_conv3d,
            "qnn.csi.conv2d_channel": relay.qnn.op.csi_conv2d_channel,
            "qnn.csi.conv2d_relu_channel": relay.qnn.op.csi_conv2d_relu_channel,
            "qnn.csi.conv2d_relu6_channel": relay.qnn.op.csi_conv2d_relu6_channel,
        }

        self.siso_handle = {
            "qnn.csi.maxpool": relay.qnn.op.csi_max_pool,
            "qnn.csi.softmax": relay.qnn.op.csi_softmax,
            "qnn.csi.log_softmax": relay.qnn.op.csi_log_softmax,
            "qnn.csi.reshape": relay.qnn.op.csi_reshape,
            "qnn.csi.relu": relay.qnn.op.csi_relu,
            "qnn.csi.relu6": relay.qnn.op.csi_relu6,
            "qnn.csi.nn_init": relay.qnn.op.csinn_init,
            "qnn.csi.nn_deinit": relay.qnn.op.csinn_deinit,
            "qnn.csi.avg_pool": relay.qnn.op.csi_avg_pool,
            "qnn.csi.avg_pool3d": relay.qnn.op.csi_avg_pool3d,
            "qnn.csi.max_pool3d": relay.qnn.op.csi_max_pool3d,
            "qnn.csi.global_avgpool": relay.qnn.op.csi_global_avgpool,
            "qnn.csi.global_maxpool": relay.qnn.op.csi_global_maxpool,
            "qnn.csi.lrn": relay.qnn.op.csi_lrn,
            "qnn.csi.leaky_relu": relay.qnn.op.csi_leaky_relu,
            "qnn.csi.upsampling": relay.qnn.op.csi_upsampling,
            "qnn.csi.transpose": relay.qnn.op.csi_transpose,
            "qnn.csi.flatten": relay.qnn.op.csi_flatten,
            "qnn.csi.sigmoid": relay.qnn.op.csi_sigmoid,
            "qnn.csi.squeeze": relay.qnn.op.csi_squeeze,
            "qnn.csi.pad": relay.qnn.op.csi_pad,
            "qnn.csi.mean": relay.qnn.op.csi_mean,
            "qnn.csi.prod": relay.qnn.op.csi_prod,
            "qnn.csi.max": relay.qnn.op.csi_max,
            "qnn.csi.min": relay.qnn.op.csi_min,
            "qnn.csi.argmax": relay.qnn.op.csi_argmax,
            "qnn.csi.argmin": relay.qnn.op.csi_argmin,
            "qnn.csi.variance": relay.qnn.op.csi_variance,
            "qnn.csi.strided_slice": relay.qnn.op.csi_strided_slice,
            "qnn.csi.sin": relay.qnn.op.csi_sin,
            "qnn.csi.cos": relay.qnn.op.csi_cos,
            "qnn.csi.tan": relay.qnn.op.csi_tan,
            "qnn.csi.asin": relay.qnn.op.csi_asin,
            "qnn.csi.acos": relay.qnn.op.csi_acos,
            "qnn.csi.atan": relay.qnn.op.csi_atan,
            "qnn.csi.sinh": relay.qnn.op.csi_sinh,
            "qnn.csi.cosh": relay.qnn.op.csi_cosh,
            "qnn.csi.tanh": relay.qnn.op.csi_tanh,
            "qnn.csi.asinh": relay.qnn.op.csi_asinh,
            "qnn.csi.acosh": relay.qnn.op.csi_acosh,
            "qnn.csi.atanh": relay.qnn.op.csi_atanh,
            "qnn.csi.exp": relay.qnn.op.csi_exp,
            "qnn.csi.abs": relay.qnn.op.csi_abs,
            "qnn.csi.expand_dims": relay.qnn.op.csi_expand_dims,
            "qnn.csi.broadcast_to": relay.qnn.op.csi_broadcast_to,
            "qnn.csi.cast": relay.qnn.op.csi_cast,
            "qnn.csi.ceil": relay.qnn.op.csi_ceil,
            "qnn.csi.floor": relay.qnn.op.csi_ceil,
            "qnn.csi.clip": relay.qnn.op.csi_clip,
            "qnn.csi.round": relay.qnn.op.csi_round,
            "qnn.csi.depth_to_space": relay.qnn.op.csi_depth_to_space,
            "qnn.csi.space_to_depth": relay.qnn.op.csi_space_to_depth,
            "qnn.csi.sqrt": relay.qnn.op.csi_sqrt,
            "qnn.csi.sum": relay.qnn.op.csi_sum,
            "qnn.csi.log": relay.qnn.op.csi_log,
            "qnn.csi.negative": relay.qnn.op.csi_negative,
            "qnn.csi.reverse": relay.qnn.op.csi_reverse,
            "qnn.csi.sign": relay.qnn.op.csi_sign,
            "qnn.csi.topk": relay.qnn.op.csi_topk,
            "qnn.csi.tile": relay.qnn.op.csi_tile,
            "qnn.csi.maxpool2d_with_argmax": relay.qnn.op.csi_max_pool2d_with_argmax,
            "qnn.csi.maxpool2d_locat": relay.qnn.op.csi_max_pool2d_locat,
        }

        self.diso_handle = {
            "qnn.csi.add": relay.qnn.op.csi_add,
            "qnn.csi.subtract": relay.qnn.op.csi_subtract,
            "qnn.csi.bias_add": relay.qnn.op.csi_bias_add,
            "qnn.csi.mul": relay.qnn.op.csi_mul,
            "qnn.csi.div": relay.qnn.op.csi_div,
            "qnn.csi.power": relay.qnn.op.csi_power,
            "qnn.csi.minimum": relay.qnn.op.csi_minimum,
            "qnn.csi.maximum": relay.qnn.op.csi_maximum,
            "qnn.csi.floor_div": relay.qnn.op.csi_floor_div,
            "qnn.csi.floor_mod": relay.qnn.op.csi_floor_mod,
            "qnn.csi.left_shift": relay.qnn.op.csi_left_shift,
            "qnn.csi.right_shift": relay.qnn.op.csi_right_shift,
            "qnn.csi.mod": relay.qnn.op.csi_mod,
            "qnn.csi.segment_max": relay.qnn.op.csi_segment_max,
            "qnn.csi.segment_min": relay.qnn.op.csi_segment_min,
            "qnn.csi.segment_mean": relay.qnn.op.csi_segment_mean,
            "qnn.csi.segment_sum": relay.qnn.op.csi_segment_sum,
            "qnn.csi.segment_prod": relay.qnn.op.csi_segment_prod,
        }

        self.other_handle = {
            "qnn.csi.dense": relay.qnn.op.csi_dense,
            "qnn.csi.concatenate": relay.qnn.op.csi_concatenate,
            "qnn.csi.bn": relay.qnn.op.csi_batch_norm,
            "qnn.csi.proposal": relay.qnn.op.csi_proposal,
            "qnn.csi.psroipooling": relay.qnn.op.csi_psroipooling,
            "qnn.csi.roipooling": relay.qnn.op.csi_roipooling,
            "qnn.csi.prelu": relay.qnn.op.csi_prelu,
            "qnn.csi.unpooling": relay.qnn.op.csi_unpooling,
            "qnn.csi.split": relay.qnn.op.csi_split,
            "qnn.csi.dilation2d": relay.qnn.op.csi_dilation2d,
            "qnn.csi.full": relay.qnn.op.csi_dilation2d,
            "qnn.csi.crop_resize": relay.qnn.op.csi_crop_resize,
            "qnn.csi.take": relay.qnn.op.csi_take,
        }

        self.all_handle = self._get_all_handle()

    def conv_op(self, name):
        return name in self.conv_handle

    def conv_handler(self, name):
        return self.conv_handle[name]

    def siso_op(self, name):
        return name in self.siso_handle

    def siso_handler(self, name):
        return self.siso_handle[name]

    def diso_op(self, name):
        return name in self.diso_handle

    def diso_handler(self, name):
        return self.diso_handle[name]

    def _get_all_handle(self):
        res = dict()
        res.update(**self.conv_handle, **self.siso_handle, **self.diso_handle, **self.other_handle)
        return res


def convert_to_csi_qnn(mod, quant_params):
    """The convert_to_csi_qnn convert add ops to qnn.csi.* ops.

    Returns
    -------
    ret: Function
        The module pass function.
    """

    # def wrapped_func(mod, ctx): # pylint: disable=unused-argument

    class ConvertToCSIMutator(relay.ExprMutator):
        """ Convert tvm ops into csi ops """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            cts = call.attrs
            hash_call = hash(call)
            scales = [x[0] for x in quant_params[hash_call]]
            zps = [x[1] for x in quant_params[hash_call]]

            if call.op.name == "nn.conv2d":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0)

                new_call = relay.qnn.op.csi_conv2d(
                    data,
                    weight,
                    bias,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.out_dtype,
                )
            elif call.op.name == "nn.conv3d":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0)
                new_call = relay.qnn.op.csi_conv3d(
                    data,
                    weight,
                    bias,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.out_dtype,
                )
            elif call.op.name == "image.dilation2d":
                data = op_args[0]
                weight = op_args[1]
                new_call = relay.qnn.op.csi_dilation2d(
                    data,
                    weight,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    cts.strides,
                    cts.padding,
                    cts.dilations,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_dtype,
                )
            elif call.op.name == "nn.dense":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0)
                new_call = relay.qnn.op.csi_dense(
                    data,
                    weight,
                    bias,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    cts.units,
                    "float32",
                )
            elif call.op.name == "nn.bias_add":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_bias_add(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "nn.relu":
                data = op_args[0]
                new_call = relay.qnn.op.csi_relu(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "sin":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sin(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "cos":
                data = op_args[0]
                new_call = relay.qnn.op.csi_cos(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "tan":
                data = op_args[0]
                new_call = relay.qnn.op.csi_tan(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "asin":
                data = op_args[0]
                new_call = relay.qnn.op.csi_asin(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "acos":
                data = op_args[0]
                new_call = relay.qnn.op.csi_acos(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "atan":
                data = op_args[0]
                new_call = relay.qnn.op.csi_atan(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "sinh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sinh(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "cosh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_cosh(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "tanh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_tanh(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "asinh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_asinh(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "acosh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_acosh(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "atanh":
                data = op_args[0]
                new_call = relay.qnn.op.csi_atanh(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "vision.segment_max":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_max(
                    data, segment_ids, scales[0], zps[0], scales[2], zps[2], cts.length, "float32"
                )
            elif call.op.name == "vision.segment_min":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_min(
                    data, segment_ids, scales[0], zps[0], scales[2], zps[2], cts.length, "float32"
                )
            elif call.op.name == "vision.segment_mean":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_mean(
                    data, segment_ids, scales[0], zps[0], scales[2], zps[2], cts.length, "float32"
                )
            elif call.op.name == "vision.segment_prod":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_prod(
                    data, segment_ids, scales[0], zps[0], scales[2], zps[2], cts.length, "float32"
                )
            elif call.op.name == "vision.segment_sum":
                data = op_args[0]
                segment_ids = op_args[1]
                new_call = relay.qnn.op.csi_segment_sum(
                    data, segment_ids, scales[0], zps[0], scales[2], zps[2], cts.length, "float32"
                )
            elif call.op.name == "nn.batch_norm":
                data = op_args[0]
                gamma = op_args[1]
                beta = op_args[2]
                moving_mean = op_args[3]
                moving_var = op_args[4]
                new_call = relay.qnn.op.csi_batch_norm(
                    data,
                    gamma,
                    beta,
                    moving_mean,
                    moving_var,
                    scales[0],
                    zps[0],
                    scales[-1],
                    zps[-1],
                    cts.axis,
                    cts.epsilon,
                    cts.center,
                    cts.scale,
                )
            elif call.op.name == "nn.avg_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_avg_pool(
                    data,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.count_include_pad,
                    cts.layout,
                )
            elif call.op.name == "nn.avg_pool3d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_avg_pool3d(
                    data,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.count_include_pad,
                    cts.layout,
                )
            elif call.op.name == "nn.max_pool3d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_max_pool3d(
                    data,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.layout,
                )
            elif call.op.name == "nn.global_avg_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_global_avgpool(
                    data, cts.layout, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "nn.global_max_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_global_maxpool(
                    data, cts.layout, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "nn.max_pool2d":
                data = op_args[0]
                new_call = relay.qnn.op.csi_max_pool(
                    data,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.layout,
                )
            elif call.op.name == "reshape":
                data = op_args[0]
                new_call = relay.qnn.op.csi_reshape(
                    data, cts.newshape, cts.reverse, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "squeeze":
                data = op_args[0]
                new_call = relay.qnn.op.csi_squeeze(
                    data, cts.axis, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "nn.softmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_softmax(
                    data, cts.axis, scales[0], zps[0], scales[1], zps[1], "uint8"
                )
            elif call.op.name == "reverse":
                data = op_args[0]
                axis = cts.axis.value
                new_call = relay.qnn.op.csi_reverse(
                    data, axis, scales[0], zps[0], scales[1], zps[1], "uint8"
                )
            elif call.op.name == "negative":
                data = op_args[0]
                new_call = relay.qnn.op.csi_negative(
                    data, scales[0], zps[0], scales[1], zps[1], "uint8"
                )
            elif call.op.name == "nn.log_softmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_log_softmax(
                    data, cts.axis, scales[0], zps[0], scales[1], zps[1], "uint8"
                )
            elif call.op.name == "nn.lrn":
                data = op_args[0]
                new_call = relay.qnn.op.csi_lrn(
                    data,
                    cts.size,
                    cts.axis,
                    cts.alpha,
                    cts.beta,
                    cts.bias,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "concatenate":
                data = op_args[0]
                input_scales = scales[:-1]
                input_zps = zps[:-1]
                new_call = relay.qnn.op.csi_concatenate(
                    data, input_scales, input_zps, scales[-1], zps[-1], cts.axis
                )
            elif call.op.name == "add":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_add(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "subtract":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_subtract(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "nn.leaky_relu":
                data = op_args[0]
                new_call = relay.qnn.op.csi_leaky_relu(
                    data, cts.alpha, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "nn.upsampling":
                data = op_args[0]
                new_call = relay.qnn.op.csi_upsampling(
                    data,
                    cts.scale_h,
                    cts.scale_w,
                    cts.align_corners,
                    cts.method,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.layout,
                )
            elif call.op.name == "image.resize":
                data = op_args[0]
                origin_shape = (call.type_args)[0].concrete_shape
                assert len(origin_shape) == 4, "Only support 4-dim shape of image.resize"
                scale_h = int(cts.size[0]) / origin_shape[2]
                scale_w = int(cts.size[1]) / origin_shape[3]
                if cts.coordinate_transformation_mode == "asymmetric":
                    align_corners = False
                else:
                    align_corners = True
                new_call = relay.qnn.op.csi_upsampling(
                    data,
                    scale_h,
                    scale_w,
                    align_corners,
                    cts.method,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.layout,
                )

            elif call.op.name == "nn.conv2d_transpose":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0)
                new_call = relay.qnn.op.csi_deconv2d(
                    data,
                    weight,
                    bias,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.output_padding,
                    "float32",
                )
            elif call.op.name == "nn.conv3d_transpose":
                data = op_args[0]
                weight = op_args[1]
                bias = relay.expr.const(0)
                new_call = relay.qnn.op.csi_deconv3d(
                    data,
                    weight,
                    bias,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    cts.strides,
                    cts.padding,
                    cts.dilation,
                    cts.groups,
                    cts.channels,
                    cts.kernel_size,
                    cts.data_layout,
                    cts.kernel_layout,
                    cts.out_layout,
                    cts.output_padding,
                    "float32",
                )
            elif call.op.name == "transpose":
                data = op_args[0]
                new_call = relay.qnn.op.csi_transpose(
                    data, cts.axes, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "nn.batch_flatten":
                data = op_args[0]
                new_call = relay.qnn.op.csi_flatten(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "sigmoid":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sigmoid(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "vision.proposal":
                cls_prob = op_args[0]
                bbox_pred = op_args[1]
                im_info = op_args[2]
                new_call = relay.qnn.op.csi_proposal(
                    cls_prob,
                    bbox_pred,
                    im_info,
                    cts.scales,
                    cts.ratios,
                    cts.feature_stride,
                    cts.threshold,
                    cts.rpn_pre_nms_top_n,
                    cts.rpn_post_nms_top_n,
                    cts.rpn_min_size,
                    cts.iou_loss,
                    [scales[0], scales[1]],
                    [zps[0], zps[1]],
                    scales[-1],
                    zps[-1],
                    "float32",
                )
            elif call.op.name == "vision.psroipooling":
                cls_prob = op_args[0]
                roi = op_args[1]
                new_call = relay.qnn.op.csi_psroipooling(
                    cls_prob,
                    roi,
                    cts.spatial_scale,
                    cts.output_dim,
                    cts.group_size,
                    [scales[0], scales[1]],
                    [zps[0], zps[1]],
                    scales[2],
                    zps[2],
                    "float32",
                )
            elif call.op.name == "vision.roi_pool":
                data = op_args[0]
                roi = op_args[1]
                new_call = relay.qnn.op.csi_roipooling(
                    data,
                    roi,
                    cts.pooled_size,
                    cts.spatial_scale,
                    [scales[0], scales[1]],
                    [zps[0], zps[1]],
                    scales[2],
                    zps[2],
                    "float32",
                )
            elif call.op.name == "multiply":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_mul(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "divide":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_div(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "power":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_power(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "mod":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_mod(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "nn.prelu":
                data = op_args[0]
                alpha = op_args[1]
                new_call = relay.qnn.op.csi_prelu(
                    data,
                    alpha,
                    cts.axis,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    scales[2],
                    zps[2],
                    "float32",
                )
            elif call.op.name == "nn.max_pool2d_with_argmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_max_pool2d_with_argmax(
                    data,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    cts.layout,
                )
            elif call.op.name == "mean":
                data = op_args[0]
                new_call = relay.qnn.op.csi_mean(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "prod":
                data = op_args[0]
                new_call = relay.qnn.op.csi_prod(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "max":
                data = op_args[0]
                new_call = relay.qnn.op.csi_max(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "min":
                data = op_args[0]
                new_call = relay.qnn.op.csi_min(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "sum":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sum(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "argmax":
                data = op_args[0]
                new_call = relay.qnn.op.csi_argmax(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    1.0,
                    0,
                    "float32",
                )
            elif call.op.name == "argmin":
                data = op_args[0]
                new_call = relay.qnn.op.csi_argmin(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "nn.pad":
                data = op_args[0]
                new_call = relay.qnn.op.csi_pad(
                    data,
                    cts.pad_width,
                    cts.pad_value,
                    cts.pad_mode,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "clip":
                data = op_args[0]
                if cts.a_min == 0 and cts.a_max == 6:
                    new_call = relay.qnn.op.csi_relu6(
                        data, scales[0], zps[0], scales[1], zps[1], "float32"
                    )
                else:
                    new_call = relay.qnn.op.csi_clip(
                        data, scales[0], zps[0], scales[1], zps[1], cts.a_min, cts.a_max, "float32"
                    )

            elif call.op.name == "vision.max_pool2d_location":
                data = op_args[0]
                new_call = relay.qnn.op.csi_max_pool2d_locat(
                    data,
                    scales[0],
                    zps[0],
                    1,
                    0,
                    cts.strides,
                    cts.padding,
                    cts.pool_size,
                    cts.ceil_mode,
                    "float32",
                    cts.layout,
                )
            elif call.op.name == "vision.unpooling":
                data = op_args[0]
                mask = op_args[1]
                scale = [cts.scale_h, cts.scale_w]
                out_padding = [cts.pad_out_h, cts.pad_out_w]
                new_call = relay.qnn.op.csi_unpooling(
                    data,
                    mask,
                    scale,
                    out_padding,
                    scales[0],
                    zps[0],
                    scales[-1],
                    zps[-1],
                    "float32",
                    cts.layout,
                )
            elif call.op.name == "strided_slice":
                data = op_args[0]
                if len(cts.strides) == 0:
                    strides = [1] * len(cts.begin)
                else:
                    strides = cts.strides
                begin = [int(i) for i in cts.begin]
                end = [int(i) for i in cts.end]
                if cts.slice_mode == "size":
                    end = list(map(lambda x: x[0] + x[1], zip(begin, end)))

                new_call = relay.qnn.op.csi_strided_slice(
                    data, begin, end, strides, scales[0], zps[0], scales[-1], zps[-1], "float32"
                )
            elif call.op.name == "split":
                data = op_args[0]
                o_scales = scales[1:]
                o_zp = zps[1:]
                new_call = relay.qnn.op.csi_split(
                    data,
                    cts.indices_or_sections,
                    cts.axis,
                    scales[0],
                    zps[0],
                    o_scales,
                    o_zp,
                    "float32",
                )
            elif call.op.name == "variance":
                data = op_args[0]
                new_call = relay.qnn.op.csi_variance(
                    data,
                    cts.axis,
                    cts.keepdims,
                    cts.exclude,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    "float32",
                )
            elif call.op.name == "exp":
                data = op_args[0]
                new_call = relay.qnn.op.csi_exp(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "log":
                data = op_args[0]
                new_call = relay.qnn.op.csi_log(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "abs":
                data = op_args[0]
                new_call = relay.qnn.op.csi_abs(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "expand_dims":
                data = op_args[0]
                new_call = relay.qnn.op.csi_expand_dims(
                    data, cts.axis, cts.num_newaxis, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "broadcast_to":
                data = op_args[0]
                new_call = relay.qnn.op.csi_broadcast_to(
                    data, cts.shape, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "cast":
                data = op_args[0]
                new_call = relay.qnn.op.csi_cast(
                    data, scales[0], zps[0], scales[1], zps[1], cts.dtype
                )
            elif call.op.name == "ceil":
                data = op_args[0]
                new_call = relay.qnn.op.csi_ceil(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "floor":
                data = op_args[0]
                new_call = relay.qnn.op.csi_floor(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "round":
                data = op_args[0]
                new_call = relay.qnn.op.csi_round(
                    data, scales[0], zps[0], scales[1], zps[1], "float32"
                )
            elif call.op.name == "minimum":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_minimum(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "maximum":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_maximum(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "right_shift":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_right_shift(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "left_shift":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_left_shift(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "floor_divide":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_floor_div(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "floor_mod":
                lhs = op_args[0]
                rhs = op_args[1]
                new_call = relay.qnn.op.csi_floor_mod(
                    lhs, rhs, scales[0], zps[0], scales[1], zps[1], scales[2], zps[2]
                )
            elif call.op.name == "image.crop_and_resize":
                data = op_args[0]
                boxes = op_args[1]
                box_indices = op_args[2]
                new_call = relay.qnn.op.csi_crop_resize(
                    data, boxes, box_indices, scales[0], zps[0], scales[-1], zps[-1], **cts
                )
            elif call.op.name == "nn.depth_to_space":
                data = op_args[0]
                new_call = relay.qnn.op.csi_depth_to_space(
                    data, scales[0], zps[0], scales[-1], zps[-1], out_dtype="float32", **cts
                )

            elif call.op.name == "nn.space_to_depth":
                data = op_args[0]
                new_call = relay.qnn.op.csi_space_to_depth(
                    data, scales[0], zps[0], scales[-1], zps[-1], out_dtype="float32", **cts
                )
            elif call.op.name == "erf":
                data = op_args[0]
                new_call = relay.qnn.op.csi_erf(
                    data, scales[0], zps[0], scales[1], zps[1], out_dtype="float32"
                )
            elif call.op.name == "sqrt":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sqrt(
                    data, scales[0], zps[0], scales[1], zps[1], out_dtype="float32"
                )
            elif call.op.name == "sign":
                data = op_args[0]
                new_call = relay.qnn.op.csi_sign(
                    data, scales[0], zps[0], scales[1], zps[1], out_dtype="float32"
                )
            elif call.op.name == "full":
                data = op_args[0]
                shape = op_args[1]
                new_call = relay.qnn.op.csi_full(data, shape, 1, 0, 1, 0, "float32")
            elif call.op.name == "take":
                data = op_args[0]
                indices = op_args[1]
                axis = cts.axis.value
                new_call = relay.qnn.op.csi_take(
                    data, indices, scales[0], zps[0], scales[-1], zps[-1], axis, cts.mode, "float32"
                )
            elif call.op.name == "tile":
                data = op_args[0]
                new_call = relay.qnn.op.csi_tile(
                    data, scales[0], zps[0], scales[1], zps[1], cts.reps, "float32"
                )
            elif call.op.name == "topk":
                data = op_args[0]
                k = cts.k.value
                new_call = relay.qnn.op.csi_topk(
                    data,
                    scales[0],
                    zps[0],
                    scales[1],
                    zps[1],
                    k,
                    cts.axis,
                    cts.ret_type,
                    cts.is_ascend,
                    cts.dtype,
                    "float32",
                )
            elif call.op.name == "unravel_index":
                data = op_args[0]
                shape = op_args[1]
                new_call = relay.qnn.op.csi_unravel_index(data, shape, 1, 0, 1, 0, "float32")
            else:
                raise ValueError("Cannot convert op:", call.op.name)

            return new_call

        def visit_tuple_getitem(self, op):
            tuple_value = self.visit(op.tuple_value)
            if not tuple_value.same_as(op.tuple_value):
                if tuple_value.op.name == "qnn.csi.bn":
                    return tuple_value
                return TupleGetItem(tuple_value, op.index)
            return tuple_value

    class RmDropoutMutator(relay.ExprMutator):
        def visit_tuple_getitem(self, item):
            call = self.visit(item.tuple_value)
            if call.op.name == "nn.dropout":
                pre_call = call.args[0]
                return pre_call
            elif not call.same_as(item.tuple_value):
                return TupleGetItem(call, item.index)
            else:
                return call

    # mod['main'] = RmDropoutMutator().visit(mod['main'])
    mod["main"] = ConvertToCSIMutator().visit(mod["main"])

    return mod


def _qnn_attrs(attrs):
    ret = {}
    for i in dir(attrs):
        if not i.startswith("_") and i not in ["handle", "same_as"]:
            ret[i] = getattr(attrs, i)
    return ret


def _get_csi_op(name):
    return csi_op().all_handle[name]


def fuse_layer(mod, fuse_relu=False):
    """remove unnecessary layer to speed up module.

    Returns
    -------
    ret: Function
        The module pass function.
    """

    # def wrapped_func(mod, ctx): # pylint: disable=unused-argument

    class FuseBiasMutator(relay.ExprMutator):
        """ Fuse bias helper class """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name == "qnn.csi.bias_add":
                pre_call = op_args[0]
                if not isinstance(pre_call, Call):
                    return Call(call.op, op_args, call.attrs)
                new_attrs = _qnn_attrs(pre_call.attrs)
                if pre_call.op.name == "qnn.csi.conv2d":
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = op_args[1]
                    new_attrs["output_scale"] = call.attrs.output_scale
                    new_attrs["output_zero_point"] = call.attrs.output_zero_point
                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_attrs)
                elif pre_call.op.name == "qnn.csi.dense":
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = op_args[1]
                    new_attrs["output_scale"] = call.attrs.output_scale
                    new_attrs["output_zero_point"] = call.attrs.output_zero_point
                    new_call = relay.qnn.op.csi_dense(data, weight, bias, **new_attrs)
                elif pre_call.op.name == "qnn.csi.deconv2d":
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = op_args[1]
                    new_attrs["output_scale"] = call.attrs.output_scale
                    new_attrs["output_zero_point"] = call.attrs.output_zero_point
                    new_call = relay.qnn.op.csi_deconv2d(data, weight, bias, **new_attrs)
                else:
                    new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)

            return new_call

    class FuseBnMutator(relay.ExprMutator):
        """ Fuse batch norm helper class """

        def __updata_params(
            self, weight_val, bias_val, call_attrs, new_attrs, mean, var, gama, beta, eps
        ):
            var[var < 0] = 0
            new_weight_val = gama * weight_val / ((var + eps) ** (0.5))
            new_bias_val = gama.reshape(-1) * (bias_val.reshape(-1) - mean.reshape(-1)) / (
                (var.reshape(-1) + eps) ** (0.5)
            ) + beta.reshape(-1)

            # recalculate scale and zero_point
            max_val, min_val = _find_minmax(new_weight_val)
            kernel_scale, kernel_zp = _asym_quantize(max_val, min_val)

            new_attrs["kernel_scale"] = kernel_scale
            new_attrs["kernel_zero_point"] = int(kernel_zp)
            new_attrs["output_scale"] = call_attrs.output_scale
            new_attrs["output_zero_point"] = call_attrs.output_zero_point

            return (
                new_weight_val.astype(np.float32),
                new_bias_val.reshape(-1).astype(np.float32),
                new_attrs,
            )

        def visit_call(self, call):
            new_fn = self.visit(call.op)
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "qnn.csi.bn":
                pre_call = op_args[0]
                new_attrs = _qnn_attrs(pre_call.attrs)
                if pre_call.op.name == "qnn.csi.conv2d":
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = pre_call.args[2]

                    weight_val = weight.data.asnumpy()
                    bias_val = bias.data.asnumpy()
                    if len(bias_val.shape) == 4:
                        logging.error(
                            "Can't fuse bn for bias shape: [%d,%d,%d,%d]"
                            % (
                                bias_val.shape[0],
                                bias_val.shape[1],
                                bias_val.shape[2],
                                bias_val.shape[3],
                            )
                        )
                    else:
                        bias_val = bias_val.reshape(-1, 1, 1, 1)
                    eps = call.attrs.epsilon
                    gama = op_args[1].data.asnumpy().reshape(-1, 1, 1, 1)
                    beta = op_args[2].data.asnumpy().reshape(-1, 1, 1, 1)
                    mean = op_args[3].data.asnumpy().reshape(-1, 1, 1, 1)
                    var = op_args[4].data.asnumpy().reshape(-1, 1, 1, 1)

                    new_weight_val, new_bias_val, new_attrs = self.__updata_params(
                        weight_val, bias_val, call.attrs, new_attrs, mean, var, gama, beta, eps
                    )
                    logging.debug(
                        "Fuse bn to update conv2d's weight scale: %s" % (new_attrs["kernel_scale"])
                    )
                    logging.debug(
                        "Fuse bn to update conv2d's weight zero_point: %s"
                        % (new_attrs["kernel_zero_point"])
                    )
                    weight.data.copyfrom(new_weight_val)
                    bias = relay.expr.const(new_bias_val)

                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_attrs)
                    return new_call
                elif pre_call.op.name == "qnn.csi.deconv2d":
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = pre_call.args[2]

                    weight_val = weight.data.asnumpy()
                    bias_val = bias.data.asnumpy().reshape(1, -1, 1, 1)
                    eps = call.attrs.epsilon
                    gama = op_args[1].data.asnumpy().reshape(1, -1, 1, 1)
                    beta = op_args[2].data.asnumpy().reshape(1, -1, 1, 1)
                    mean = op_args[3].data.asnumpy().reshape(1, -1, 1, 1)
                    var = op_args[4].data.asnumpy().reshape(1, -1, 1, 1)
                    new_weight_val, new_bias_val, new_attrs = self.__updata_params(
                        weight_val, bias_val, call.attrs, new_attrs, mean, var, gama, beta, eps
                    )
                    logging.debug(
                        "Fuse bn to update deconv2d's weight scale: %s"
                        % (new_attrs["kernel_scale"])
                    )
                    logging.debug(
                        "Fuse bn to update deconv2d's weight zero_point: %s"
                        % (new_attrs["kernel_zero_point"])
                    )
                    weight.data.copyfrom(new_weight_val)
                    bias = relay.expr.const(new_bias_val)

                    new_call = relay.qnn.op.csi_deconv2d(data, weight, bias, **new_attrs)
                    return new_call
                else:
                    new_call = Call(new_fn, op_args, call.attrs)
            else:
                new_call = Call(new_fn, op_args, call.attrs)
            return new_call

    class FuseReluMutator(relay.ExprMutator):
        """ Fuse relu layer helper class """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name == "qnn.csi.relu":
                pre_call = op_args[0]
                if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.conv2d":
                    new_attrs = _qnn_attrs(pre_call.attrs)
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = pre_call.args[2]
                    new_attrs["output_scale"] = call.attrs.output_scale
                    new_attrs["output_zero_point"] = call.attrs.output_zero_point
                    new_call = relay.qnn.op.csi_conv2d_relu(data, weight, bias, **new_attrs)
                    return new_call

            #                elif pre_call.op.name == "qnn.csi.dense":
            #                    data = pre_call.args[0]
            #                    weight = pre_call.args[1]
            #                    bias = pre_call.op_args[2]
            #                    new_attrs['axis'] = 0
            #                    new_attrs['output_scale'] = call.attrs.output_scale
            #                    new_attrs['output_zero_point'] = call.attrs.output_zero_point
            #                    new_call = relay.qnn.op.csi_dense(data, weight, bias, **new_attrs)
            #                elif pre_call.op.name == "qnn.csi.deconv2d":
            #                    data = pre_call.args[0]
            #                    weight = pre_call.args[1]
            #                    bias = pre_call.op_args[2]
            #                    new_attrs['output_scale'] = call.attrs.output_scale
            #                    new_attrs['output_zero_point'] = call.attrs.output_zero_point
            #                    new_call = relay.qnn.op.csi_deconv2d(data, weight, bias, **new_attrs)
            elif call.op.name == "qnn.csi.relu6":
                pre_call = op_args[0]
                if pre_call.op.name == "qnn.csi.conv2d":
                    new_attrs = _qnn_attrs(pre_call.attrs)
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = pre_call.args[2]
                    new_attrs["output_scale"] = call.attrs.output_scale
                    new_attrs["output_zero_point"] = call.attrs.output_zero_point
                    new_call = relay.qnn.op.csi_conv2d_relu6(data, weight, bias, **new_attrs)
                    return new_call

            new_call = Call(call.op, op_args, call.attrs)

            return new_call

    class FusePadMutator(relay.ExprMutator):
        """ Fuse pad layer helper class """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name == "qnn.csi.conv2d":
                pre_call = op_args[0]
                if not pre_call or isinstance(pre_call, tvm.relay.expr.Var):
                    new_call = Call(call.op, op_args, call.attrs)
                    return new_call

                if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.pad":
                    if not (
                        pre_call.attrs.pad_mode == "constant" and pre_call.attrs.pad_value == 0
                    ):
                        new_call = Call(call.op, op_args, call.attrs)
                        return new_call

                    new_attrs = _qnn_attrs(call.attrs)
                    data = pre_call.args[0]
                    weight = op_args[1]
                    bias = op_args[2]
                    new_attrs["input_scale"] = pre_call.attrs.input_scale
                    new_attrs["input_zero_point"] = pre_call.attrs.input_zero_point
                    pad_len = len(call.attrs.padding)
                    if pad_len == 4:
                        new_attrs["padding"] = [
                            pre_call.attrs.pad_width[2][0],
                            pre_call.attrs.pad_width[2][1],
                            pre_call.attrs.pad_width[3][0],
                            pre_call.attrs.pad_width[3][1],
                        ]
                    elif pad_len == 2:
                        new_attrs["padding"] = [
                            pre_call.attrs.pad_width[2][0],
                            pre_call.attrs.pad_width[3][0],
                        ]
                    else:
                        raise ValueError("Unsupport padding size:", pad_len)

                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_attrs)
                else:
                    new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)
            return new_call

    class FuseReshapeMutator(relay.ExprMutator):
        """ Fuse reshape helper class """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name == "qnn.csi.dense":
                pre_call = op_args[0]
                new_attrs = _qnn_attrs(call.attrs)
                if pre_call.op.name == "qnn.csi.reshape":
                    data = pre_call.args[0]
                    weight = call.args[1]
                    bias = call.args[2]
                    new_call = relay.qnn.op.csi_dense(data, weight, bias, **new_attrs)
                else:
                    new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)

            return new_call

    def fuse_params_add_mul_before_conv(weight, bias, mul_val, add_val):
        """ update the params in convolution op while add or/and mul op in front of it."""
        assert len(weight.shape) == 4
        new_weight = weight * mul_val
        new_bias = weight * add_val
        new_bias = np.sum(new_bias, (1, 2, 3))
        new_bias = new_bias + bias

        return new_weight.astype(np.float32), new_bias.reshape(-1).astype(np.float32)

    def update_conv_attrs(weight_val, attrs):
        """ update the attrubutions for conv2d op with new weight value."""
        weight_scale, weight_zp = get_weight_params(weight_val)

        attrs["kernel_scale"] = weight_scale
        attrs["kernel_zero_point"] = int(weight_zp)

    class FuseAddBeforeConv(relay.ExprMutator):
        """ Fuse add op in front of the convolution op. """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name == "qnn.csi.conv2d":
                new_conv2d_attrs = _qnn_attrs(call.attrs)
                pre_call = op_args[0]
                if (
                    isinstance(pre_call, Call)
                    and (pre_call.op.name in ("qnn.csi.add", "qnn.csi.bias_add"))
                    and isinstance(pre_call.args[1], Constant)
                ):
                    data = pre_call.args[0]
                    weight = call.args[1]
                    bias = call.args[2]

                    weight_val = weight.data.asnumpy()
                    bias_val = bias.data.asnumpy()
                    add_rhs_val = pre_call.args[1].data.asnumpy()

                    if len(bias_val.shape) == 0:
                        bias_val = np.zeros(weight_val.shape[0])
                    if len(add_rhs_val.shape) == 1:
                        add_rhs_val = np.reshape(add_rhs_val, (1, add_rhs_val.shape[0], 1, 1))

                    if add_rhs_val.size != weight_val.shape[0]:
                        new_call = Call(call.op, op_args, call.attrs)
                        return new_call

                    mul_rhs_val = np.ones_like(add_rhs_val)

                    new_weight_val, new_bias_val = fuse_params_add_mul_before_conv(
                        weight_val, bias_val, mul_rhs_val, add_rhs_val
                    )

                    new_conv2d_attrs["input_scale"] = pre_call.attrs.lhs_scale
                    new_conv2d_attrs["input_zero_point"] = pre_call.attrs.lhs_zero_point
                    update_conv_attrs(new_weight_val, new_conv2d_attrs)

                    weight.data.copyfrom(new_weight_val)
                    bias = relay.expr.const(new_bias_val)

                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                    return new_call
                else:
                    new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)
            return new_call

    class FuseMulBeforeConv(relay.ExprMutator):
        """ Fuse mul op in front of the convolution op. """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]

            if call.op.name == "qnn.csi.conv2d":
                new_conv2d_attrs = _qnn_attrs(call.attrs)
                pre_call = op_args[0]
                if (
                    isinstance(pre_call, Call)
                    and pre_call.op.name == "qnn.csi.mul"
                    and isinstance(pre_call.args[1], Constant)
                ):
                    data = pre_call.args[0]
                    weight = call.args[1]
                    bias = call.args[2]

                    weight_val = weight.data.asnumpy()
                    bias_val = bias.data.asnumpy()
                    mul_rhs_val = pre_call.args[1].data.asnumpy()

                    if len(bias_val.shape) == 0:
                        bias_val = np.zeros(weight_val.shape[0])
                    if mul_rhs_val.size != mul_rhs_val.shape[1]:
                        new_call = Call(call.op, op_args, call.attrs)
                        return new_call

                    add_rhs_val = np.zeros_like(mul_rhs_val)

                    new_weight_val, new_bias_val = fuse_params_add_mul_before_conv(
                        weight_val, bias_val, mul_rhs_val, add_rhs_val
                    )

                    new_conv2d_attrs["input_scale"] = pre_call.attrs.lhs_scale
                    new_conv2d_attrs["input_zero_point"] = pre_call.attrs.lhs_zero_point
                    update_conv_attrs(new_weight_val, new_conv2d_attrs)

                    weight.data.copyfrom(new_weight_val)
                    bias = relay.expr.const(new_bias_val)

                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                    return new_call
                else:
                    new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)
            return new_call

    def fuse_params_add_mul_after_conv(weight, bias, mul_val, add_val):
        """ update the params in convolution op while add or/and mul op in behind it."""
        assert len(weight.shape) == 4
        mul_val = np.reshape(mul_val, (-1, 1, 1, 1))
        add_val = np.reshape(add_val, (-1, 1, 1, 1))
        new_weight = weight * mul_val

        new_bias = np.reshape(bias, mul_val.shape) * mul_val + add_val
        assert new_bias.size == bias.size
        return new_weight.astype(np.float32), new_bias.reshape(-1).astype(np.float32)

    class FuseAddAfterConv(relay.ExprMutator):
        """ Fuse add op in behind the convolution op. """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name in ("qnn.csi.add", "qnn.csi.bias_add") and isinstance(
                op_args[1], Constant
            ):
                pre_call = op_args[0]
                if not isinstance(pre_call, Call):
                    return Call(call.op, op_args, call.attrs)
                if pre_call.op.name == "qnn.csi.conv2d":
                    new_conv2d_attrs = _qnn_attrs(pre_call.attrs)
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = pre_call.args[2]

                    weight_val = weight.data.asnumpy()
                    bias_val = bias.data.asnumpy()
                    add_rhs_val = op_args[1].data.asnumpy()

                    if len(bias_val.shape) == 0:
                        bias_val = np.zeros(weight_val.shape[0])
                    if len(add_rhs_val.shape) == 1:
                        add_rhs_val = np.reshape(add_rhs_val, (1, add_rhs_val.shape[0], 1, 1))

                    if add_rhs_val.size != weight_val.shape[0]:
                        new_call = Call(call.op, op_args, call.attrs)
                        return new_call

                    mul_rhs_val = np.ones_like(add_rhs_val)

                    new_weight_val, new_bias_val = fuse_params_add_mul_after_conv(
                        weight_val, bias_val, mul_rhs_val, add_rhs_val
                    )

                    new_conv2d_attrs["output_scale"] = call.attrs.output_scale
                    new_conv2d_attrs["output_zero_point"] = call.attrs.output_zero_point
                    update_conv_attrs(new_weight_val, new_conv2d_attrs)

                    weight.data.copyfrom(new_weight_val)
                    bias = relay.expr.const(new_bias_val)

                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                    return new_call
                else:
                    if call.op.name == "qnn.csi.bias_add":
                        lhs_shape = _infer_shape(pre_call)
                        rhs_shape = op_args[1].checked_type.concrete_shape
                        if len(lhs_shape) == 4 and len(rhs_shape) == 1:
                            newshape = (1, -1, 1, 1)
                            rhs_data = op_args[1].data.asnumpy()
                            rhs_data = np.reshape(rhs_data, newshape)
                            rhs = relay.expr.const(rhs_data)

                            new_attrs = _qnn_attrs(call.attrs)
                            new_call = relay.qnn.op.csi_add(pre_call, rhs, **new_attrs)
                        else:
                            new_call = Call(call.op, op_args, call.attrs)
                    else:
                        new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)
            return new_call

    class FuseMulAfterConv(relay.ExprMutator):
        """ Fuse mul op in behind the convolution op. """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "qnn.csi.mul" and isinstance(op_args[1], Constant):
                pre_call = op_args[0]
                if isinstance(pre_call, Call) and pre_call.op.name == "qnn.csi.conv2d":
                    new_conv2d_attrs = _qnn_attrs(pre_call.attrs)
                    data = pre_call.args[0]
                    weight = pre_call.args[1]
                    bias = pre_call.args[2]

                    weight_val = weight.data.asnumpy()
                    bias_val = bias.data.asnumpy()
                    mul_rhs_val = op_args[1].data.asnumpy()

                    if len(bias_val.shape) == 0:
                        bias_val = np.zeros(weight_val.shape[0])
                    if mul_rhs_val.size != weight_val.shape[0]:
                        new_call = Call(call.op, op_args, call.attrs)
                        return new_call

                    add_rhs_val = np.zeros_like(mul_rhs_val)

                    new_weight_val, new_bias_val = fuse_params_add_mul_after_conv(
                        weight_val, bias_val, mul_rhs_val, add_rhs_val
                    )

                    new_conv2d_attrs["output_scale"] = call.attrs.output_scale
                    new_conv2d_attrs["output_zero_point"] = call.attrs.output_zero_point
                    update_conv_attrs(new_weight_val, new_conv2d_attrs)

                    weight.data.copyfrom(new_weight_val)
                    bias = relay.expr.const(new_bias_val)

                    new_call = relay.qnn.op.csi_conv2d(data, weight, bias, **new_conv2d_attrs)
                    return new_call
                else:
                    new_call = Call(call.op, op_args, call.attrs)
            else:
                new_call = Call(call.op, op_args, call.attrs)
            return new_call

    mod["main"] = FuseBiasMutator().visit(mod["main"])
    # mod["main"] = FuseBnMutator().visit(mod["main"])
    mod["main"] = FusePadMutator().visit(mod["main"])

    mod["main"] = FuseAddBeforeConv().visit(mod["main"])
    mod["main"] = FuseMulBeforeConv().visit(mod["main"])
    mod["main"] = FuseMulAfterConv().visit(mod["main"])
    mod["main"] = FuseAddAfterConv().visit(mod["main"])

    if fuse_relu:
        mod["main"] = FuseReluMutator().visit(mod["main"])
    mod["main"] = FuseReshapeMutator().visit(mod["main"])
    return mod


def optimize_quantization(mod, broadcast_quantization=False):
    """ Optimize quantization for anole """

    class OptimizeReluMutator(relay.ExprMutator):
        """ Optimize relu layer """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            current_attrs = _qnn_attrs(call.attrs)
            if call.op.name == "qnn.csi.relu":
                pre_call = op_args[0]
                pre_attrs = _qnn_attrs(pre_call.attrs)
                if pre_call.op.name == "qnn.csi.conv2d":
                    pre_attrs["output_scale"] = current_attrs["output_scale"]
                    pre_attrs["output_zero_point"] = current_attrs["output_zero_point"]
                    new_pre_call = relay.qnn.op.csi_conv2d(*pre_call.args, **pre_attrs)

                    current_attrs["input_scale"] = current_attrs["output_scale"]
                    current_attrs["input_zero_point"] = current_attrs["output_zero_point"]
                    new_current_call = relay.qnn.op.csi_relu(new_pre_call, **current_attrs)

                    return new_current_call
                else:
                    new_current_call = Call(call.op, op_args, call.attrs)
            else:
                new_current_call = Call(call.op, op_args, call.attrs)
            return new_current_call

    class OptimizeUnitePreLayerMutator(relay.ExprMutator):
        """ Optimize reshape, resize layer """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            current_attrs = _qnn_attrs(call.attrs)
            if call.op.name in ["qnn.csi.reshape", "qnn.csi.upsampling"]:
                pre_call = op_args[0]
                if isinstance(pre_call, Call):
                    pre_attrs = _qnn_attrs(pre_call.attrs)
                    if (
                        current_attrs["input_scale"] != pre_attrs["output_scale"]
                        or current_attrs["input_zero_point"] != pre_attrs["output_zero_point"]
                    ):
                        assert current_attrs["input_scale"] == current_attrs["output_scale"]
                        assert (
                            current_attrs["input_zero_point"] == current_attrs["output_zero_point"]
                        )

                        pre_attrs["output_scale"] = current_attrs["output_scale"]
                        pre_attrs["output_zero_point"] = current_attrs["output_zero_point"]

                        new_pre_call = _get_csi_op(pre_call.op.name)(*pre_call.args, **pre_attrs)
                        new_current_call = _get_csi_op(call.op.name)(new_pre_call, **current_attrs)

                        return new_current_call

            new_current_call = Call(call.op, op_args, call.attrs)
            return new_current_call

    class OptimizeConcatMutator(relay.ExprMutator):
        """ Optimize concat layer """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            current_attrs = _qnn_attrs(call.attrs)
            if call.op.name == "qnn.csi.concatenate":
                data = op_args[0]

                new_data = list()
                for pre_call in data:
                    try:
                        pre_attrs = _qnn_attrs(pre_call.attrs)
                    except AttributeError:
                        new_current_call = Call(call.op, op_args, call.attrs)
                        return new_current_call
                    pre_attrs["output_scale"] = current_attrs["output_scale"]
                    pre_attrs["output_zero_point"] = current_attrs["output_zero_point"]
                    if pre_call.op.name in ["qnn.csi.reshape", "qnn.csi.upsampling"]:
                        pre_attrs["input_scale"] = current_attrs["output_scale"]
                        pre_attrs["input_zero_point"] = current_attrs["output_zero_point"]
                    new_pre_call = _get_csi_op(pre_call.op.name)(*pre_call.args, **pre_attrs)
                    new_data.append(new_pre_call)

                data_num = len(new_data)
                assert len(current_attrs["input_scales"]) == len(
                    current_attrs["input_zero_points"]
                ), "The length of scales should be the same zero_points"

                current_attrs["input_scales"] = [current_attrs["output_scale"]] * data_num
                current_attrs["input_zero_points"] = [current_attrs["output_zero_point"]] * data_num
                new_current_call = relay.qnn.op.csi_concatenate(Tuple(new_data), **current_attrs)

                return new_current_call

            return Call(call.op, op_args, call.attrs)

    class OptimizeMaxpoolMutator(relay.ExprMutator):
        """Optimize maxpool layer"""

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            current_attrs = _qnn_attrs(call.attrs)
            if call.op.name == "qnn.csi.maxpool":
                pre_call = op_args[0]
                if isinstance(pre_call, Call):
                    pre_attrs = _qnn_attrs(pre_call.attrs)
                    if (
                        current_attrs["input_scale"] != current_attrs["output_scale"]
                        or current_attrs["input_zero_point"] != current_attrs["output_zero_point"]
                    ):
                        current_attrs["input_scale"] = current_attrs["output_scale"]
                        current_attrs["input_zero_point"] = current_attrs["output_zero_point"]

                        pre_attrs["output_scale"] = current_attrs["output_scale"]
                        pre_attrs["output_zero_point"] = current_attrs["output_zero_point"]

                    new_pre_call = _get_csi_op(pre_call.op.name)(*pre_call.args, **pre_attrs)
                    new_call = _get_csi_op(call.op.name)(new_pre_call, **current_attrs)
                    return new_call
            new_call = Call(call.op, op_args, call.attrs)
            return new_call

    mod["main"] = OptimizeConcatMutator().visit(mod["main"])
    if broadcast_quantization:
        mod["main"] = OptimizeMaxpoolMutator().visit(mod["main"])
    mod["main"] = OptimizeUnitePreLayerMutator().visit(mod["main"])
    # mod['main'] = OptimizeReluMutator().visit(mod['main'])
    return mod


def const_to_uint8(mod):
    """for asym quantization, convert const to uint8.

    Returns
    -------
    ret: Function
        The module pass function.
    """
    cop = csi_op()
    visited_var = dict()

    # def wrapped_func(mod, ctx): # pylint: disable=unused-argument
    class AddInitMutator(relay.ExprMutator):
        """ Add init """

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if isinstance(call.args[0], tvm.relay.expr.Var):
                if cop.diso_op(call.op.name):
                    output_scale = call.attrs.lhs_scale
                    output_zp = call.attrs.lhs_zero_point
                else:
                    output_scale = call.attrs.input_scale
                    output_zp = call.attrs.input_zero_point

                # The Var node that has been visited, should be the common input.
                if op_args[0] in visited_var.keys():
                    init_call = visited_var[op_args[0]]
                else:
                    init_call = relay.qnn.op.csinn_init(
                        op_args[0], output_scale, output_zp, op_args[0].checked_type.dtype
                    )
                    visited_var[op_args[0]] = init_call

                new_attrs = _qnn_attrs(call.attrs)
                if cop.siso_op(call.op.name):
                    handler = cop.siso_handler(call.op.name)
                    new_call = handler(init_call, **new_attrs)
                elif cop.diso_op(call.op.name):
                    handler = cop.diso_handler(call.op.name)
                    new_call = handler(init_call, op_args[1], **new_attrs)
                elif call.op.name == "qnn.csi.conv2d":
                    new_call = relay.qnn.op.csi_conv2d(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.deconv3d":
                    new_call = relay.qnn.op.csi_deconv3d(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.conv2d_relu":
                    new_call = relay.qnn.op.csi_conv2d_relu(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.conv2d_relu6":
                    new_call = relay.qnn.op.csi_conv2d_relu6(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.dense":
                    new_call = relay.qnn.op.csi_dense(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.crop_resize":
                    new_call = relay.qnn.op.csi_crop_resize(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.deconv2d":
                    new_call = relay.qnn.op.csi_deconv2d(
                        init_call, op_args[1], op_args[2], **new_attrs
                    )
                elif call.op.name == "qnn.csi.split":
                    new_call = relay.qnn.op.csi_split(init_call, **new_attrs)
                else:
                    raise ValueError("Unsupport op:", call.op.name)
            else:
                new_call = Call(call.op, op_args, call.attrs)
            return new_call

    class AddDeinitMutator(relay.ExprMutator):
        """ Add deinit """

        def visit_call(self, call):
            input_scale = call.attrs.output_scale
            input_zp = call.attrs.output_zero_point
            new_call = relay.qnn.op.csinn_deinit(call, input_scale, input_zp)
            return new_call

    class CastU8Mutator(relay.ExprMutator):
        """ Cast """

        def _cast_weight(self, wdata, wscale, wzp):
            weight_dtype = qconfig().dtype_weight
            bit = qconfig().nbit_weight
            if "uint" in weight_dtype:
                lmt_min = 0
                lmt_max = 2 ** bit - 1
            else:
                lmt_min = -(2 ** (bit - 1) - 1)
                lmt_max = 2 ** (bit - 1) - 1
            wdata = np.round(wdata / wscale + wzp)
            wdata = np.clip(wdata, lmt_min, lmt_max)
            wdata = wdata.astype(weight_dtype)
            weight = relay.expr.const(wdata, dtype=weight_dtype)
            return weight

        def __update_channel_params(self, call, op_args):
            input_scale = call.attrs.input_scale
            weight = op_args[1]
            bias = op_args[2]
            wdata = weight.data.asnumpy()
            bdata = bias.data.asnumpy()

            wscales, wzps = get_weight_params_per_channel(wdata)

            # Prevent the value of bias from exceeding the range of int32 after round
            overflow_map = abs(bdata / (wscales * input_scale)) > 2147483647

            if len(overflow_map[overflow_map == True]) > 0:
                logging.warning("bias will overflow! Force changed wscale!")
                wscales[overflow_map] = np.abs(np.max(wdata, axis=(1, 2, 3))[overflow_map])
                wscales[overflow_map] = 1.0

            wscales = wscales.reshape([-1, 1, 1, 1])
            wzps = wzps.reshape([-1, 1, 1, 1])
            wdata = np.round(wdata / wscales + wzps)
            wdata = np.clip(wdata, 0, 255)
            wdata = wdata.astype(qconfig().dtype_weight)
            weight = relay.expr.const(wdata, dtype=qconfig().dtype_weight)

            bscale = wscales * input_scale
            bdata = np.round(bdata / bscale.reshape(-1)).astype("int32")
            bias = relay.expr.const(bdata, dtype="int32")
            wscales = relay.expr.const(wscales.reshape(-1))
            wzps = relay.expr.const(wzps.reshape(-1))
            return weight, bias, wscales, wzps

        def __update_params(self, call, op_args):
            activation_dtype = qconfig().dtype_activation
            weight = op_args[1]
            wdata = weight.data.asnumpy()
            wzp = call.attrs.kernel_zero_point
            wscale = call.attrs.kernel_scale
            weight = self._cast_weight(wdata, wscale, wzp)
            bias = op_args[2]
            bdata = bias.data.asnumpy()
            bscale = wscale * call.attrs.input_scale
            bdata = np.round(bdata / bscale).astype(activation_dtype)
            bias = relay.expr.const(bdata, dtype=activation_dtype)

            return weight, bias

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            new_attrs = _qnn_attrs(call.attrs)
            if "out_dtype" in new_attrs:
                new_attrs["out_dtype"] = qconfig().dtype_input
            cop = csi_op()
            if cop.conv_op(call.op.name):
                if qconfig().channel_quantization:
                    op_handler = cop.conv_handler(call.op.name + "_channel")
                    weight, bias, wscales, wzps = self.__update_channel_params(call, op_args)
                    del new_attrs["kernel_scale"]
                    del new_attrs["kernel_zero_point"]
                    new_call = op_handler(op_args[0], weight, bias, wscales, wzps, **new_attrs)
                else:
                    op_handler = cop.conv_handler(call.op.name)
                    weight, bias = self.__update_params(call, op_args)
                    new_call = op_handler(op_args[0], weight, bias, **new_attrs)
            elif call.op.name == "qnn.csi.dense":
                weight, bias = self.__update_params(call, op_args)
                new_call = relay.qnn.op.csi_dense(op_args[0], weight, bias, **new_attrs)
            elif call.op.name == "qnn.csi.prelu":
                data = op_args[0]
                alpha = op_args[1]
                adata = alpha.data.asnumpy()
                ascale = call.attrs.alpha_scale
                azp = call.attrs.alpha_zero_point
                alpha = self._cast_weight(adata, ascale, azp)
                new_call = relay.qnn.op.csi_prelu(data, alpha, **new_attrs)
            elif call.op.name == "qnn.csi.full":
                data = op_args[0]
                shape = op_args[1]
                del new_attrs["shape"]
                new_call = relay.qnn.op.csi_full(data, shape, **new_attrs)

            elif cop.diso_op(call.op.name):
                lhs = op_args[0]
                rhs = op_args[1]
                if (
                    call.op.name
                    in [
                        "qnn.csi.add",
                        "qnn.csi.mul",
                        "qnn.csi.bias_add",
                        "qnn.csi.subtract",
                        "qnn.csi.div",
                        "qnn.csi.power",
                        "qnn.csi.mod",
                    ]
                ) and isinstance(rhs, Constant):
                    rdata = rhs.data.asnumpy()
                    rzp = call.attrs.rhs_zero_point
                    rscale = call.attrs.rhs_scale
                    rdata = np.round(rdata / rscale + rzp).astype(qconfig().dtype_weight)
                    rhs = relay.expr.const(rdata, dtype=qconfig().dtype_weight)
                op_handler = cop.diso_handler(call.op.name)
                new_call = op_handler(lhs, rhs, **new_attrs)
            elif call.op.name == "qnn.csi.maxpool2d_locat":
                data = op_args[0]
                new_attrs["out_dtype"] = "int32"
                new_call = relay.qnn.op.csi_max_pool2d_locat(data, **new_attrs)
            elif call.op.name == "qnn.csi.unravel_index":
                data = op_args[0]
                shape = op_args[1]
                new_attrs["out_dtype"] = "int32"
                new_call = relay.qnn.op.csi_unravel_index(data, shape, **new_attrs)
            else:
                op_handler = _get_csi_op(call.op.name)
                new_call = op_handler(*op_args, **new_attrs)

            return new_call

    func = mod["main"]
    func = AddInitMutator().visit(func)
    func = AddDeinitMutator().visit(func)
    func = CastU8Mutator().visit(func)
    mod = IRModule.from_expr(func)
    return mod
