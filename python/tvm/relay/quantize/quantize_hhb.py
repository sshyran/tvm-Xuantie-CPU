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
# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import
"""Automatic quantization toolkit."""
import logging
from tvm import relay
from tvm.relay.frontend.common import infer_shape as _infer_shape
from tvm.relay.frontend.common import create_span
from .csi_layout_convert import csi_layout_convert
from .custom_fusion_pass import FuseCacheMatMul, FuseLayerNormal, TConv1dAddT
from .custom_fusion_pass import Conv2dSqueezeAdd, FuseCacheConv1d
from .convert_to_relay import convert_to_relay


from .op_spliter import ConvSpliter

from .. import transform as _transform
from .. import expr as _expr
from ...ir import transform

from ._convert_to_csi import (
    calibration,
    convert_to_csi_qnn,
    fuse_layer,
    optimize_quantization,
    current_csinn_config,
)

LOG = 25
logger = logging.getLogger("HHB")


def _check_unsupported_ops(target, model):
    x86_op_list = [
        "abs",
        "acos",
        "acosh",
        "add",
        "argmax",
        "argmin",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "broadcast_to",
        "cast",
        "ceil",
        "clip",
        "clip",
        "concatenate",
        "cos",
        "cosh",
        "divide",
        "equal",
        "erf",
        "exp",
        "expand_dims",
        "floor",
        "floor_divide",
        "floor_mod",
        "full",
        "image.dilation2d",
        "image.resize2d",
        "left_shift",
        "log",
        "max",
        "maximum",
        "mean",
        "min",
        "minimum",
        "mod",
        "multiply",
        "negative",
        "nn.avg_pool2d",
        "nn.avg_pool3d",
        "nn.batch_flatten",
        "nn.batch_matmul",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv1d",
        "nn.conv2d_transpose",
        "nn.conv3d",
        "nn.conv3d_transpose",
        "nn.dense",
        "nn.depth_to_space",
        "nn.fsmn",
        "nn.global_avg_pool2d",
        "nn.global_max_pool2d",
        "nn.layer_norm",
        "nn.leaky_relu",
        "nn.log_softmax",
        "nn.lrn",
        "nn.max_pool2d",
        "nn.max_pool2d_with_argmax",
        "nn.max_pool3d",
        "nn.pad",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.space_to_depth",
        "nn.upsampling",
        "power",
        "prod",
        "reshape",
        "reverse",
        "right_shift",
        "round",
        "rsqrt",
        "scatter_nd",
        "sigmoid",
        "sign",
        "sin",
        "sinh",
        "split",
        "sqrt",
        "squeeze",
        "strided_slice",
        "subtract",
        "sum",
        "take",
        "tan",
        "tanh",
        "tile",
        "transpose",
        "vision.max_pool2d_location",
        "vision.proposal",
        "vision.psroipooling",
        "vision.roi_pool",
        "segment_max",
        "segment_mean",
        "segment_min",
        "segment_prod",
        "segment_sum",
        "vision.unpooling",
    ]
    anole_op_list = [
        "add",
        "cast",
        "clip",
        "concatenate",
        "divide",
        "equal",
        "exp",
        "image.resize2d",
        "mean",
        "minimum",
        "multiply",
        "nn.avg_pool2d",
        "nn.batch_flatten",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv2d_transpose",
        "nn.dense",
        "nn.global_avg_pool2d",
        "nn.global_max_pool2d",
        "nn.leaky_relu",
        "nn.lrn",
        "nn.max_pool2d",
        "nn.max_pool2d_with_argmax",
        "nn.pad",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.upsampling",
        "reshape",
        "sigmoid",
        "split",
        "squeeze",
        "strided_slice",
        "subtract",
        "transpose",
        "vision.max_pool2d_location",
        "vision.proposal",
        "vision.psroipooling",
        "vision.roi_pool",
        "vision.unpooling",
    ]
    ch8601_op_list = [
        "add",
        "clip",
        "concatenate",
        "exp",
        "image.resize2d",
        "mean",
        "multiply",
        "nn.avg_pool2d",
        "nn.batch_flatten",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv2d_transpose",
        "nn.dense",
        "nn.global_avg_pool2d",
        "nn.global_max_pool2d",
        "nn.leaky_relu",
        "nn.lrn",
        "nn.max_pool2d",
        "nn.max_pool2d_with_argmax",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.upsampling",
        "reshape",
        "sigmoid",
        "split",
        "squeeze",
        "strided_slice",
        "transpose",
        "vision.max_pool2d_location",
        "vision.proposal",
        "vision.psroipooling",
        "vision.unpooling",
    ]
    dp1k_op_list = [
        "add",
        "clip",
        "concatenate",
        "image.resize2d",
        "mean",
        "multiply",
        "nn.avg_pool2d",
        "nn.global_avg_pool2d",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv2d_transpose",
        "nn.dense",
        "nn.max_pool2d",
        "nn.leaky_relu",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.upsampling",
        "reshape",
        "sigmoid",
        "strided_slice",
        "transpose",
    ]
    light_op_list = [
        "add",
        "argmax",
        "cast",
        "clip",
        "concatenate",
        "divide",
        "exp",
        "expand_dims",
        "image.resize2d",
        "mean",
        "multiply",
        "nn.avg_pool2d",
        "nn.batch_flatten",
        "nn.bias_add",
        "nn.conv2d",
        "nn.conv2d_transpose",
        "nn.dense",
        "nn.depth_to_space",
        "nn.global_avg_pool2d",
        "nn.global_max_pool2d",
        "nn.leaky_relu",
        "nn.lrn",
        "nn.max_pool2d",
        "nn.max_pool2d_with_argmax",
        "nn.pad",
        "nn.prelu",
        "nn.relu",
        "nn.softmax",
        "nn.upsampling",
        "minimum",
        "maximum",
        "reshape",
        "sigmoid",
        "split",
        "squeeze",
        "strided_slice",
        "subtract",
        "transpose",
        "vision.max_pool2d_location",
        "vision.proposal",
        "vision.psroipooling",
        "vision.roi_pool",
        "vision.unpooling",
    ]

    qnn_op_list = [
        "qnn.csi.add",
        "qnn.csi.avgpool2d",
        "qnn.csi.concatenate",
        "qnn.csi.conv2d",
        "qnn.csi.depth_to_space",
        "qnn.csi.dense",
        "qnn.csi.minimum",
        "qnn.csi.relu6",
        "qnn.csi.relu",
        "qnn.csi.reshape",
        "qnn.csi.softmax",
    ]

    custom_op_list = [
        "cache_matmul",
        "cache_conv1d",
    ]

    op_maps = {
        "x86_ref": x86_op_list,
        "anole": anole_op_list,
        "light": light_op_list,
        "light_new": light_op_list,
        "ch8601": ch8601_op_list,
        "dp1k": dp1k_op_list,
        "c906": x86_op_list,
        "c908": x86_op_list,
        "i805": x86_op_list,
        "c860": x86_op_list,
        "hlight": x86_op_list,
        "asp": x86_op_list,
    }

    class GetModelOps(relay.ExprVisitor):
        """Get the operation name of the input model used"""

        def __init__(self):
            super(GetModelOps, self).__init__()
            self.op_lists = []

        def visit_call(self, call):
            _ = [self.visit(arg) for arg in call.args]
            op_name = call.op.name
            if op_name not in self.op_lists:
                self.op_lists.append(op_name)

    if target not in op_maps:
        raise Exception(f'Unspported this target "{target}"')

    get_model_ops = GetModelOps()
    get_model_ops.visit(model["main"])
    model_ops = get_model_ops.op_lists
    unsupported_ops = []
    quanted_model = False
    for op_name in model_ops:
        if op_name not in op_maps[target] + qnn_op_list + custom_op_list:
            unsupported_ops.append(op_name)
        if op_name in qnn_op_list:
            quanted_model = True
    if len(unsupported_ops) > 0:
        raise Exception(f"Unspported ops {unsupported_ops} in target {target}")
    return quanted_model


def _bind_params(func, params):
    """Bind the params to the expression."""
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def check_bn_variance(model):
    "Make sure data in variance is not negtive"

    class CheckBNVar(relay.ExprMutator):
        def visit_call(self, call):
            new_fn = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "nn.batch_norm":
                var = new_args[4].data.asnumpy()
                var[var < 0] = 0
                new_args[4] = _expr.const(var)

            return _expr.Call(new_fn, new_args, call.attrs, call.type_args, call.span)

    model["main"] = CheckBNVar().visit(model["main"])
    return model


def get_count_call(mod):
    """Get the count of call in relay ir"""

    class GetCountVisitor(relay.ExprVisitor):
        """Counting the number of call"""

        def __init__(self):
            super(GetCountVisitor, self).__init__()
            self.memo_map = {}
            self.call_count = 0

        def visit_call(self, call):
            self.call_count += 1
            _ = [self.visit(arg) for arg in call.args]

    gc = GetCountVisitor()
    gc.visit(mod["main"])
    return gc.call_count


def InsertNOp(mod):
    """insert Nop"""

    class BetweenLekayReLUAndAdd(relay.ExprMutator):
        """insert Nop between leakyrelu and and"""

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name == "add":
                l_pre_call = op_args[0]
                r_pre_call = op_args[1]

                if isinstance(l_pre_call, _expr.Call) and l_pre_call.op.name == "nn.leaky_relu":
                    shape = _infer_shape(l_pre_call)
                    mul_call = relay.op.strided_slice(l_pre_call, [0, 0, 0, 0], shape)
                    mul_call = _expr.Call(
                        mul_call.op,
                        mul_call.args,
                        mul_call.attrs,
                        mul_call.type_args,
                        create_span("strided_slice_inserted_between_leakyrelu_add"),
                    )
                    new_call = relay.op.add(mul_call, r_pre_call)
                    new_call = _expr.Call(
                        new_call.op, new_call.args, new_call.attrs, new_call.type_args, call.span
                    )
                    return new_call
            new_call = _expr.Call(call.op, op_args, call.attrs, call.type_args, call.span)
            return new_call

    mod["main"] = BetweenLekayReLUAndAdd().visit(mod["main"])

    return mod


def quantize_hhb(module, params=None, dataset=None, target="x86_ref"):
    """The quantization procedure.

    Parameters
    ---------
    module: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """

    curr_qconfig = current_csinn_config()
    if target in ("light", "hlight") and curr_qconfig.quantization_scheme not in [
        "int16_sym",
        "int8_sym",
    ]:
        module = InsertNOp(module)

    if params:
        module["main"] = _bind_params(module["main"], params)

    module = check_bn_variance(module)

    call_count = get_count_call(module)
    opt_seq = [
        _transform.SimplifyInference(),
        _transform.DynamicToStatic(),
        _transform.FoldConstant(),
        # _transform.FoldScaleAxis(),
        # _transform.CanonicalizeOps(),
        # _transform.FoldConstant(),
        # user-define passes
        # _transform.SpaceToBatch2AtrousConv(),
    ]
    if call_count > 1:
        opt_seq.insert(2, _transform.SimplifyExpr())
    if curr_qconfig.use_custom_fusion:
        logger.warning("Using custom fusion.")
        opt_seq += [FuseCacheMatMul(), FuseLayerNormal(), TConv1dAddT(), FuseCacheConv1d()]
    optimizer = transform.Sequential(opt_seq)
    logger.log(LOG, "Start optimization.")
    module = optimizer(module)
    logger.debug("Optimized model:")
    logger.debug(module["main"])
    logger.log(LOG, "Optimization completed!")

    quanted_model = _check_unsupported_ops(target, module)

    dtype_float = False
    if curr_qconfig.dtype_weight in ("float16", "bfloat16") or (
        (target not in ("light", "hlight")) and curr_qconfig.dtype_weight == "float32"
    ):
        dtype_float = True

    if curr_qconfig.convert_to_relay and quanted_model:
        convert_to_relay(module)
        quanted_model = False

    if dtype_float:
        logger.log(LOG, "Start conversion to csinn.")
        if dataset:
            logger.log(LOG, "Ignore calibrate dataset in f16/bf16/f32 conversion.")
        module = convert_to_csi_qnn(module, None)
        logger.debug("Converted model:")
        logger.debug(module["main"])
        logger.log(LOG, "Conversion completed!")
    elif dataset and not quanted_model:
        quant_params = calibration(module, dataset)
        logger.log(LOG, "Start conversion to csinn.")
        module = convert_to_csi_qnn(module, quant_params)
        logger.debug("Converted model:")
        logger.debug(module["main"])
        logger.log(LOG, "Conversion completed!")
    else:
        if not quanted_model:
            raise Exception("Can't find calibration dataset!")

    logger.log(LOG, "Start operator fusion.")
    fuse_pass = [Conv2dSqueezeAdd()]
    fuser = transform.Sequential(fuse_pass)
    module = fuser(module)
    csi_module = fuse_layer(module)
    logger.debug("Fused model:")
    logger.debug(csi_module["main"])
    logger.log(LOG, "Operator fusion completed!")

    csi_module = optimize_quantization(
        csi_module, curr_qconfig.broadcast_quantization, target=curr_qconfig.target
    )

    logger.log(LOG, "Start operator split.")
    split_pass = [ConvSpliter(curr_qconfig)]
    spliter = transform.Sequential(split_pass)
    csi_module = spliter(csi_module)
    logger.log(LOG, "Operator split completed!")

    csi_module = relay.transform.InferType()(csi_module)
    if curr_qconfig.layout == "NHWC":
        logger.log(LOG, "Start layout convert.")
        csi_module = csi_layout_convert(csi_module)
        logger.log(LOG, "Layout convert completed!")

    csi_module = relay.transform.InferType()(csi_module)
    logger.info("Quantized model:")
    logger.info(csi_module["main"])

    return csi_module
