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
"""Convert csinn model layout."""
from tvm import relay
from ..frontend.common import infer_shape
from ._convert_to_csi import _qnn_attrs, _get_csi_op
from ..expr import Constant, Tuple
from .. import function as _function

NCHW2NHWC_FUNCS = {}


def nchw2nhwc_attrs_changer(attrs):
    """Change layout attributes"""

    attrs = _qnn_attrs(attrs)
    if "data_layout" in attrs:
        attrs["data_layout"] = "NHWC"
    if "out_layout" in attrs:
        attrs["out_layout"] = "NHWC"
    if "kernel_layout" in attrs:
        attrs["kernel_layout"] = "OHWI"
    if "layout" in attrs:
        attrs["layout"] = "OHWI"
    return attrs


def nchw2nhwc_func_register(func_name):
    """Register func in NCHW2NHWC_FUNCS"""

    def decorator(func):
        NCHW2NHWC_FUNCS[func_name] = func.__name__

        def wrapper(self, call, op_args):
            attrs = nchw2nhwc_attrs_changer(call.attrs)
            return func(self, op_args, attrs)

        return wrapper

    return decorator


def nchw_to_nhwc(mod):
    """Convert layout from NCHW to NHWC"""

    class NCHW2NHWCMutaor(relay.ExprMutator):
        """Convert layout from NCHW to NHWC"""

        def __init__(self):
            super(NCHW2NHWCMutaor, self).__init__()
            self.unregistered_func = []

        def list_convert(self, src_list):
            if len(src_list) == 4:
                return [src_list[i] for i in [0, 2, 3, 1]]
            return src_list

        def axis_convert(self, axis):
            convert_axis = [0, 3, 1, 2]
            return convert_axis[axis]

        def constant_convert(self, src_constat, is_depthwise=False):
            """Convert constant value layout"""
            if isinstance(src_constat, Constant):
                np_value = src_constat.data.asnumpy()
                value_rank = len(np_value.shape)
                if value_rank == 4:
                    if is_depthwise:
                        np_value = np_value.transpose([1, 2, 3, 0])
                    else:
                        np_value = np_value.transpose([0, 2, 3, 1])

                return relay.const(np_value, str(np_value.dtype))
            return src_constat

        def visit_call(self, call):
            op_args = [self.visit(arg) for arg in call.args]
            if call.op.name in NCHW2NHWC_FUNCS:
                func = getattr(self, NCHW2NHWC_FUNCS[call.op.name])
                new_call = func(call=call, op_args=op_args)
            else:
                attrs = nchw2nhwc_attrs_changer(call.attrs)
                func = _get_csi_op(call.op.name)
                new_call = func(*op_args, **attrs)

            return new_call

        def visit_var(self, var):
            shape = list(var.checked_type.concrete_shape)
            new_shape = self.list_convert(shape)
            dtype = var.checked_type.dtype
            name = var.name_hint
            return relay.var(name, shape=new_shape, dtype=dtype)

        def diso_convert(self, op_args, attrs, op_name):
            op_args[1] = self.constant_convert(op_args[1])
            func = _get_csi_op("qnn.csi." + op_name)
            return func(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.conv2d")
        def conv2d(self, op_args, attrs):
            """convert conv2d layout"""
            dshape = infer_shape(op_args[0])
            wshape = infer_shape(op_args[1])
            is_depthwise = False
            if attrs["groups"] != 1 and attrs["groups"] == dshape[3] == wshape[0]:
                is_depthwise = True
            op_args[1] = self.constant_convert(op_args[1], is_depthwise)
            return relay.qnn.op.csi_conv2d(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.reshape")
        def reshape(self, op_args, attrs):
            """convert reshape layout"""
            in_shape_rank = infer_shape(op_args[0])
            newshape_rank = len(attrs["newshape"])
            if in_shape_rank == 4 and newshape_rank != 4:
                axes = [0, 3, 1, 2]
                out_dtype = attrs["out_dtype"]
                q_params = attrs["q_params"]
                layer_name = attrs["layer_name"]
                op_args[1] = relay.qnn.op.csi_transpose(
                    op_args[1], axes, out_dtype, q_params, layer_name
                )
            attrs["newshape"] = self.list_convert(attrs["newshape"])
            return relay.qnn.op.csi_reshape(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.depth_to_space")
        def depth_to_space(self, op_args, attrs):
            """convert depth_to_space layout"""
            attrs["layout"] = "NHWC"
            return relay.qnn.op.csi_depth_to_space(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.softmax")
        def softmax(self, op_args, attrs):
            """convert softmax layout"""
            in_expr = op_args[0]
            in_shape_rank = len(infer_shape(in_expr))
            if in_shape_rank == 4:
                attrs["axis"] = self.axis_convert(attrs["axis"])
            return relay.qnn.op.csi_softmax(*op_args, **attrs)

        @nchw2nhwc_func_register("qnn.csi.squeeze")
        def squeeze(self, op_args, attrs):
            """convert squeeze layout"""
            in_expr = op_args[0]
            in_shape_rank = len(infer_shape(in_expr))
            if in_shape_rank == 4:
                new_axis = []
                for i in attrs["axis"]:
                    new_axis.append(self.axis_convert(int(i)))
                attrs["axis"] = new_axis
            return relay.qnn.op.csi_squeeze(*op_args, **attrs)

        # DISO
        @nchw2nhwc_func_register("qnn.csi.subtract")
        def subtract(self, op_args, attrs):
            """convert subtract layout"""
            return self.diso_convert(op_args, attrs, "subtract")

        @nchw2nhwc_func_register("qnn.csi.mul")
        def mul(self, op_args, attrs):
            """convert mul layout"""
            return self.diso_convert(op_args, attrs, "mul")

        @nchw2nhwc_func_register("qnn.csi.add")
        def add(self, op_args, attrs):
            """convert add layout"""
            return self.diso_convert(op_args, attrs, "add")

        @nchw2nhwc_func_register("qnn.csi.div")
        def div(self, op_args, attrs):
            """convert div layout"""
            return self.diso_convert(op_args, attrs, "div")

        @nchw2nhwc_func_register("qnn.csi.minimum")
        def minimum(self, op_args, attrs):
            """convert minimum layout"""
            return self.diso_convert(op_args, attrs, "minimum")

        @nchw2nhwc_func_register("qnn.csi.concatenate")
        def concatenate(self, op_args, attrs):
            """convert concatenate layout"""

            in_rank = len(infer_shape(op_args[0].fields[0]))
            new_args = []
            for arg in op_args[0]:
                new_args.append(self.constant_convert(arg))
            if in_rank == 4:
                attrs["axis"] = self.axis_convert(attrs["axis"])
            return relay.qnn.op.csi_concatenate(Tuple(new_args), **attrs)

        def visit_function(self, fn):
            new_params = [self.visit(x) for x in fn.params]
            new_body = self.visit(fn.body)
            return _function.Function(list(new_params), new_body)

    convert = NCHW2NHWCMutaor()
    mod["main"] = convert.visit(mod["main"])
    return mod


def csi_layout_convert(mod, src_layout="NCHW", dest_layout="NHWC"):
    """layout convert"""

    if src_layout == "NCHW" and dest_layout == "NHWC":
        mod = nchw_to_nhwc(mod)

    return mod
