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
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
"""Definition of vivante operator strategy."""
from .generic import *
from .. import op as _op


@dense_strategy.register("vivante")
def dense_strategy_vivante(attrs, inputs, out_type, target):
    """dense vivante strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_dense(topi.mali.dense),
        wrap_topi_schedule(topi.mali.schedule_dense),
        name="dense.mali",
    )
    return strategy


@dilation2d_strategy.register("vivante")
def dilation2d_strategy_vivante(attrs, inputs, out_type, target):
    """dilation2d_strategy generic strategy"""
    logger.warning("dilation2d_strategy is not optimized for this platform.")
    strategy = _op.OpStrategy()
    dilations = get_const_tuple(attrs.dilations)
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout

    assert layout in ["NCHW", "NHWC"]
    (dilation_h, dilation_w) = dilations
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if layout == "NCHW":
        assert kernel_layout == "IHW"
        strategy.add_implementation(
            wrap_compute_dilation2d(topi.image.dilation2d_nchw),
            wrap_topi_schedule(topi.vivante.schedule_injective),
            name="dilation2d_nchw.vivante",
        )
    elif layout == "NHWC":
        assert kernel_layout == "HWI"
        strategy.add_implementation(
            wrap_compute_dilation2d(topi.image.dilation2d_nhwc),
            wrap_topi_schedule(topi.vivante.schedule_injective),
            name="dilation2d_nhwc.vivante",
        )
    else:
        raise RuntimeError("Unsupported dilation2d layout {}".format(layout))
    return strategy


@softmax_strategy.register(["vivante"])
def softmax_strategy_vivante(attrs, inputs, out_type, target):
    """softmax vivante strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_softmax(topi.nn.softmax),
        wrap_topi_schedule(topi.vivante.schedule_softmax),
        name="softmax.vivante",
    )
    return strategy


@schedule_log_softmax.register(["vivante"])
def schedule_log_softmax_vivante(attrs, outs, target):
    """scheudle log_softmax for vivante"""
    with target:
        return topi.vivante.schedule_softmax(outs)


@schedule_lrn.register(["vivante"])
def schedule_lrn_vivante(attrs, outs, target):
    """schedule LRN for vivante"""
    with target:
        return topi.vivante.schedule_lrn(outs)


@conv2d_transpose_strategy.register(["vivante"])
def conv2d_transpose_strategy_vivante(attrs, inputs, out_type, target):
    """conv2d_transpose vivante strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCHW", "only support nchw for now"
    assert dilation == (1, 1), "not support dilate now"
    assert groups == 1, "only support groups == 1 for now"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d_transpose(topi.cuda.conv2d_transpose_nchw),
        wrap_topi_schedule(topi.vivante.schedule_conv2d_transpose_nchw),
        name="conv2d_transpose_nchw.vivante",
    )
    return strategy


@conv1d_transpose_strategy.register(["vivante"])
def conv1d_transpose_strategy_vivante(attrs, inputs, out_type, target):
    """conv1d_transpose vivante strategy"""
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    groups = attrs.groups
    assert layout == "NCW", "conv1d_transpose ncw only supported"
    assert dilation == (1,), "conv1d_transpose dilation is not supported"
    assert groups == 1, "conv1d_transpose groups == 1 only supported"
    strategy.add_implementation(
        wrap_compute_conv1d_transpose(topi.cuda.conv1d_transpose_ncw),
        wrap_topi_schedule(topi.vivante.schedule_conv1d_transpose_ncw),
        name="conv1d_transpose_ncw.vivante",
    )
    return strategy


@deformable_conv2d_strategy.register(["vivante"])
def deformable_conv2d_strategy_vivante(attrs, inputs, out_type, target):
    """deformable_conv2d vivante strategy"""
    layout = attrs.data_layout
    assert layout == "NCHW"
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_deformable_conv2d(topi.cuda.deformable_conv2d_nchw),
        wrap_topi_schedule(topi.vivante.schedule_deformable_conv2d_nchw),
        name="deformable_conv2d_nchw.vivante",
    )
    return strategy


@conv1d_strategy.register(["vivante"])
def conv1d_strategy_vivante(attrs, inputs, out_type, target):
    """conv1d vivante strategy"""
    layout = attrs.data_layout
    dilation = get_const_tuple(attrs.dilation)
    if dilation[0] < 1:
        raise ValueError("dilation should be a positive value")
    strategy = _op.OpStrategy()
    if layout == "NCW":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.cuda.conv1d_ncw),
            wrap_topi_schedule(topi.vivante.schedule_conv1d_ncw),
            name="conv1d_ncw.vivante",
        )
    elif layout == "NWC":
        strategy.add_implementation(
            wrap_compute_conv1d(topi.cuda.conv1d_nwc),
            wrap_topi_schedule(topi.vivante.schedule_conv1d_nwc),
            name="conv1d_nwc.vivante",
        )
    else:
        raise ValueError("Unsupported conv1d layout {}".format(layout))
    return strategy


@conv2d_strategy.register(["vivante"])
def conv2d_strategy_vivante(attrs, inputs, out_type, target):
    """conv2d vivante strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    stride_h, stride_w = attrs.get_int_tuple("strides")
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    # padding = attrs.get_int_tuple("padding")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            if data.dtype in ("int8", "uint8") and kernel.dtype in ("int8", "uint8"):
                assert data.dtype == kernel.dtype
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_int8),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_int8),
                    name="conv2d_nchw_int8.cuda",
                )
            else:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw),
                    name="conv2d_nchw.cuda",
                )
            _, _, kh, kw = get_const_tuple(kernel.shape)
            if (
                2 < kh < 8
                and 2 < kw < 8
                and kh == kw
                and stride_h == 1
                and stride_w == 1
                and dilation_h == 1
                and dilation_w == 1
            ):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.cuda.conv2d_nchw_winograd),
                    wrap_topi_schedule(topi.cuda.schedule_conv2d_nchw_winograd),
                    name="conv2d_nchw_winograd.cuda",
                    plevel=5,
                )
        elif layout == "HWCN":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_hwcn),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_hwcn),
                name="conv2d_hwcn.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc),
                name="conv2d_nhwc.cuda",
            )
            N, _, _, _ = get_const_tuple(data.shape)
            _, _, CI, CO = get_const_tuple(kernel.shape)
            if target.target_name == "cuda" and nvcc.have_tensorcore(tvm.gpu(0).compute_version):
                if (
                    (N % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
                    or (N % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
                    or (N % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.cuda.conv2d_nhwc_tensorcore),
                        wrap_topi_schedule(topi.cuda.schedule_conv2d_nhwc_tensorcore),
                        name="conv2d_nhwc_tensorcore.cuda",
                        plevel=20,
                    )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.cuda.schedule_conv2d_NCHWc_int8),
                name="conv2d_NCHWc_int8.cuda",
            )
        else:
            raise RuntimeError("Unsupported conv2d layout {} for CUDA".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.depthwise_conv2d_nchw),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nchw),
                name="dpethwise_nchw.cuda",
            )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                wrap_topi_schedule(topi.cuda.schedule_depthwise_conv2d_nhwc),
                name="depthwise_conv2d_nhwc.cuda",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {}".format(layout))
    else:  # group_conv2d
        if layout == "NCHW":
            # TODO(@vinx13, @icemelon9): Use group_conv2d_NCHWc_int8 when dtype is int8/uint8.
            assert kernel_layout == "OIHW"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_nchw, has_groups=True),
                wrap_topi_schedule(topi.vivante.schedule_group_conv2d_nchw),
                name="group_conv2d_nchw.vivante",
            )
        elif layout == "NCHW4c" and data.dtype in ["int8", "uint8"]:
            assert kernel_layout == "OIHW4o4i"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.cuda.group_conv2d_NCHWc_int8, True),
                wrap_topi_schedule(topi.vivante.schedule_group_conv2d_NCHWc_int8),
                name="group_conv2d_NCHWc_int8.vivante",
            )
        else:
            raise RuntimeError("Unsupported group_conv2d layout {}".format(layout))
    return strategy


@schedule_reduce.register(["vivante"])
def schedule_reduce_vivante(attrs, outs, target):
    """schedule reduction ops for vivante"""
    with target:
        return topi.vivante.schedule_reduce(outs)


@schedule_injective.register(["vivante"])
def schedule_injective_vivante(attrs, outs, target):
    """schedule injective ops for vivante"""
    with target:
        return topi.vivante.schedule_injective(outs)


@schedule_concatenate.register(["vivante"])
def schedule_concatenate_vivante(attrs, outs, target):
    """schedule concatenate for vivante"""
    with target:
        return topi.vivante.schedule_injective(outs)


@schedule_roi_pool.register(["vivante"])
def schedule_roi_pool_vivante(attrs, outs, target):
    """schedule roi_pool for vivante"""
    with target:
        return topi.vivante.schedule_injective(outs)


@roi_align_strategy.register(["vivante"])
def roi_align_strategy_vivante(attrs, inputs, out_type, target):
    """roi_align vivante strategy"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_roi_align(topi.vision.rcnn.roi_align_nchw),
        wrap_topi_schedule(topi.vivante.schedule_injective),
        name="roi_align_nchw.vivante",
    )
    return strategy
