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
# pylint: disable=invalid-name
"""QNN dialect operators."""

from __future__ import absolute_import as _abs
import tvm
from tvm.relay.expr import Tuple, TupleWrapper
from tvm.relay.op.nn.util import get_pad_tuple2d
from . import _make


def requantize(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=-1,
    rounding="UPWARD",
    out_dtype="int8",
):
    r"""Requantized operator.

    The requantize operator converts one quantized tensor representation to
    another quantized tensor representation. For the output tensor, we are
    provided with output scale and zero point. The computation is as follows

    Q_output = zp_output +  (scale_input)/(scale_output) * (Q_input - zp_input)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    input_scale: tvm.relay.Expr
        The quantization scale for the input tensor.

    input_zero_point: tvm.relay.Expr
        The zero point of the input tensor.

    output_scale: tvm.relay.Expr
        The quantization scale for the output tensor.

    output_zero_point: tvm.relay.Expr
        The zero point of the output tensor.

    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.

    rounding : string, optional
        Defines the rounding direction when the value is midway between two
        representable values.

    out_dtype : str, optional
        Specifies the output data type.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.requantize(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        axis,
        rounding,
        out_dtype,
    )


def quantize(data, output_scale, output_zero_point, axis=-1, out_dtype="int8"):
    r"""Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : tvm.relay.Expr
        The output zero_point.
    output_scale : tvm.relay.Expr
        The output scale.
    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8, int32]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.quantize(data, output_scale, output_zero_point, axis, out_dtype)


def dequantize(data, input_scale, input_zero_point, axis=-1):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point : tvm.relay.Expr
        The input zero_point.
    input_scale : tvm.relay.Expr
        The input scale.
    axis : int
        The channel axis for quantization. Default value is -1 which corresponds to the last axis.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dequantize(data, input_scale, input_zero_point, axis)


def csinn_deinit(data, input_scale, input_zero_point):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point : int
        The output zero_point.
    input_scale : float
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.CSINNDeinit(data, input_scale, input_zero_point)


def csinn_init(data, output_scale, output_zero_point, out_dtype="uint8"):
    r"""init op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : int
        The output zero_point.
    output_scale : float
        The output scale.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSINNInit(data, output_scale, output_zero_point, out_dtype)


def concatenate(data, input_scales, input_zero_points, output_scale, output_zero_point, axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr], TupleWrapper[relay.Expr])
        The list of quantized tensors.

    input_scales : List[relay.Expr]
        The list of scales of input quantized tensors.

    input_zero_points : List[relay.Expr]
        The list of zero points of input quantized tensors.

    output_scale : relay.Expr
        The scale of the output quantized tensor.

    output_zero_point : relay.Expr
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    if isinstance(data, (list, tuple)):
        data = Tuple(data)
    elif isinstance(data, TupleWrapper):
        data = data.tuple_value
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")
    input_scales = list(input_scales)
    input_zero_points = list(input_zero_points)

    return _make.concatenate(
        data, Tuple(input_scales), Tuple(input_zero_points), output_scale, output_zero_point, axis
    )


def conv2d(
    data,
    kernel,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    kernel_size,
    channels,
    strides=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    groups=1,
    data_layout="NCHW",
    kernel_layout="OIHW",
    out_layout="",
    out_dtype="int32",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: tvm.relay.Expr
           The zero point of the data distribution.

    kernel_zero_point: tvm.relay.Expr
           The zero point of the quantized_kernel distribution.

    input_scale: tvm.relay.Expr
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: tvm.relay.Expr
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_size : tuple of int
        The spatial width and height of the convolution kernel.

    channels : int
        Number of output channels of this convolution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    padding = get_pad_tuple2d(padding)
    return _make.conv2d(
        data,
        kernel,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
    )


def add(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.add(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )


def dense(
    data,
    weight,
    input_zero_point,
    kernel_zero_point,
    input_scale,
    kernel_scale,
    units,
    out_dtype="int32",
):
    """Qnn Dense operator.
    Applies a quantized linear transformation

     .. math::

     `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    input_zero_point: tvm.relay.Expr
        The input zero point.
    kernel_zero_point: tvm.relay.Expr
        The kernel zero point.
    input_scale: tvm.relay.Expr
        The scale for the input tensor.
    kernel_scale: tvm.relay.Expr
        The scale for the weight tensor. The scale for the weight tensor is
        stored for access to this during relay. This information is not
        needed in the pass pipeline after qnn.conv2d is lowered to the
        sequence of steps as in nn.conv2d. See also input_scale in Requantize.
    units : int
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.dense(
        data,
        weight,
        input_zero_point,
        kernel_zero_point,
        input_scale,
        kernel_scale,
        units,
        out_dtype,
    )


def mul(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Quantized multiplication with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.mul(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )


def subtract(
    lhs, rhs, lhs_scale, lhs_zero_point, rhs_scale, rhs_zero_point, output_scale, output_zero_point
):
    """Quantized subtraction with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: relay.Expr
        The scale of the lhs quantized expr.

    lhs_zero_point: relay.Expr
       The zero point of lhs quantized expr.

    rhs_scale: relay.Expr
        The scale of the rhs quantized expr.

    rhs_zero_point: relay.Expr
       The zero point of rhs quantized expr.

    output_scale: relay.Expr
        The scale of the output quantized expr.

    output_zero_point: relay.Expr
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.subtract(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )


def csi_concatenate(
    data,
    input_scales,
    input_zero_points,
    output_scale,
    output_zero_point,
    axis,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        The list of quantized tensors.

    input_scales : List[float32]
        The list of scales of input quantized tensors.

    input_zero_points : List[int32]
        The list of zero points of input quantized tensors.

    output_scale : float32
        The scale of the output quantized tensor.

    output_zero_point : int32
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    data = list(data)
    if not data:
        raise ValueError("relay.concatenate requires data to be non-empty.")
    if not isinstance(axis, int):
        raise ValueError("For now, we only support integer axis")

    input_scales = [
        tvm.tir.FloatImm("float64", x) if isinstance(x, float) else x for x in input_scales
    ]
    input_zero_points = [
        tvm.tir.FloatImm("int32", x) if isinstance(x, int) else x for x in input_zero_points
    ]
    return _make.CSIConcatenate(
        Tuple(data),
        input_scales,
        input_zero_points,
        output_scale,
        output_zero_point,
        axis,
        max_values,
        min_values,
        layer_name,
    )


def csi_quantize(
    data,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantize op
    This operator takes float32 as input and produces quantized int8 or unit8 as output.
    The input tensor can be of any shape. The output shape is the same as input shape.

    Q_output = clamp((round(input_tensor/output_scale) + output_zero_point),
                     out_dtype::min,
                     out_dtype::max)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be quantized. Can be of type float32.
    output_zero_point : int
        The output zero_point.
    output_scale : float
        The output scale.
    out_dtype : str, optional
        The data type of the input tensor. Can be [int8, uint8]
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.CSIQuantize(
        data, output_scale, output_zero_point, out_dtype, max_values, min_values, layer_name
    )


def csi_dequantize(
    data, input_scale, input_zero_point, max_values=tuple(), min_values=tuple(), layer_name=""
):
    r"""Dequantize op
    This operator takes quantized int8 and unit8 as input and produces
    dequantized float32 as output. The output shape is the same as input shape. The input
    tensor can be of any shape.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input tensor to be dequantized. Can be of type [int8, uint8].
    input_zero_point : int
        The output zero_point.
    input_scale : float
        The output scale.
    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    return _make.CSIDequantize(
        data, input_scale, input_zero_point, max_values, min_values, layer_name
    )


def csi_conv2d(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2D(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_conv2d_channel(
    data,
    weight,
    bias,
    kernel_scale,
    kernel_zero_point,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DChannel(
        data,
        weight,
        bias,
        kernel_scale,
        kernel_zero_point,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_conv2d_relu_channel(
    data,
    weight,
    bias,
    kernel_scale,
    kernel_zero_point,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DReluChannel(
        data,
        weight,
        bias,
        kernel_scale,
        kernel_zero_point,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_conv2d_relu6_channel(
    data,
    weight,
    bias,
    kernel_scale,
    kernel_zero_point,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DRelu6Channel(
        data,
        weight,
        bias,
        kernel_scale,
        kernel_zero_point,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_conv3d(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 3D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv3D(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_dilation2d(
    data,
    weight,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilations,
    data_layout,
    kernel_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized dilation2d.
    Computes grayscale dilation of 4D input and 3D filter.
    - **data**: This depends on the `layout` parameter. Input is 4D array of shape
                (batch_size, in_channels, height, width) if `layout` is `NCHW`.
    - **weight**: (in_channels, height, width)
    - **out**:  This depends on the `layout` parameter. Output is 4D array of shape
                (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
    """
    return _make.CSIDilation2D(
        data,
        weight,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilations,
        data_layout,
        kernel_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_conv2d_relu(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DRelu(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_conv2d_relu6(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D convolution.

    This operator convolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIConv2DRelu6(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_deconv2d(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    output_padding,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 2D deconvolution.

    This operator deconvolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDeConv2D(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_deconv3d(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    dilation,
    groups,
    channels,
    kernel_size,
    data_layout,
    kernel_layout,
    out_layout,
    output_padding,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""Quantized 3D deconvolution.

    This operator deconvolves quantized data with quantized kernel. The scale of
    the output quantized tensor is the product of the kernel_scale and
    input_scale of the input quantized tensors. The zero point of the output
    quantized tensor is 0. By default, the dtype of output is int32. Please also
    refer to Requantize operator to understand how to scale back the int32
    output to (u)int8.

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    kernel : tvm.relay.Expr
        The kernel expressions.

    input_zero_point: int
           The zero point of the data distribution.

    input_scale: float
           The scale for the input tensor. The scale for the input tensor is
           stored purely for convenience here. See more commentary below.

    kernel_scale: float
           The scale for the weight tensor. The scale for the weight tensor is
           stored for access to this during relay. This information is not
           needed in the pass pipeline after qnn.conv2d is lowered to the
           sequence of steps as in nn.conv2d. See also input_scale in Requantize.

    kernel_zero_point: int
           The zero point of the quantized_kernel distribution.

    strides : tuple of int, optional
        The strides of convolution.

    padding : tuple of int, optional
        The padding of convolution on both sides of inputs before convolution.

    dilation : tuple of int, optional
        Specifies the dilation rate to be used for dilated convolution.

    groups : int, optional
        Number of groups for grouped convolution.

    channels : int, optional
        Number of output channels of this convolution.

    kernel_size : tuple of int, optional
        The spatial of the convolution kernel.

    data_layout : str, optional
        Layout of the input.

    kernel_layout : str, optional
        Layout of the kernel.

    out_layout : str, optional
        Layout of the output, by default, out_layout is the same as data_layout

    out_dtype : str, optional
        Specifies the output data type for mixed precision conv2d.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDeConv3D(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        dilation,
        groups,
        channels,
        kernel_size,
        data_layout,
        kernel_layout,
        out_layout,
        output_padding,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_add(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAdd(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_subtract(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized subtract with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISubtract(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_bias_add(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized addition with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIBiasAdd(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_mul(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized multiplication with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    input_scale: float
        The scale of the input quantized expr.

    input_zero_point: int
       The zero point of input quantized expr.

    kernel_scale: float
        The scale of the kernel quantized expr.

    kernel_zero_point: int
       The zero point of kernel quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMul(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_div(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized divide with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    input_scale: float
        The scale of the input quantized expr.

    input_zero_point: int
       The zero point of input quantized expr.

    kernel_scale: float
        The scale of the kernel quantized expr.

    kernel_zero_point: int
       The zero point of kernel quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIDiv(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_power(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized power with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    input_scale: float
        The scale of the input quantized expr.

    input_zero_point: int
       The zero point of input quantized expr.

    kernel_scale: float
        The scale of the kernel quantized expr.

    kernel_zero_point: int
       The zero point of kernel quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIPower(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_mod(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized power with numpy-style broadcasting.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    weight : relay.Expr
        The quantized weight data.

    bias : relay.Expr
        The quantized bias data.

    input_scale: float
        The scale of the input quantized expr.

    input_zero_point: int
       The zero point of input quantized expr.

    kernel_scale: float
        The scale of the kernel quantized expr.

    kernel_zero_point: int
       The zero point of kernel quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMod(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_dense(
    data,
    weight,
    bias,
    input_scale,
    input_zero_point,
    kernel_scale,
    kernel_zero_point,
    output_scale,
    output_zero_point,
    units,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Qnn Dense operator.
    Applies a quantized linear transformation

     .. math::

     `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    input_zero_point: int
        The input zero point.
    kernel_zero_point: int
        The kernel zero point.
    input_scale: float
        The scale for the input tensor.
    kernel_scale: float
        The scale for the weight tensor. The scale for the weight tensor is
        stored for access to this during relay. This information is not
        needed in the pass pipeline after qnn.conv2d is lowered to the
        sequence of steps as in nn.conv2d. See also input_scale in Requantize.
    units : int, optional
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    return _make.CSIDense(
        data,
        weight,
        bias,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        output_scale,
        output_zero_point,
        units,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_sin(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation sin.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISin(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_cos(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation cos.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICos(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_tan(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation tan.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITan(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_asin(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation asin.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAsin(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_acos(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation acos.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAcos(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_atan(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation atan.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAtan(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_sinh(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation sinh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISinh(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_cosh(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation cosh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICosh(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_tanh(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation tanh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITanh(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_asinh(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation asinh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAsinh(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_acosh(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation acosh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAcosh(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_atanh(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation atanh.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAtanh(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_relu(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRelu(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_leaky_relu(
    data,
    alpha,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.alpha

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILeakyRelu(
        data,
        alpha,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_prelu(
    data,
    alpha,
    axis,
    input_scale,
    input_zero_point,
    alpha_scale,
    alpha_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    alpha : relay.Expr
        The quantized alpha.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIPRelu(
        data,
        alpha,
        axis,
        input_scale,
        input_zero_point,
        alpha_scale,
        alpha_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_relu6(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation relu.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRelu6(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_max_pool(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    strides,
    padding,
    pool_size,
    ceil_mode,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation max pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        list(strides),
        list(padding),
        list(pool_size),
        ceil_mode,
        str(layout),
        max_values,
        min_values,
        layer_name,
    )


def csi_max_pool2d_with_argmax(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    strides,
    padding,
    pool_size,
    ceil_mode,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation max pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool2dWithArgmax(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        strides,
        padding,
        pool_size,
        ceil_mode,
        str(layout),
        max_values,
        min_values,
        layer_name,
    )


def csi_max_pool2d_locat(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    strides,
    padding,
    pool_size,
    ceil_mode,
    out_dtype,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation max pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool2DLocat(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        strides,
        padding,
        pool_size,
        ceil_mode,
        out_dtype,
        layout,
        max_values,
        min_values,
        layer_name,
    )


def csi_avg_pool(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    strides,
    padding,
    pool_size,
    ceil_mode,
    count_include_pad,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation average pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAvgPool(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        strides,
        padding,
        pool_size,
        ceil_mode,
        count_include_pad,
        layout,
        max_values,
        min_values,
        layer_name,
    )


def csi_avg_pool3d(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    strides,
    padding,
    pool_size,
    ceil_mode,
    count_include_pad,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation average pooling.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIAvgPool3D(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        strides,
        padding,
        pool_size,
        ceil_mode,
        count_include_pad,
        layout,
        max_values,
        min_values,
        layer_name,
    )


def csi_max_pool3d(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    strides,
    padding,
    pool_size,
    ceil_mode,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation max pooling 3D.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaxPool3D(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        strides,
        padding,
        pool_size,
        ceil_mode,
        layout,
        max_values,
        min_values,
        layer_name,
    )


def csi_reshepe(
    data,
    newshape,
    reverse,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation reshepe.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIReshape(
        data,
        newshape,
        reverse,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_proposal(
    cls_prob,
    bbox_pred,
    im_info,
    scales,
    ratios,
    feature_stride,
    threshold,
    rpn_pre_nms_top_n,
    rpn_post_nms_top_n,
    rpn_min_size,
    iou_loss,
    input_scales,
    input_zero_points,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized proposal.

    Parameters
    ----------
    cls_prob: 4-D with shape [batch, 2 * num_anchors, height, width].
    bbox_pred: 4-D with shape [batch, 4 * num_anchors, height, width].
    im_info: 2-D with shape [batch, 3].

    Returns
    -------
    result : relay.Expr
        2-D with shape [batch * rpn_post_nms_top_n, 5].

    """
    return _make.CSIProposal(
        cls_prob,
        bbox_pred,
        im_info,
        scales,
        ratios,
        feature_stride,
        threshold,
        rpn_pre_nms_top_n,
        rpn_post_nms_top_n,
        rpn_min_size,
        iou_loss,
        input_scales,
        input_zero_points,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_psroipooling(
    cls_prob,
    roi,
    spatial_scale,
    output_dim,
    group_size,
    input_scales,
    input_zero_points,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized psroipooling.

    Parameters
    ----------


    Returns
    -------
    result : relay.Expr


    """
    return _make.CSIPSROIPooling(
        cls_prob,
        roi,
        spatial_scale,
        output_dim,
        group_size,
        input_scales,
        input_zero_points,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_roipooling(
    data,
    roi,
    pooled_size,
    spatial_scale,
    input_scales,
    input_zero_points,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized roipooling.

    Parameters
    ----------


    Returns
    -------
    result : relay.Expr


    """
    return _make.CSIROIPooling(
        data,
        roi,
        pooled_size,
        spatial_scale,
        input_scales,
        input_zero_points,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_unpooling(
    data,
    mask,
    scales,
    out_padding,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized unpooling.

    Parameters
    ----------


    Returns
    -------
    result : relay.Expr


    """
    return _make.CSIUnPooling(
        data,
        mask,
        scales,
        out_padding,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        layout,
        max_values,
        min_values,
        layer_name,
    )


def csi_upsampling(
    data,
    scale_h,
    scale_w,
    align_corners,
    method,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    layout,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized upsampling.

    Parameters
    ----------


    Returns
    -------
    result : relay.Expr


    """

    return _make.CSIUpSampling(
        data,
        scale_h,
        scale_w,
        align_corners,
        method,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        layout,
        max_values,
        min_values,
        layer_name,
    )


def csi_flatten(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation flatten.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFlatten(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_sigmoid(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation flatten.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISigmoid(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_transpose(
    data,
    axes,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation transpose.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    newshape : tuple of int, optional
        The shape of output tensor.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITranspose(
        data,
        axes,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_softmax(
    data,
    axis,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation softmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISoftMax(
        data,
        axis,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_reverse(
    data,
    axis,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized reverse.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIReverse(
        data,
        axis,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_log_softmax(
    data,
    axis,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation softmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILogSoftMax(
        data,
        axis,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_lrn(
    data,
    size,
    axis,
    alpha,
    beta,
    bias,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    size : int
        The size of the local region to be considered for normalization.

    bias : int
        The offset parameter to avoid division by 0.

    alpha : float
        The scaling parameter.

    beta : float
        The exponent parameter.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.
    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILRN(
        data,
        size,
        axis,
        alpha,
        beta,
        bias,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_global_avgpool(
    data,
    layout,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized global average pooling layer.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIGlobalAvgPool(
        data,
        layout,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_global_maxpool(
    data,
    layout,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized global max pooling layer.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIGlobalMaxPool(
        data,
        layout,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_mean(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMean(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_prod(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized prod.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIProd(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_max(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized Max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMax(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_min(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized Max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMin(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_sum(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation sum.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISum(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_pad(
    data,
    pad_width,
    pad_value,
    pad_mode,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized pad.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIPad(
        data,
        pad_width,
        pad_value,
        pad_mode,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_squeeze(
    data,
    axis,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISqueeze(
        data,
        axis,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_reshape(
    data,
    newshape,
    reverse,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized activation lrn.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIReshape(
        data,
        newshape,
        reverse,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_batch_norm(
    data,
    gamma,
    beta,
    moving_mean,
    moving_var,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis=1,
    epsilon=1e-5,
    center=True,
    scale=True,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    r"""
    Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalizes the input at each batch, i.e. applies a transformation
    that maintains the mean activation close to 0 and the activation
    standard deviation close to 1.

    .. math::

        data\_mean[i] = mean(data[:,i,:,...]) \\
        data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

        out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}}
            * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated by

    .. code:: python

        moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
        moving_var = moving_var * momentum + data_var * (1 - momentum)

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is 1.
    Specifying -1 sets the channel axis to be the last item in the input shape.

    .. note::

        This operator can be optimized away for inference.

    Parameters
    ----------
    data : tvm.relay.Expr
        Input to which batch_norm will be applied.

    gamma : tvm.relay.Expr
        The gamma scale factor.

    beta : tvm.relay.Expr
        The beta offset factor.

    moving_mean : tvm.relay.Expr
        Running mean of input,

    moving_var : tvm.relay.Expr
        Running variance of input.

    axis : int, optional, default=1
        Specify along which shape axis the channel is specified.

    epsilon : double, optional, default=1e-5
        Small float added to variance to avoid dividing by zero.

    center : boolean, optional, default=True
        If True, add offset of beta to normalized tensor, If False,
        beta is ignored.

    scale : boolean, optional, default=True
        If true, multiply by gamma. If False, gamma is not used.
        When the next layer is piecewise linear (also e.g. nn.relu),
        this can be disabled since the scaling will be done by the next layer.

    Returns
    -------
    result : relay.Tuple([tvm.relay.Expr, tvm.relay.Expr, tvm.relay.Expr])
        Tuple of normed data (same shape as input),
        new running mean (k-length vector),
        and new running variance (k-length vector)
    """
    return _make.csi_batch_norm(
        data,
        gamma,
        beta,
        moving_mean,
        moving_var,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        axis,
        epsilon,
        center,
        scale,
        max_values,
        min_values,
        layer_name,
    )


def csi_strided_slice(
    data,
    begin,
    end,
    strides,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Strided slice of an array.

    Parameters
    ----------
    data : relay.Expr
        The source array to be sliced.

    begin: list of int
        The indices to begin with in the slicing.

    end: list of int
        Indices indicating end of the slice.

    strides: list of int, optional
        Specifies the stride values, it can be negative in that case,
        the input tensor will be reversed in that particular axis.

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    return _make.CSIStridedSlice(
        data,
        list(begin),
        list(end),
        list(strides),
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_split(
    data,
    indices_or_sections,
    axis,
    input_scale,
    input_zero_point,
    output_scales,
    output_zero_points,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Split input tensor along axis by sections or indices.

    If indices_or_sections is an integer, the input will be divided equally
    along given axis. If such a split is not possible, an error is raised.

    If indices_or_sections is a tuple of sorted integers,
    the entries indicate where along axis the array is split.

    Parameters
    ----------
    data : relay.Expr
        The source array.

    indices_or_sections : int or tuple of int
        Indices or sections to split into. Accepts an int or a tuple

    axis : int, optional
        The axis over which to split.

    Returns
    -------
    ret : relay.Tuple([relay.Expr, relay.Expr])
        The computed result.
    """
    return _make.CSISplit(
        data,
        indices_or_sections,
        axis,
        input_scale,
        input_zero_point,
        output_scales,
        output_zero_points,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_variance(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Computes the variance of data over given axes.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a variance operation is performed.
        The default, axis=None, will compute the variance of all elements in the input array.
        If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.CSIVariance(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_exp(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Take exponetial of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSIExp(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_segment_max(
    data,
    ids,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    length,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentMax(
        data,
        ids,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        length,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_segment_min(
    data,
    ids,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    length,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentMin(
        data,
        ids,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        length,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_segment_mean(
    data,
    ids,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    length,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentMean(
        data,
        ids,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        length,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_segment_prod(
    data,
    ids,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    length,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentProd(
        data,
        ids,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        length,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_segment_sum(
    data,
    ids,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    length,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized segment max.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    ids : relay.Expr
        The index.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISegmentSum(
        data,
        ids,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        length,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_log(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Take log of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSILog(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_negative(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Take negative of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSINegative(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_abs(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Take abs of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSIAbs(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_expand_dims(
    data,
    axis,
    num_newaxis,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Take abs of input x.

    Parameters
    ----------
    data : PrimExpr
        Input argument.
    axis : int

    Returns
    -------
    y : PrimExpr
        The result.
    """
    return _make.CSIExpandDims(
        data,
        axis,
        num_newaxis,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_argmax(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized argmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIArgmax(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_argmin(
    data,
    axis,
    keepdims,
    exclude,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized argmax.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIArgmin(
        data,
        axis,
        keepdims,
        exclude,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_broadcast_to(
    data,
    shape,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized broadcast_to.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIBroadCastTo(
        data,
        shape,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_cast(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized cast.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICast(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_ceil(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized ceil.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICeil(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_floor(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized floor.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFloor(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_clip(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    a_min,
    a_max,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized clip.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIClip(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        a_min,
        a_max,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_erf(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized Erf.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIErf(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_round(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized round.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRound(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_maximum(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized maximun with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMaximum(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_floor_div(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized floor_div with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFloorDiv(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_floor_mod(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized floor_mod with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFloorMod(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_left_shift(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized left_shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSILeftShift(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_right_shift(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized right_shift with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIRightShift(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_minimum(
    lhs,
    rhs,
    lhs_scale,
    lhs_zero_point,
    rhs_scale,
    rhs_zero_point,
    output_scale,
    output_zero_point,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized minimun with numpy-style broadcasting.

    Parameters
    ----------
    lhs : relay.Expr
        The left hand side quantized input data.

    rhs : relay.Expr
        The right hand side quantized input data.

    lhs_scale: float
        The scale of the lhs quantized expr.

    lhs_zero_point: int
       The zero point of lhs quantized expr.

    rhs_scale: float
        The scale of the rhs quantized expr.

    rhs_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIMinimum(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
        max_values,
        min_values,
        layer_name,
    )


def csi_crop_resize(
    data,
    boxes,
    box_indices,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    crop_size,
    layout,
    method,
    extrapolation_value,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized crop_resize with numpy-style broadcasting.

    Parameters
    ----------

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSICropResize(
        data,
        boxes,
        box_indices,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        crop_size,
        layout,
        method,
        extrapolation_value,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_depth_to_space(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    block_size,
    layout,
    mode,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized crop_resize with numpy-style broadcasting.

    Parameters
    ----------

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIDepthToSpace(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        block_size,
        layout,
        mode,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_space_to_depth(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    block_size,
    layout,
    mode,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized crop_resize with numpy-style broadcasting.

    Parameters
    ----------

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISpaceToDepth(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        block_size,
        layout,
        mode,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_sqrt(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized sqrt.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISqrt(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_sign(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized sign.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSISign(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_full(
    data,
    shape,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized fill.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIFull(
        data,
        shape,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_take(
    data,
    indices,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    axis,
    mode,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized Take.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITake(
        data,
        indices,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        axis,
        mode,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_tile(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    reps,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized tile.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITile(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        reps,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_unravel_index(
    data,
    shape,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized unravel_index.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSIUnRavelIndex(
        data,
        shape,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )


def csi_topk(
    data,
    input_scale,
    input_zero_point,
    output_scale,
    output_zero_point,
    k,
    axis,
    ret_type,
    is_ascend,
    dtype,
    out_dtype,
    max_values=tuple(),
    min_values=tuple(),
    layer_name="",
):
    """Quantized Take.

    Parameters
    ----------
    data : relay.Expr
        The quantized input data.

    input_scale: float
        The scale of the rhs quantized expr.

    input_zero_point: int
       The zero point of rhs quantized expr.

    output_scale: float
        The scale of the output quantized expr.

    output_zero_point: int
       The zero point of output quantized expr.

    out_dtype : str
        Specifies the output data type for mixed precision dense can be uint8.

    Returns
    -------
    result : relay.Expr
        The computed result.

    """
    return _make.CSITopK(
        data,
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
        k,
        axis,
        ret_type,
        is_ascend,
        dtype,
        out_dtype,
        max_values,
        min_values,
        layer_name,
    )
