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
"""Internal utilities for parsing Python subset to TIR"""

import ast
import inspect
import logging
import sys
import numpy

import tvm.runtime
from tvm._ffi.base import numeric_types
from tvm.ir.container import Array

from tvm.tir import expr as _expr
from tvm.tir import stmt as _stmt
from tvm.te.tensor import Tensor


# pylint: disable=invalid-name
np_arg_types = tuple(list(numeric_types) + [numpy.ndarray])
tvm_arg_types = (Tensor, Array, _expr.Var, _expr.ConstExpr)
halide_imm_types = (_expr.IntImm, _expr.FloatImm)


def _internal_assert(cond, err):
    """Simplify the code segment like if not XXX then raise an error"""
    if not cond:
        raise ValueError(err)


# Useful constants. In avoid of runtime dependences, we use function calls to return them.
def make_nop():
    """Returns a 'no operation' node in HalideIR."""
    return _stmt.Evaluate(tvm.runtime.const(0, dtype="int32"))


def is_docstring(node):
    """Checks if a Python AST node is a docstring"""
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Str)


def _pruned_source(func):
    """Prune source code's extra leading spaces"""
    try:
        lines = inspect.getsource(func).split("\n")
        leading_space = len(lines[0]) - len(lines[0].lstrip(" "))
        lines = [line[leading_space:] for line in lines]
        return "\n".join(lines)
    except IOError as err:
        if sys.version_info[0] == 2 and str(err) == "could not get source code":
            logging.log(
                logging.CRITICAL,
                "This module is not fully operated under Python2... " "Please move to Python3!",
            )
            raise err

    if func.__name__ == "hybrid_psroipooling":
        return '''
@hybrid.script
def hybrid_psroipooling(data, rois, output_dim, group_size, spatial_scale):
    """PSROI pool operator.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    rois : tvm.Tensor
        2-D with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    output_dim : int
        The number of output's channel.

    group_size : int
        The width and height of output

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]
    """
    # dtype = rois.dtype
    num_rois = rois.shape[0]
    channel = data.shape[1]
    height = data.shape[2]
    width = data.shape[3]
    output = output_tensor((num_rois, output_dim, group_size, group_size), "float32")

    for n in range(num_rois):
        roi_start_w = float32(round(rois[n, 1]) * spatial_scale)
        roi_start_h = float32(round(rois[n, 2]) * spatial_scale)
        roi_end_w = float32(round((rois[n, 3] + 1.0)) * spatial_scale)
        roi_end_h = float32(round((rois[n, 4] + 1.0)) * spatial_scale)

        roi_height = max(roi_end_h - roi_start_h, 0.1)
        roi_width = max(roi_end_w - roi_start_w, 0.1)
        bin_size_h = roi_height / float32(group_size)
        bin_size_w = roi_width / float32(group_size)

        for ctop in range(output_dim):
            for ph in range(group_size):
                for pw in range(group_size):
                    hstart = int32(floor(float32(ph) * bin_size_h + roi_start_h))
                    wstart = int32(floor(float32(pw) * bin_size_w + roi_start_w))
                    hend = int32(ceil(float32((ph + 1)) * bin_size_h + roi_start_h))
                    wend = int32(ceil(float32((pw + 1)) * bin_size_w + roi_start_w))

                    hstart = min(max(hstart, 0), height)
                    hend = min(max(hend, 0), height)
                    wstart = min(max(wstart, 0), width)
                    wend = min(max(wend, 0), width)

                    c = (ctop * group_size + ph) * group_size + pw
                    out_sum = 0.0
                    for h in range(hend - hstart):
                        for w in range(wend - wstart):
                            out_sum = out_sum + data[0, c, h + hstart, w + wstart]

                    bin_area = (hend - hstart) * (wend - wstart)

                    if hstart < hend and wstart < wend:
                        output[n, ctop, ph, pw] = out_sum / float32(bin_area)
                    else:
                        output[n, ctop, ph, pw] = 0.0
    return output
                    '''
    if func.__name__ == "hybrid_segment_max":
        return '''
@hybrid.script
def hybrid_segment_max(data, segment_ids, length):
    """segment_max operator.

    Parameters
    ----------
    data: 1-D or 2-D Tensor.

    segment_ids: num_samples: 0-D. Number of independent samples to draw for each row slice.

    length: integer, length of output.
    """
    flag = 1
    if len(data.shape) == 1:
        output = output_tensor((length,), data.dtype)
        segment_length = segment_ids.shape[0]
        for i in range(length):
            flag = 1
            for j in range(segment_length):
                if segment_ids[j] == i:
                    if flag == 1:
                        output[i] = data[j]
                        flag = 0
                    if flag == 0:
                        if output[i] < data[j]:
                            output[i] = data[j]
            if flag == 1:
                output[i] = -3.4028235e38

    if len(data.shape) == 2:
        output = output_tensor((data.shape[0], length), "float32")
        segment_length = segment_ids.shape[0]
        for i in range(length):
            flag = 1
            for j in range(segment_length):
                if segment_ids[j] == i:
                    for k in range(data.shape[1]):
                        if flag == 1:
                            flag = 0
                            for c in range(data.shape[1]):
                                output[i, k] = data[j, c]
                        if flag == 0:
                            if output[i, k] < data[j, k]:
                                output[i, k] = data[j, k]
            if flag == 1:
                for k in range(data.shape[1]):
                    output[i, k] = -3.4028235e38
    return output
                    '''
    if func.__name__ == "hybrid_segment_min":
        return '''
@hybrid.script
def hybrid_segment_min(data, segment_ids, length):
    """segment_min operator.

    Parameters
    ----------
    data: 1-D or 2-D Tensor.

    segment_ids: num_samples: 0-D. Number of independent samples to draw for each row slice.

    length: integer, length of output.
    """
    flag = 1
    if len(data.shape) == 1:
        output = output_tensor((length,), data.dtype)
        segment_length = segment_ids.shape[0]
        for i in range(length):
            flag = 1
            for j in range(segment_length):
                if segment_ids[j] == i:
                    if flag == 1:
                        output[i] = data[j]
                        flag = 0
                    if flag == 0:
                        if output[i] > data[j]:
                            output[i] = data[j]
            if flag == 1:
                output[i] = 3.4028235e38
    if len(data.shape) == 2:
        output = output_tensor((data.shape[0], length), "float32")
        segment_length = segment_ids.shape[0]
        for i in range(length):
            flag = 1
            for j in range(segment_length):
                if segment_ids[j] == i:
                    for k in range(data.shape[1]):
                        if flag == 1:
                            flag = 0
                            for c in range(data.shape[1]):
                                output[i, k] = data[j, c]
                        if flag == 0:
                            if output[i, k] > data[j, k]:
                                output[i, k] = data[j, k]
            if flag == 1:
                for k in range(data.shape[1]):
                    output[i, k] = 3.4028235e38
    return output
                    '''
    if func.__name__ == "hybrid_segment_sum":
        return '''
@hybrid.script
def hybrid_segment_sum(data, segment_ids, length):
    """segment_sum operator.

    Parameters
    ----------
    data: 1-D or 2-D Tensor.

    segment_ids: num_samples: 0-D. Number of independent samples to draw for each row slice.

    length: integer, length of output.
    """

    if len(data.shape) == 1:
        output = output_tensor((length,), "float32")

        segment_length = segment_ids.shape[0]
        for i in range(length):
            output[i] = float32(0)
            for j in range(segment_length):
                if segment_ids[j] == i:
                    output[i] = output[i] + data[j]

    if len(data.shape) == 2:
        output = output_tensor((data.shape[0], length), "float32")
        segment_length = segment_ids.shape[0]
        for i in range(length):
            for k in range(data.shape[1]):
                output[i, k] = float32(0)
            for j in range(segment_length):
                if segment_ids[j] == i:
                    for k in range(data.shape[1]):
                        output[i, k] = output[i, k] + data[j, k]

    return output
                    '''
    if func.__name__ == "hybrid_segment_mean":
        return '''
@hybrid.script
def hybrid_segment_mean(data, segment_ids, length):
    """segment_mean operator.

    Parameters
    ----------
    data: 1-D or 2-D Tensor.

    segment_ids: num_samples: 0-D. Number of independent samples to draw for each row slice.

    length: integer, length of output.
    """
    count = float32(0)
    if len(data.shape) == 1:
        output = output_tensor((length,), "float32")
        segment_length = segment_ids.shape[0]
        for i in range(length):
            count = float32(0)
            output[i] = float32(0)
            for j in range(segment_length):
                if segment_ids[j] == i:
                    output[i] = output[i] + data[j]
                    count = count + float32(1)
            output[i] = output[i] / count

    if len(data.shape) == 2:
        output = output_tensor((data.shape[0], length), "float32")
        segment_length = segment_ids.shape[0]
        for i in range(length):
            count = float32(0)
            for k in range(data.shape[1]):
                output[i, k] = float32(0)
            for j in range(segment_length):
                if segment_ids[j] == i:
                    count = count + float32(1)
                    for k in range(data.shape[1]):
                        output[i, k] = output[i, k] + data[j, k]
            for k in range(data.shape[1]):
                output[i, k] = output[i, k] / count

    return output
                    '''
    if func.__name__ == "hybrid_segment_prod":
        return '''
@hybrid.script
def hybrid_segment_prod(data, segment_ids, length):
    """segment_prod operator.

    Parameters
    ----------
    data: 1-D or 2-D Tensor.

    segment_ids: num_samples: 0-D. Number of independent samples to draw for each row slice.

    length: integer, length of output.
    """
    if len(data.shape) == 1:
        output = output_tensor((length,), "float32")

        segment_length = segment_ids.shape[0]
        for i in range(length):
            output[i] = float32(1)
            for j in range(segment_length):
                if segment_ids[j] == i:
                    output[i] = output[i] * data[j]

    if len(data.shape) == 2:
        output = output_tensor((data.shape[0], length), "float32")
        segment_length = segment_ids.shape[0]
        for i in range(length):
            count = float32(0)
            for k in range(data.shape[1]):
                output[i, k] = float32(1)
            for j in range(segment_length):
                if segment_ids[j] == i:
                    count = count + float32(1)
                    for k in range(data.shape[1]):
                        output[i, k] = output[i, k] * data[j, k]

    return output
                    '''
    if func.__name__ == "hybrid_categorical":
        return '''
@hybrid.script
def hybrid_categorical(data):
    """categorical operator.

    Parameters
    ----------
    logits: 2-D Tensor with shape [batch_size, num_classes]. Each slice [i, :]
            represents the unnormalized log-probabilities for all classes.

    num_samples: 0-D. Number of independent samples to draw for each row slice.

    dtype: integer type to use for the output. Defaults to int64.
    """

    length = data.shape[0]
    output = output_tensor((length,), "int32")

    for i in range(length):
        output[data[i]] = i

    return output
                '''
    if func.__name__ == "hybrid_extract_image_patches":
        return '''
@hybrid.script
def hybrid_extract_image_patches(data, ksizes, strides, rates, padding):
    """extract_image_patches operator.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    ksizes : tvm.array
        1-D with shape [kernel_size_h, kernel_size_w].

    strides : tvm.array
        1-d with shape [stride_h, stride_w].

    rates : tvm.array
        just list dilated.
        1-d with shape [dilated_h, dilated_w].

    padding : tvm.array
        1-d with shape [pad_l, pad_t, pad_r, pad_d]
    """
    # dtype = rois.dtype
    batch, channel, in_height, in_width = data.shape
    k_h, k_w = ksizes
    stride_h, stride_w = strides
    dilated_h, dilated_w = rates
    pad_l, pad_t, pad_r, pad_d = padding

    # output shape
    out_channel = k_h * k_w * channel

    output = output_tensor((batch, out_channel, out_height, out_width), "float32")

    for b in range(batch):
        for c in range(channel):
            for out_y in range(out_height):
                for out_x in range(out_width):
                    in_x_origin = (out_x * stride_width) - pad_left
                    in_y_origin = (out_y * stride_height) - pad_top
                    for filter_y in range(filter_height):
                        for filter_x in range(filter_width):
                            in_x = in_x_origin + dilation_width_factor * filter_x
                            in_y = in_y_origin + dilation_height_factor * filter_y
                            o_x = out_x + filter_x
                            o_y = out_y + filter_y
                            if (
                                (in_x >= 0)
                                and (in_x < input_width)
                                and (in_y >= 0)
                                and (in_y < input_height)
                            ):
                                output[b, c, o_y, o_x] = data[batch, c, in_y, in_x]

    return output
                '''
    if func.__name__ == "hybrid_invert_permutation":
        return '''
@hybrid.script
def hybrid_invert_permutation(data):
    """invert_permutation operator.

    Parameters
    ----------
    data : tvm.Tensor
        Must be one of the following types: int32, int64. 1-D.
    """

    length = data.shape[0]
    output = output_tensor((length,), "int32")

    for i in range(length):
        output[data[i]] = i

    return output
                '''
    if func.__name__ == "hybrid_rearrange_box_out":
        return '''
@hybrid.script
def hybrid_rearrange_box_out(data, one, batch_size, num_anchors):
    """Hybrid routine to rearrange nms output to
    move all valid entries to top.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    one: tvm.tir.const
        Constant one with the same dtype as data.

    batch_size: tvm.tir.IntImm or tvm.tir.Var
        Batch size. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    num_anchors: tvm.tir.IntImm or tvm.tir.Var
        Number of anchors.

    Returns
    -------
    output : tvm.te.Tensor or numpy NDArray
        Transformed NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].
    """
    elem_length = data.shape[2]
    output = output_tensor((batch_size, num_anchors, elem_length), data.dtype)

    for i in parallel(batch_size):
        valid_idx = 0
        for j in range(num_anchors):
            if data[i, j, 0] >= 0:
                for k in range(elem_length):
                    output[i, valid_idx, k] = data[i, j, k]
                valid_idx += 1
            if j >= valid_idx:
                for k in range(elem_length):
                    output[i, j, k] = -one
    return output
                '''
    if func.__name__ == "hybrid_unpooling":
        return '''
@hybrid.script
def hybrid_unpooling(data, mask_data, scale_h=2, scale_w=2, pad_out_h=0, pad_out_w=0):
    """Upsampling.

    This operator takes data as input and does 2D scaling to the given scale factor.
    In the default case, where the data_layout is `NCHW`
    with data of shape (n, c, h, w)
    out will have a shape (n, c, h*scale_h, w*scale_w)

    Parameters
    ----------
    data : tvm.relay.Expr
        The input data to the operator.

    scale_h : tvm.relay.Expr
        The scale factor for height upsampling.

    scale_w : tvm.relay.Expr
        The scale factor for width upsampling.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """
    numb_ = int32(data.shape[0])
    channels_ = int32(data.shape[1])
    height_ = int32(data.shape[2])
    width_ = int32(data.shape[3])

    upsample_h_ = int32(height_ * scale_h - pad_out_h)
    upsample_w_ = int32(width_ * scale_w - pad_out_w)

    out_data = output_tensor((numb_, channels_, upsample_h_, upsample_w_), "float32")

    for n in range(numb_):
        for c in range(channels_):
            for i in range(upsample_h_):
                for j in range(upsample_w_):
                    out_data[n, c, i, j] = 0.0

    for n in range(numb_):
        for c in range(channels_):
            for i in range(height_):
                for j in range(width_):
                    idx = mask_data[n, c, i, j]
                    if idx < upsample_h_ * upsample_w_:
                        o_h = int32(idx / float32(upsample_w_))
                        o_w = int32(idx - o_h * upsample_w_)
                        out_data[n, c, o_h, o_w] = float32(data[n, c, i, j])

    return out_data
                '''


def replace_io(body, rmap):
    """Replacing tensors usage according to the dict given"""
    # pylint: disable=import-outside-toplevel
    from tvm.tir import stmt_functor

    def replace(op):
        if isinstance(op, _stmt.ProducerStore) and op.producer.op in rmap.keys():
            buf = rmap[op.producer.op]
            return _stmt.ProducerStore(buf, op.value, op.indices)
        if isinstance(op, _expr.ProducerLoad) and op.producer.op in rmap.keys():
            buf = rmap[op.producer.op]
            return _expr.ProducerLoad(buf, op.indices)
        return None

    return stmt_functor.ir_transform(body, None, replace, ["tir.ProducerStore", "tir.ProducerLoad"])


def _is_tvm_arg_types(args):
    """Determine a list of element is either a list of tvm arguments of a list of numpy arguments.
    If neither is true, raise a value error."""
    if isinstance(args[0], tvm_arg_types):
        for elem in args[1:]:
            _internal_assert(
                isinstance(elem, tvm_arg_types),
                "Expecting a Var, Tensor or ConstExpr instance but %s get!" % str(type(elem)),
            )
        return True

    _internal_assert(
        isinstance(args[0], np_arg_types), "Expect a numpy type but %s get!" % str(type(args[0]))
    )
    for elem in args[1:]:
        _internal_assert(
            isinstance(elem, np_arg_types), "Expect a numpy type but %s get!" % str(type(elem))
        )
    return False
