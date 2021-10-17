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
# pylint: disable=invalid-name, too-many-nested-blocks
"""segment operator"""
import tvm
from tvm.te import hybrid


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


@tvm.target.generic_func
def segment_max(data, segment_ids, length):

    output = hybrid_segment_max(data, segment_ids, tvm.tir.const(length, "int32"))
    return output


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


@tvm.target.generic_func
def segment_min(data, segment_ids, length):

    output = hybrid_segment_min(data, segment_ids, tvm.tir.const(length, "int32"))
    return output


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


@tvm.target.generic_func
def segment_sum(data, segment_ids, length):

    output = hybrid_segment_sum(data, segment_ids, tvm.tir.const(length, "int32"))
    return output


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


@tvm.target.generic_func
def segment_mean(data, segment_ids, length):

    output = hybrid_segment_mean(data, segment_ids, tvm.tir.const(length, "int32"))
    return output


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


@tvm.target.generic_func
def segment_prod(data, segment_ids, length):

    output = hybrid_segment_prod(data, segment_ids, tvm.tir.const(length, "int32"))
    return output
