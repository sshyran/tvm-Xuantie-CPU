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
"""Non-maximum suppression operations."""
from . import _make


def segment_max(data, segment_ids, length):
    """Segment Max operator.

    Parameters
    ----------
    data: A `Tensor`. Must be one of the following types:
     `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`,
      `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.

    segment_ids: A `Tensor`. Must be one of the following types:
     `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

    length: 'Int'. The length of output.

    Returns
    -------
    A `Tensor`. Has the same type as `data`.
    """
    return _make.segment_max(data, segment_ids, length)


def segment_sum(data, segment_ids, length):
    """Segment Sum operator.

    Parameters
    ----------
    data: A `Tensor`. Must be one of the following types:
     `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`,
      `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.

    segment_ids: A `Tensor`. Must be one of the following types:
     `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

    length: 'Int'. The length of output.

    Returns
    -------
    A `Tensor`. Has the same type as `data`.
    """
    return _make.segment_sum(data, segment_ids, length)


def segment_mean(data, segment_ids, length):
    """Segment Mean operator.

    Parameters
    ----------
    data: A `Tensor`. Must be one of the following types:
     `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`,
      `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.

    segment_ids: A `Tensor`. Must be one of the following types:
     `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

    length: 'Int'. The length of output.

    Returns
    -------
    A `Tensor`. Has the same type as `data`.
    """
    return _make.segment_mean(data, segment_ids, length)


def segment_prod(data, segment_ids, length):
    """Segment Prod operator

    Parameters
    ----------
    data: A `Tensor`. Must be one of the following types:
     `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`,
      `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.

    segment_ids: A `Tensor`. Must be one of the following types:
     `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

    length: 'Int'. The length of output.

    Returns
    -------
    A `Tensor`. Has the same type as `data`.
    """
    return _make.segment_prod(data, segment_ids, length)


def segment_min(data, segment_ids, length):
    """Segment Min operator.

    Parameters
    ----------
    data: A `Tensor`. Must be one of the following types:
     `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`,
      `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.

    segment_ids: A `Tensor`. Must be one of the following types:
     `int32`, `int64`.
      A 1-D tensor whose size is equal to the size of `data`'s
      first dimension.  Values should be sorted and can be repeated.

    length: 'Int'. The length of output.

    Returns
    -------
    A `Tensor`. Has the same type as `data`.
    """
    return _make.segment_min(data, segment_ids, length)
