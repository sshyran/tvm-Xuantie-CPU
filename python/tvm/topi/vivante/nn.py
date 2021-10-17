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
# pylint: disable=invalid-name, unused-argument
"""scheduler functions for vivante backend"""
from __future__ import absolute_import as _abs
from tvm import te
from .injective import schedule_injective_from_existing


def schedule_lrn(outs):
    """Schedule for LRN
       It imitates from topi/include/topi/cuda/normalization.h,
       and adjust the num_thread to 8 to suit the vivante npu

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    out_ops = []
    for t in outs:
        out_ops.append(t.op)

    s = te.create_schedule(out_ops)

    lrn = outs[0]
    sqr_sum_up = lrn.op.input_tensors[1]
    sqr_sum = sqr_sum_up.op.input_tensors[0]
    set_pad = sqr_sum.op.input_tensors[0]

    schedule_injective_from_existing(s, set_pad)
    schedule_injective_from_existing(s, sqr_sum)
    schedule_injective_from_existing(s, sqr_sum_up)
    schedule_injective_from_existing(s, lrn)

    return s
