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

# pylint: disable=redefined-builtin, wildcard-import
"""vivante specific declaration and schedules."""
from __future__ import absolute_import as _abs

from .softmax import schedule_softmax
from .nn import schedule_lrn
from .conv2d_transpose_nchw import *
from .reduction import schedule_reduce
from .conv1d_transpose_ncw import *
from .conv1d import *
from .group_conv2d_nchw import *
from .injective import schedule_injective, schedule_elemwise, schedule_broadcast
from .deformable_conv2d import *
