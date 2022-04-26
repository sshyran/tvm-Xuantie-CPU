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
"""Load Lib for C++ TOPI ops and schedules"""
import ctypes

from tvm._ffi import libinfo


def _load_lib(libname):
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path(libname)
    if lib_path is None:
        raise ValueError(
            "Loading lib " + libname + " faild, "
            "use env var TVM_LIBRARY_PATH or LD_LIBRARY_PATH to specify libpath"
        )
    ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)


_load_lib("libcsi_nn2_ref_x86.so")
