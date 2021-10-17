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
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
# pylint: disable=no-else-return, inconsistent-return-statements
"""Minimum graph runtime that executes graph containing TVM PackedFunc."""
import os
import numpy as np
import tvm._ffi

from .._ffi.runtime_ctypes import TVMContext
from ..rpc import base as rpc_base


def create(libmod, origin_mod, ctx, output_dir="."):
    """Create a runtime executor module given a graph and module.

    Parameters
    ----------
    libmod : tvm.runtime.Module
        The module of the corresponding function

    ctx : TVMContext or list of TVMContext
        The context to deploy the module. It can be local or remote when there
        is only one TVMContext. Otherwise, the first context in the list will
        be used as this purpose. All context should be given for heterogeneous
        execution.

    Returns
    -------
    graph_module : CSIModule
        Runtime graph module that can be used to execute the graph.
    """
    contrib_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    source_dir = os.path.join(contrib_dir, "..", "..", "..")
    include_path = os.path.join(source_dir, "install_nn2", "include")

    lib_path = os.path.join(output_dir, "quant.so")
    kwargs = {}
    kwargs["options"] = ["-O2", "-g", "-I" + include_path]
    libmod.export_hhb_library(lib_path, fcompile=False, output_dir=output_dir, **kwargs)
    lib = tvm.runtime.load_module(lib_path)

    ctx, _, device_type_id = get_device_ctx(lib, ctx)
    fcreate = tvm._ffi.get_global_func("tvm.hhb_runtime.create")
    mod = HHBModule(fcreate(lib, *device_type_id), origin_mod)

    return mod


def get_device_ctx(libmod, ctx):
    """Parse and validate all the device context(s).

    Parameters
    ----------
    libmod : tvm.runtime.Module
        The module of the corresponding function

    ctx : TVMContext or list of TVMContext

    Returns
    -------
    ctx : list of TVMContext
    num_rpc_ctx : Number of rpc contexts
    device_type_id : List of device type and device id
    """

    if isinstance(ctx, TVMContext):
        ctx = [ctx]
    elif not isinstance(ctx, (list, tuple)):
        raise ValueError("ctx has to be the type of TVMContext or a list of " "TVMCTVMContext")
    for cur_ctx in ctx:
        if not isinstance(cur_ctx, TVMContext):
            raise ValueError("ctx has to be the type of TVMContext or a list " "of TVMContext")

    # device_type_id[0], device_type_id[1] are used as the primary/fallback
    # context type and id. All other ones are used as device context for
    # heterogeneous execution.
    num_rpc_ctx = 0
    device_type_id = []
    for cur_ctx in ctx:
        device_type = cur_ctx.device_type
        if device_type >= rpc_base.RPC_SESS_MASK:
            assert libmod.type_key == "rpc"
            assert rpc_base._SessTableIndex(libmod) == cur_ctx._rpc_sess._tbl_index
            num_rpc_ctx += 1
            device_type = cur_ctx.device_type % rpc_base.RPC_SESS_MASK
        device_type_id.append(device_type)
        device_type_id.append(cur_ctx.device_id)

    if 0 < num_rpc_ctx < len(ctx):
        raise ValueError("Either all or none of the contexts should be rpc.")
    return ctx, num_rpc_ctx, device_type_id


class HHBModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.

    Attributes
    ----------
    module : tvm.runtime.Module
        The internal tvm module that holds the actual graph functions.
    """

    def __init__(self, module, omod):
        self.module = module
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._set_output = module["set_output"]
        self._get_input = module["get_input"]
        self._set_input = module["set_input"]
        self._set_params = module["set_params"]

        if isinstance(omod, dict):
            for idx, shape in enumerate(omod["output_shape_list"]):
                output = np.array(np.zeros(shape, dtype=omod["output_dtype_list"][idx])) + idx
                self.set_output(output)
            self._output = omod["output_shape_list"]
        else:
            func = omod["main"]
            self._output = func.ret_type

            # if type(func.ret_type) == tvm.ir.type.TupleType:
            if isinstance(func.ret_type, tvm.ir.type.TupleType):
                shapes = [[y.value for y in x.shape] for x in func.ret_type.fields]
                for i, shape in enumerate(shapes):
                    output = np.array(np.zeros(shape, dtype=func.ret_type.fields[i].dtype)) + i
                    self.set_output(output)
            else:
                shape = [x.value for x in func.ret_type.shape]
                output = np.array(np.zeros(shape, dtype=func.ret_type.dtype))
                self.set_output(output)

    def set_params(self, params_path):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        self._set_params(params_path)

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        if params:
            # upload big arrays first to avoid memory issue in rpc mode
            keys = list(params.keys())
            keys.sort(key=lambda x: -np.prod(params[x].shape))
            for k in keys:
                self._set_input(tvm.runtime.ndarray.array(params[k].astype(np.float32)))

    def set_output(self, ref):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        self._set_output(tvm.runtime.ndarray.array(ref))

    def run(self, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        if isinstance(self._output, tvm.ir.tensor_type.TensorType):
            return 1
        elif isinstance(self._output, tvm.ir.type.TupleType):
            return len(self._output.fields)
        elif isinstance(self._output, list):
            return len(self._output)

    def get_input(self, index, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(index).copyto(out)
            return out

        return self._get_input(index)

    def get_output(self, index, out=None):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container
        """
        if out:
            self._get_output(index, out)
            return out

        return self._get_output(index)
