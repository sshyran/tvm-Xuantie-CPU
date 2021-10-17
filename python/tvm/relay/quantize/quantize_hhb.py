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
# pylint: disable=invalid-name, wildcard-import, unused-wildcard-import
"""Automatic quantization toolkit."""
from .. import transform as _transform
from .. import expr as _expr
from ...ir import transform

from .quantize import current_qconfig
from ._convert_to_csi import (
    calibration,
    convert_to_csi_qnn,
    fuse_layer,
    optimize_quantization,
    const_to_uint8,
)


def _bind_params(func, params):
    """Bind the params to the expression."""
    name_dict = {}
    for arg in func.params:
        name = arg.name_hint
        if name in name_dict:
            name_dict[name] = None
        else:
            name_dict[name] = arg
    bind_dict = {}
    for k, v in params.items():
        if k not in name_dict:
            continue
        arg = name_dict[k]
        if arg is None:
            raise ValueError("Multiple args in the function have name %s" % k)
        bind_dict[arg] = _expr.const(v)
    return _expr.bind(func, bind_dict)


def quantize_hhb(module, params=None, dataset=None):
    """The quantization procedure.

    Parameters
    ---------
    module: Module
        The original module.

    params : dict of str to NDArray
        Input parameters to the graph that do not change
        during inference time. Used for constant folding.

    dataset: list of dict of Var -> NDArray
        The calibration dataset.

    Returns
    -------
    ret: Function
        The graph after quantization
    """

    if params:
        module["main"] = _bind_params(module["main"], params)
    optimize = transform.Sequential(
        [
            _transform.SimplifyInference(),
            _transform.FoldConstant(),
            _transform.SimplifyExpr(),
            _transform.FoldScaleAxis(),
            _transform.CanonicalizeOps(),
            _transform.FoldConstant(),
        ]
    )
    module = optimize(module)
    quant_params = calibration(module, dataset)
    csi_module = convert_to_csi_qnn(module, quant_params)
    curr_qconfig = current_qconfig()

    csi_module = fuse_layer(csi_module, fuse_relu=curr_qconfig.fuse_relu)
    csi_module = optimize_quantization(
        csi_module, broadcast_quantization=curr_qconfig.broadcast_quantization
    )

    csi_module = const_to_uint8(csi_module)
    return csi_module
