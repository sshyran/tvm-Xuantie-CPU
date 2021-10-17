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
"""Conv2d transpose template for vivante backend"""

import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from ..util import get_const_tuple, traverse_inline


@autotvm.register_topi_schedule("conv2d_transpose_nchw.vivante")
def schedule_conv2d_transpose_nchw(cfg, outs):
    """TOPI Schedule callback for conv2d transpose operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The parameters for this template

    outs: Array of Tensor
        The computation graph description of conv2d transpose
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d transpose.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _fallback_schedule(N, F, Y, X):
        # pylint: disable=unused-argument
        # split N (batch dimension)
        if N > 1:
            cfg["tile_n"] = SplitEntity([-1, 1, 1, 4])
        else:
            cfg["tile_n"] = SplitEntity([1, 1, 1, 1])
        # split F (output channel dimension)
        if F > 1:
            cfg["tile_f"] = SplitEntity([-1, 1, 8, 1])
        # split Y (height dimension)
        y_split_factor = 1
        for candidate in range(5, 17):
            if Y % candidate == 0:
                y_split_factor = candidate
                break
        cfg["tile_y"] = SplitEntity([-1, 1, 1, y_split_factor])
        # split X (width dimension)
        x_split_factor = 1
        for candidate in range(5, 17):
            if X % candidate == 0:
                x_split_factor = candidate
                break
        cfg["tile_x"] = SplitEntity([-1, x_split_factor, 1, 1])
        # split RC (input channel dimension, which is a reduction axis)
        cfg["tile_rc"] = SplitEntity([-1, 1, 16])
        # other configurations
        cfg["fuse_yx"] = OtherOptionEntity(False)
        cfg["unroll_explicit"] = OtherOptionEntity(True)
        cfg["auto_unroll_max_step"] = OtherOptionEntity(64)

    def _callback(op):
        if op.tag == "conv2d_transpose_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)

            ##### space definition begin #####
            n, f, y, x = s[conv].op.axis
            rc = s[conv].op.reduce_axis[0]
            cfg.define_split("tile_n", cfg.axis(n), num_outputs=4)
            cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
            cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
            cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)
            cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=3)
            cfg.define_knob("auto_unroll_max_step", [64, 512, 1500])

            target = tvm.target.Target.current()
            if target.target_name in ["nvptx", "rocm"]:
                cfg.define_knob("unroll_explicit", [1])
            else:
                cfg.define_knob("unroll_explicit", [0, 1])

            if cfg.is_fallback:
                N, F, Y, X = get_const_tuple(conv.shape)
                _fallback_schedule(N, F, Y, X)

            ##### space definition end #####

            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            if conv.op in s.outputs:
                output = conv
                OL = s.cache_write(conv, "local")
            else:
                output = s.outputs[0].output(0)
                s[conv].set_scope("local")
                OL = conv

            # create cache stage
            s[pad_data].set_scope("shared")
            AA = pad_data
            WW = s.cache_read(kernel, "shared", [OL])

            # tile and bind spatial axes
            n, f, y, x = s[output].op.axis
            kernel_scope, n = s[output].split(n, nparts=1)
            bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
            bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
            by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].reorder(bn, bf, by, bx, vn, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
            s[output].bind(bn, te.thread_axis("blockIdx.z"))
            s[output].bind(bf, te.thread_axis("blockIdx.y"))
            s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
            s[output].bind(vn, te.thread_axis("vthread"))
            s[output].bind(vf, te.thread_axis("vthread"))
            s[output].bind(vy, te.thread_axis("vthread"))
            s[output].bind(vx, te.thread_axis("vthread"))

            cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf

            if cfg["fuse_yx"].val:
                s[output].bind(tn, te.thread_axis("threadIdx.z"))
                s[output].bind(tf, te.thread_axis("threadIdx.y"))
                tyx = s[output].fuse(ty, tx)
                s[output].bind(s[output].fuse(ty, tx), te.thread_axis("threadIdx.x"))
                s[OL].compute_at(s[output], tyx)

                # number of threads
                n_tz = cfg["tile_n"].size[2]
                n_ty = cfg["tile_f"].size[2]
                n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
            else:
                s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
                s[output].bind(ty, te.thread_axis("threadIdx.y"))
                s[output].bind(tx, te.thread_axis("threadIdx.x"))
                s[OL].compute_at(s[output], tx)

                # number of threads
                n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
                n_ty = cfg["tile_y"].size[2]
                n_tx = cfg["tile_x"].size[2]

            # tile reduction axes
            n, f, y, x = s[OL].op.axis
            rc, ry, rx = s[OL].op.reduce_axis
            rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
            s[OL].reorder(rco, rcm, ry, rx, rci, n, f, y, x)

            s[AA].compute_at(s[OL], rx)
            s[WW].compute_at(s[OL], rx)

            # cooperative fetching
            for load in [AA, WW]:
                n, f, y, x = s[load].op.axis
                fused = s[load].fuse(f, y, x)
                tz, fused = s[load].split(fused, nparts=n_tz)
                ty, fused = s[load].split(fused, nparts=n_ty)
                tx, fused = s[load].split(fused, nparts=n_tx)
                s[load].bind(tz, te.thread_axis("threadIdx.z"))
                s[load].bind(ty, te.thread_axis("threadIdx.y"))
                s[load].bind(tx, te.thread_axis("threadIdx.x"))

            s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    traverse_inline(s, outs[0].op, _callback)

    return s
