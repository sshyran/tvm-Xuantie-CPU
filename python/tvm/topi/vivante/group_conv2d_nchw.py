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
"""The template for cuda group_conv2d_nchw"""
import tvm
from tvm import te
from tvm import autotvm

from .injective import schedule_injective_from_existing
from ..cuda.tensor_intrin import dp4a
from ..util import traverse_inline, get_const_tuple, get_const_int


@autotvm.register_topi_schedule("group_conv2d_nchw.vivante")
def schedule_group_conv2d_nchw(cfg, outs):
    """TOPI schedule callback of group conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for group conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "group_conv2d_nchw":
            _schedule_group_conv2d_nchw_direct(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_group_conv2d_nchw_direct(cfg, s, conv):
    """Schedule group conv2d NCHW direct template"""
    workload = conv.op.attrs["workload"]
    groups = get_const_int(workload[6])
    num_filters = get_const_int(conv.shape[1])

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_g", cfg.axis(groups), num_outputs=2)
    cfg.define_split("tile_f", cfg.axis(num_filters // groups), num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.target_name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    kernel_scope, n = s[output].split(n, nparts=1)

    cfg.fallback_split("tile_n", [-1, 1, 2, 1])
    cfg.fallback_split("tile_g", [-1, 2])
    # cfg.fallback_split('tile_f', [-1, 1, 2,1])
    cfg.fallback_split("tile_y", [-1, 1, 2, 1])
    cfg.fallback_split("tile_x", [-1, 1, 2, 1])

    g, f = s[output].split(f, nparts=groups)
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bg, vg = cfg["tile_g"].apply(s, output, g)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bg, bf, by, bx, vn, vg, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[output].bind(bn, te.thread_axis("blockIdx.z"))
    s[output].bind(s[output].fuse(bg, bf), te.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
    s[output].bind(vn, te.thread_axis("vthread"))
    s[output].bind(vg, te.thread_axis("vthread"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))

    cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(tf, te.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(ty, tx)
        s[output].bind(tyx, te.thread_axis("threadIdx.x"))
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
    rco, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_ry"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    N, CO, OH, OW = get_const_tuple(output.shape)
    _, CI_div_groups, KH, KW = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OH * OW * CO * CI_div_groups * KH * KW)


@autotvm.register_topi_schedule("group_conv2d_NCHWc_int8.vivante")
def schedule_group_conv2d_NCHWc_int8(cfg, outs):
    """TOPI schedule callback of group conv2d for cuda gpu

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for group conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "group_conv2d_NCHWc_int8":
            _schedule_group_conv2d_NCHWc_int8(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


_dp4a = dp4a("shared", "shared", "local")


def _schedule_group_conv2d_NCHWc_int8(cfg, s, output):
    """Schedule group conv2d int8 NCHWc template"""
    workload = output.op.attrs["workload"]
    groups = get_const_int(workload[6])

    conv = output.op.input_tensors[0]
    packed_data, packed_kernel = conv.op.input_tensors

    if isinstance(packed_data.op, tvm.te.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # skip this part during tuning to make records accurate
        # this part will be pre-computed during NNVM's pre-compute optimization pass
        s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
        s[packed_kernel].pragma(s[packed_kernel].op.axis[0], "debug_skip_region")
    else:
        if isinstance(packed_kernel.op, tvm.te.ComputeOp) and packed_kernel.name == "packed_kernel":
            # data and kernel are not pre-computed, schedule layout transform here
            schedule_injective_from_existing(s, packed_data)
            schedule_injective_from_existing(s, packed_kernel)

    if pad_data != packed_data:
        s[pad_data].compute_inline()

    # create cache stage
    AA = s.cache_read(pad_data, "shared", [conv])
    WW = s.cache_read(packed_kernel, "shared", [conv])

    s[conv].set_scope("local")

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    oc_chunk = get_const_int(output.shape[1])
    # tile and bind spatial axes
    n, f, y, x, c = s[output].op.axis
    cfg.define_split("tile_n", n, num_outputs=4)
    cfg.define_split("tile_g", cfg.axis(groups), num_outputs=2)
    cfg.define_split("tile_f", cfg.axis(oc_chunk // groups), num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    g, f = s[output].split(f, nparts=groups)
    s[output].bind(n, te.thread_axis("blockIdx.z"))
    bn, vn, tn, ni = cfg["tile_n"].apply(s, output, n)
    bg, vg = cfg["tile_g"].apply(s, output, g)
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(bn, bg, bf, by, bx, vn, vg, vf, vy, vx, tn, tf, ty, tx, ni, fi, yi, xi)
    s[output].bind(bn, te.thread_axis("blockIdx.z"))
    s[output].bind(s[output].fuse(bg, bf), te.thread_axis("blockIdx.y"))
    s[output].bind(s[output].fuse(by, bx), te.thread_axis("blockIdx.x"))
    s[output].bind(vn, te.thread_axis("vthread"))
    s[output].bind(vg, te.thread_axis("vthread"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    cfg.define_knob("fuse_yx", [0, 1])  # fuse ty,tx or tn,tf
    if cfg["fuse_yx"].val:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(tf, te.thread_axis("threadIdx.y"))
        tyx = s[output].fuse(ty, tx)
        s[output].bind(tyx, te.thread_axis("threadIdx.x"))
        s[conv].compute_at(s[output], tyx)

        # number of threads
        n_tz = cfg["tile_n"].size[2]
        n_ty = cfg["tile_f"].size[2]
        n_tx = cfg["tile_y"].size[2] * cfg["tile_x"].size[2]
    else:
        s[output].bind(tn, te.thread_axis("threadIdx.z"))
        s[output].bind(s[output].fuse(tn, tf), te.thread_axis("threadIdx.z"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))
        s[conv].compute_at(s[output], tx)

        # number of threads
        n_tz = cfg["tile_n"].size[2] * cfg["tile_f"].size[2]
        n_ty = cfg["tile_y"].size[2]
        n_tx = cfg["tile_x"].size[2]

    # tile and bind reduction axes
    n, f, y, x, c = s[conv].op.axis
    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)
    rco, rci = cfg["tile_rc"].apply(s, conv, rc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)
    _, rc_block = s[conv].split(rc_block, factor=4)
    s[conv].tensorize(rc_block, _dp4a)

    s[AA].compute_at(s[conv], rxo)
    s[WW].compute_at(s[conv], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        c = s[load].op.axis[-1]
        c_outer, c = s[load].split(c, factor=4)
        s[load].vectorize(c)
        fused = s[load].op.axis[:-1] + [c_outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        fused, tz = s[load].split(fused, factor=n_tz)
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # double buffer
    cfg.define_knob("AA_double_buffer", [0, 1])
    cfg.define_knob("WW_double_buffer", [0, 1])
    if cfg["AA_double_buffer"].val:
        s[AA].double_buffer()
    if cfg["WW_double_buffer"].val:
        s[WW].double_buffer()

    # unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", False)

    return s
