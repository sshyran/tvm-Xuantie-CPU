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
# pylint: disable=no-else-return, unused-variable, not-callable, consider-using-enumerate
""" CSI testing utilities """
import time
import numpy as np
import tvm
import tvm.contrib.cc as cc
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

from tvm import rpc
from tvm.contrib import util
from tvm.contrib import graph_runtime
from .rpc import base as rpc_base

# from tvm.contrib.debugger import debug_runtime as graph_runtime


def is_remote():
    return True


def csi_get_target_host():
    if not is_remote():
        return "llvm"
    else:
        # return 'c'
        return "llvm -target=csky-unknown-linux -mcpu=c860 -mfloat-abi=hard"


_remote = None


def csi_get_remote():
    assert is_remote()
    global _remote
    if _remote is None:
        _remote = rpc.connect("172.16.202.57", 9000)

    return _remote


def csi_get_host_compiler(isRemote, *kargs):
    """ Get host compiler """
    if isRemote:
        program = "/home/wuzx/toolchain-csky/bin/csky-abiv2-linux-g++"
        options = ["-mcpu=c860", "-mfloat-abi=hard", "-fPIC", "-O3"]
    else:
        program = "g++"
        options = []

    options.extend(*kargs)

    return cc.cross_compiler(program, options=options, output_format=".so")


def csi_convert_to_remote(mod, ctx, name, *kargs):
    """ Convert to remote """
    if ctx.device_type >= rpc_base.RPC_SESS_MASK:
        temp = util.tempdir()
        path_dso = temp.relpath(name)
        mod.export_library(path_dso, csi_get_host_compiler(True, kargs))
        csi_get_remote().upload(path_dso)
        return csi_get_remote().load_module(name)
    else:
        temp = util.tempdir()
        path_dso = temp.relpath(name)
        mod.export_library(path_dso, csi_get_host_compiler(False, kargs))
        return tvm.runtime.load_module(path_dso)


def get_ctx_list():
    if not is_remote():
        target = "llvm"
        ctx = tvm.cpu(0)
    else:
        target = tvm.target.vivante()
        ctx = csi_get_remote().cl()

    return [(target, ctx)]


# Leverage the pass manager to write a simple white list based annotator
@transform.function_pass(opt_level=0)
class WhiteListAnnotator:
    """ White list """

    def __init__(self, op_list_, compiler):
        self.op_list = op_list_
        self.compiler = compiler

    def transform_function(self, func, mod, ctx):
        """ transform """
        annotator = self

        class Annotator(tvm.relay.ExprMutator):
            """ Annotator """

            def visit_call(self, call):
                op_name = call.op.name
                if (
                    op_name in annotator.op_list.keys()
                    and call.checked_type.dtype in annotator.op_list[op_name]
                ):
                    new_args = []
                    for arg in call.args:
                        ann = compiler_begin(super().visit(arg), annotator.compiler)
                        new_args.append(ann)
                    new_call = relay.Call(call.op, new_args, call.attrs, call.type_args)
                    return compiler_end(new_call, annotator.compiler)
                else:
                    return super().visit_call(call)

        return Annotator().visit(func)


op_list = {
    "nn.conv2d": ["uint8"],
    "nn.dense": ["uint8"],
    "nn.relu": ["uint8"],
    "add": ["int8", "float32"],
    "substract": ["int8", "float32"],
}


def get_pattern_table():
    """ Get pattern table """

    def make_add_relu_pattern():

        x = relay.var("x")
        y = relay.var("y")
        t = relay.add(x, y)
        z = relay.var("z")
        return relay.add(t, z)

    pattern_table = [("add_add_add", make_add_relu_pattern())]

    return pattern_table


def partition_graph(mod, op_list_, compiler):
    """ Partition graph """
    mod = WhiteListAnnotator(op_list_, compiler)(mod)
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)
    return mod


def merge_composite(mod, pattern_table):
    """ Merge composite """
    empty_attr = tvm.ir.make_node("DictAttrs")
    newMod = tvm.IRModule()
    for key, value in mod.functions.items():
        if value.attrs is not None and "Compiler" in value.attrs.keys():
            old_attrs = value.attrs
            tmpf = relay.Function(
                value.params,
                value.body,
                ret_type=value.ret_type,
                type_params=value.type_params,
                attrs=empty_attr,
            )
            tmpMod = tvm.IRModule()
            tmpMod[key] = tmpf
            tmpMod = relay.transform.MergeComposite(pattern_table)(tmpMod)
            new_func = tmpMod[key]

            newMod[key] = relay.Function(
                new_func.params,
                new_func.body,
                ret_type=new_func.ret_type,
                type_params=new_func.type_params,
                attrs=old_attrs,
            )
        else:
            newMod[key] = value

    return newMod


def eval_time(opname, func, ref_result, *kargs):
    """ Get run time """
    if not isinstance(ref_result, list):
        ref_result = [ref_result]

    target, ctx = get_ctx_list()[0]

    CSI_Codegen = False

    mod = tvm.IRModule()
    mod["main"] = func

    if CSI_Codegen:
        mod = partition_graph(mod, op_list, "csinn")
        mod = merge_composite(mod, get_pattern_table())

    # print(mod)

    with tvm.target.vivante():
        with tvm.relay.build_config(opt_level=3):
            graph, lib, params = tvm.relay.build(mod, target, target_host=csi_get_target_host())

    # print(lib.get_source());
    # print(lib.imported_modules[0].get_source())
    # print(lib.imported_modules[1].get_source())

    lib = csi_convert_to_remote(
        lib, ctx, "func.so", "-I/home/wuzx/workspace/tvm/src/runtime/contrib/"
    )

    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)

    shapeList = []
    for i in range(len(kargs)):
        shapeList.append(kargs[i].shape)
    for i in ref_result:
        shapeList.append(i.shape)

    time_start = None
    time_end = None

    for i in range(len(kargs)):
        ctx.sync()
        time_start = time.clock()
        m.set_input(i, kargs[i])
        ctx.sync()
        time_end = time.clock()

        # print("set input time = ", (time_end - time_start) * 1000, " ms")

    ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=3)
    prof_res = np.array(ftimer().results) * 1000  # convert to millisecond

    for i in range(len(ref_result)):
        # op_res = tvm.nd.empty(ref_result[i].shape, ref_result[i].dtype, ctx)
        ctx.sync()
        time_start = time.clock()
        # op_res = None
        op_res = m.get_output(i)
        ctx.sync()
        time_end = time.clock()
        # print("get output time = ", (time_end - time_start) * 1000, " ms")

        tvm.testing.assert_allclose(op_res.asnumpy(), ref_result[i], rtol=1e-3, atol=1e-3)

    print(
        "op = \t%s\t oshape = \t%s\t dtype = \t%s\t Mean inference time (std dev): "
        "\t%.4f\t ms (\t%.4f\t ms)"
        % (opname.__name__, shapeList, op_res.dtype, np.mean(prof_res), np.std(prof_res))
    )
