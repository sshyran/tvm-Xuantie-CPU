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
"""Manage Codegen"""
import logging
import sys
import os
import shutil
from tvm import relay

from .common import argument_filter_helper, hhb_exit
from .common import ALL_ARGUMENTS_INFO
from .common import AttributeDict
from .common import HHBException
from .hhbir_manage import HHBBoardQnnCodegenIR


logger = logging.getLogger("HHB")


@argument_filter_helper
def collect_codegen_config(filtered_args, extra=None):
    """collect codegen arguments"""
    filtered_args.codegen_config = AttributeDict()
    for k in ALL_ARGUMENTS_INFO["codegen"]:
        filtered_args.codegen_config[k] = filtered_args[k]


@argument_filter_helper
def set_codegen_config(filtered_args, extra=None):
    """set codegen arguments"""

    def _set_memory_type(io_memory_type, io_num, unify_type=None):
        res = io_memory_type
        if io_memory_type is None:
            if unify_type is None:
                res = [0] * io_num
            else:
                res = [unify_type] * io_num
        else:
            if len(io_memory_type) == 1:
                res = io_memory_type * io_num
            else:
                if len(io_memory_type) != io_num:
                    hhb_exit(
                        "There are {} input/output, but get {} input/output memory".format(
                            io_num, len(io_memory_type)
                        )
                    )
        return res

    if not hasattr(filtered_args, "codegen_config"):
        raise HHBException("Please execute 'collect_codegen_config' filter first.")
    if not hasattr(extra, "input_num"):
        raise HHBException("extra has no input_num attr")
    if not hasattr(extra, "output_num"):
        raise HHBException("extra has no output_num attr")

    filtered_args.codegen_config.input_memory_type = _set_memory_type(
        filtered_args.codegen_config.input_memory_type,
        extra.input_num,
        filtered_args.codegen_config.memory_type,
    )

    filtered_args.codegen_config.output_memory_type = _set_memory_type(
        filtered_args.codegen_config.output_memory_type,
        extra.output_num,
        filtered_args.codegen_config.memory_type,
    )


def get_execute_path():
    if hasattr(sys, "_MEIPASS"):
        execute_path = os.path.dirname(os.path.realpath(sys.executable))
    else:
        execute_path, _ = os.path.split(os.path.abspath(__file__))
        execute_path = os.path.join(execute_path, "..")
    return execute_path


def main_c_codegen(
    codegen_obj: HHBBoardQnnCodegenIR,
    input_shape,
    output_shape,
    board,
    output_path,
    postprocess="top5",
    model_save="run_only",
    without_preprocess=False,
    preprocess_params=None,
    multithread=False,
    input_memory_type=None,
    q_scheme=None,
):
    """ Generate the main.c file """

    execute_path = get_execute_path()
    if board == "anole":
        if multithread:
            main_file = os.path.join(execute_path, "config", "anole_multithread.tp")
        else:
            main_file = os.path.join(execute_path, "config", "anole.tp")
    elif board in ("light", "hlight", "ch8601"):
        main_file = os.path.join(execute_path, "config", "light.tp")
    elif board == "c906" or board == "c908":
        main_file = os.path.join(execute_path, "config", "c906.tp")
    elif board == "dp1k":
        main_file = os.path.join(execute_path, "config", "dp1k.tp")
    else:
        main_file = os.path.join(execute_path, "config", "thead.tp")

    with open(main_file, "r") as f:
        code_str = f.read()

    template_dir = os.path.join(execute_path, "config", "template")

    # check options setting
    if preprocess_params.calibrate_data_format == "npz":
        without_preprocess = True
    if board in ("asp", "ch8601", "dp1k"):
        without_preprocess = True
    if board != "anole" and board != "light":
        # disable_nbg = True
        model_save = "run_only"

    #######################################################################
    #
    # Header Codegen
    #
    with open(os.path.join(template_dir, "header.tp"), "r") as f:
        header_str = f.read()
    if board == "anole":
        header_str += '\n#include "csi_ovx.h"'
    elif board in ("light", "hlight", "asp", "c906", "c908", "dp1k", "ch8601"):
        header_str += '\n#include "csi_ref.h"'
    else:
        header_str += '\n#include "csi_nn.h"'

    if not without_preprocess:
        header_str += '\n#include "process.h"'
        process_c_path = os.path.join(execute_path, "config", "process", "src", "process.c")
        process_c = os.path.join(output_path, codegen_obj.preprocess_source_name)
        process_h_path = os.path.join(execute_path, "config", "process", "include", "process.h")
        process_h = os.path.join(output_path, codegen_obj.preprocess_header_name)
        logger.info("write process header to %s", process_h)
        logger.info("write process source to %s", process_c)
        shutil.copy(process_h_path, process_h)
        shutil.copy(process_c_path, process_c)
    io_c_path = os.path.join(execute_path, "config", "process", "src", "io.c")
    io_c = os.path.join(output_path, codegen_obj.preio_source_name)
    io_h_path = os.path.join(execute_path, "config", "process", "include", "io.h")
    io_h = os.path.join(output_path, codegen_obj.preio_header_name)
    logger.info("write io header to %s", io_h)
    logger.info("write io source to %s", io_c)
    shutil.copy(io_h_path, io_h)
    shutil.copy(io_c_path, io_c)

    code_str = code_str.replace("#_hhb_header_files_#", header_str)

    #######################################################################
    #
    # Macro Codegen
    #
    with open(os.path.join(template_dir, "macro_def.tp"), "r") as f:
        macro_str = f.read()
    code_str = code_str.replace("#_hhb_macro_def_#", macro_str)

    #######################################################################
    #
    # Function Declaration Codegen
    #
    with open(os.path.join(template_dir, "function_decl.tp"), "r") as f:
        function_str = f.read()
    # if disable_nbg == False:
    if model_save != "run_only":
        if multithread and board == "anole":
            function_str += "\nvoid *csinn_nbg(const char *nbg_file_name, int deviceIndex);"
        else:
            function_str += "\nvoid *csinn_nbg(const char *nbg_file_name);"
    csinn_args = ""
    for i in range(len(input_shape)):
        csinn_args += "void *data" + str(i) + ", "
    function_str = function_str.replace("#_csinn_args#", csinn_args)
    if multithread and board == "anole":
        function_str = function_str.replace(
            "void *csinn_(char *params);", "void *csinn_(char *params, int deviceIndex);"
        )
    code_str = code_str.replace("#_hhb_function_decl_#", function_str)

    if board == "c860":
        csinn_args = ""
        for i in range(len(input_shape) + len(output_shape)):
            csinn_args += "void *data" + str(i) + ", "
        code_str = code_str.replace("#_thead_csinn_args#", csinn_args)

    #######################################################################
    #
    # Global Variable Codegen
    #
    with open(os.path.join(template_dir, "global_var_decl.tp"), "r") as f:
        global_var_str = f.read()

    def _convert_shape2str(shape_list):
        res = ""
        for shape in shape_list:
            shape = shape if len(shape) != 0 else [1]
            tmp_str = list(map(str, shape))
            tmp_str = " * ".join(tmp_str)
            if q_scheme == "int16_sym":
                tmp_str += " * 2"
            res += tmp_str + ", "
        return res

    global_var_str = global_var_str.replace("#_input_size_define#", _convert_shape2str(input_shape))
    global_var_str = global_var_str.replace("#_model_name_define#", "network")
    code_str = code_str.replace("#_hhb_global_var_decl_#", global_var_str)

    #######################################################################
    #
    # Preprocess Codegen
    #
    preprocess_str = ""
    if not without_preprocess:
        with open(os.path.join(template_dir, "preprocess_def.tp"), "r") as f:
            preprocess_str = f.read()
        preprocess_str = _preprocess_macro_define(preprocess_params, preprocess_str)
    code_str = code_str.replace("#_hhb_preprocess_def_#", preprocess_str)

    #######################################################################
    #
    # Utils Codegen
    #
    with open(os.path.join(template_dir, "utils_def.tp"), "r") as f:
        utils_str = f.read()
    code_str = code_str.replace("#_hhb_utils_def_#", utils_str)

    #######################################################################
    #
    # Postprocess Codegen
    #
    with open(os.path.join(template_dir, "postprocess_def.tp"), "r") as f:
        postprocess_str = f.read()

    convert_fouput = ""
    if board in ("light", "hlight", "asp", "c906", "c908"):
        convert_fouput = "struct csi_tensor *foutput = csi_ref_tensor_transform_f32(output);"

    postprocess_str = postprocess_str.replace("#_convert_fouput_#", convert_fouput)

    show_top5 = ""
    if "top5" in postprocess:
        if board in ("ch8601", "dp1k"):
            show_top5 = "csi_show_top5(output, sess);"
        elif board in ("light", "hlight", "asp", "c906", "c908"):
            show_top5 = "csi_show_top5(foutput, sess);"
        else:
            show_top5 = "csi_ovx_show_top5(i, sess);"
    postprocess_str = postprocess_str.replace("#_show_top5_stats_#", show_top5)

    free_anole_input_data = ""
    free_output_data = ""
    if board == "anole":
        free_anole_input_data = "free(input->data);"
        free_output_data = "free(output->data);"
    if board in ("light", "hlight", "asp", "c906", "c908"):
        free_output_data = "csi_ref_tensor_transform_free_f32(foutput);\n"
        if board in ("c906", "c908"):
            free_output_data += " " * 8 + "if (!output->is_const) {\n"
            free_output_data += " " * 12 + "free(output->data);\n"
            free_output_data += " " * 8 + "}"
    postprocess_str = postprocess_str.replace("#_free_anole_input_data_#", free_anole_input_data)
    postprocess_str = postprocess_str.replace("#_free_output_data_#", free_output_data)

    save_output = ""
    if "save" in postprocess:
        save_output = "char filename[FILE_LENGTH] = {0};\n"
        save_output += " " * 8 + "char shape[SHAPE_LENGHT] = {0};\n"
        save_output += (
            " " * 8 + "shape2string(output->dim, output->dim_count, shape, SHAPE_LENGHT);\n"
        )
        save_output += (
            " " * 8
            + 'snprintf(filename, FILE_LENGTH, "%s_output%u_%s.txt", filename_prefix, i, shape);\n'
        )
        if board in ("light", "hlight", "asp", "c906", "c908"):
            save_output += " " * 8 + "int output_size = csi_tensor_size(foutput);\n"
            save_output += (
                " " * 8 + "save_data_to_file(filename, (float*)foutput->data, output_size);\n"
            )
        else:
            save_output += " " * 8 + "csi_ovx_save_output(i, filename, sess);\n"
    postprocess_str = postprocess_str.replace("#_save_output_stats_#", save_output)
    code_str = code_str.replace("#_hhb_postprocess_def_#", postprocess_str)

    #######################################################################
    #
    # Main Codegen
    #
    code_str = code_str.replace("#_input_num#", str(len(input_shape)))
    code_str = code_str.replace("#_output_num#", str(len(output_shape)))

    create_graph_stats = ""

    # if disable_nbg:
    if model_save == "run_only":
        if multithread and board == "anole":
            create_graph_stats += "sess = csinn_(params, device_index);"
        else:
            create_graph_stats += "sess = csinn_(params);"
    else:
        if board == "anole":
            create_graph_stats += (
                " " * 4 + "char *suffix = params_path + (strlen(params_path) - 3);\n"
            )
            create_graph_stats += " " * 4 + 'if (strcmp(suffix, ".nb") == 0) {\n'
            create_graph_stats += " " * 8 + "// create binary graph\n"
            if multithread:
                create_graph_stats += " " * 8 + "sess = csinn_nbg(params_path, device_index);\n"
            else:
                create_graph_stats += " " * 8 + "sess = csinn_nbg(params_path);\n"
            create_graph_stats += " " * 4 + "} else {\n"
            create_graph_stats += " " * 8 + "// create general graph\n"
            if multithread:
                create_graph_stats += " " * 8 + "sess = csinn_(params, device_index);\n"
            else:
                create_graph_stats += " " * 8 + "sess = csinn_(params);\n"
            create_graph_stats += " " * 4 + "}"
        elif board in ("light", "hlight"):
            create_graph_stats += "char *suffix = params_path + (strlen(params_path) - 8);\n"
            create_graph_stats += " " * 4 + 'if (strcmp(suffix, ".mbs.bin") == 0) {\n'
            create_graph_stats += " " * 8 + "// create binary graph\n"
            create_graph_stats += " " * 8 + "sess = csinn_nbg(params_path);\n"
            create_graph_stats += " " * 4 + "} else {\n"
            if model_save == "save_and_run":
                create_graph_stats += " " * 8 + "// create general graph\n"
                create_graph_stats += " " * 8 + "sess = csinn_(params);\n"
            else:
                create_graph_stats += " " * 8 + "exit(0);\n"
            create_graph_stats += " " * 4 + "}"
    code_str = code_str.replace("#_create_graph_stats_#", create_graph_stats)

    aligned_buffer_stats = ""
    if input_memory_type and (1 in input_memory_type):
        aligned_buffer_stats += "void *input_aligned[input_num];\n"
        aligned_buffer_stats += " " * 4 + "for (i = 0; i < input_num; i++) {\n"
        aligned_buffer_stats += (
            " " * 8 + "input_aligned[i] = csi_mem_alloc_aligned(input_size[i], 0);\n"
        )
        aligned_buffer_stats += " " * 4 + "}\n"
    code_str = code_str.replace("#_aligned_buffer_stats_#", aligned_buffer_stats)

    aligned_buffer_copy = ""
    if input_memory_type:
        for i in range(len(input_shape)):
            if input_memory_type[i] == 1:  # cpu aligned
                if i != 0:
                    aligned_buffer_copy += " " * 8
                aligned_buffer_copy += (
                    "memcpy(input_aligned["
                    + str(i)
                    + "], input["
                    + str(i)
                    + "], input_size["
                    + str(i)
                    + "]);\n"
                )
    code_str = code_str.replace("#_aligned_buffer_copy_#", aligned_buffer_copy)

    get_input_data_stats = ""
    if without_preprocess:
        get_input_data_stats += "if (get_file_type(data_path[i * input_num + j]) != FILE_BIN) {\n"
        get_input_data_stats += (
            " " * 16
            + 'printf("Please input binary files, since you compiled the model without preprocess.\\n");\n'
        )
        get_input_data_stats += " " * 16 + "return -1;\n"
        get_input_data_stats += " " * 12 + "}\n"
        get_input_data_stats += (
            " " * 12 + "inputf[j] = (float*)get_binary_from_file(data_path[i * input_num + j]);\n"
        )
    else:
        is_rgb = 1
        if preprocess_params["gray"]:
            is_rgb = 0

        if preprocess_params["pixel_format"] == "RGB":
            to_bgr = 0
        elif preprocess_params["pixel_format"] == "BGR":
            to_bgr = 1
        get_input_data_stats += "int input_len = input_size[j]"
        if q_scheme == "int16_sym":
            get_input_data_stats += " / 2;\n"
        else:
            get_input_data_stats += ";\n"
        get_input_data_stats += (
            " " * 12
            + "struct image_data *img = get_input_data(data_path[i * input_num + j], input_len);\n"
        )
        get_input_data_stats += (
            " " * 12
            + "if (get_file_type(data_path[i * input_num + j]) == FILE_PNG || get_file_type(data_path[i * input_num + j]) == FILE_JPEG) {\n"
        )
        get_input_data_stats += (
            " " * 16 + "preprocess(img, " + str(is_rgb) + ", " + str(to_bgr) + ");\n"
        )
        get_input_data_stats += " " * 12 + "}\n"
        get_input_data_stats += " " * 12 + "inputf[j] = img->data;\n"
        get_input_data_stats += " " * 12 + "free_image_data(img);\n"
    code_str = code_str.replace("#_get_input_data_stats_#", get_input_data_stats)

    run_csinn_stats_anole = ""
    run_csinn_stats_thead = ""
    for i in range(len(input_shape)):
        if input_memory_type and input_memory_type[i] == 1:
            run_csinn_stats_anole += "input_aligned[" + str(i) + "], "
        else:
            run_csinn_stats_anole += "input[" + str(i) + "], "
        run_csinn_stats_thead += "input[" + str(i) + "], "
    code_str = code_str.replace("#_anole_value_pass#", run_csinn_stats_anole)

    if board == "c860":
        for i in range(len(output_shape)):
            run_csinn_stats_thead += "output[" + str(i) + "], "
        code_str = code_str.replace("#_thead_value_pass#", run_csinn_stats_thead)

    logger.info("save main souce to %s", os.path.join(output_path, codegen_obj.main_source_name))
    with open(os.path.join(output_path, codegen_obj.main_source_name), "w") as f:
        f.write(code_str)


def _preprocess_macro_define(preprocess_params, preprocess_str):
    if len(preprocess_params["data_mean"]) not in (1, 3):
        raise HHBException(
            "do not know how to deal with mean values:{}".format(preprocess_params["data_mean"])
        )
    if preprocess_params["add_preprocess_node"]:
        preprocess_params["data_mean"] = [0, 0, 0]
        preprocess_params["data_scale"] = 1.0
    if len(preprocess_params["data_mean"]) == 1:
        preprocess_params["data_mean"] = preprocess_params["data_mean"] * 3
    data_resize = preprocess_params["data_resize"]
    if isinstance(data_resize, int):
        data_resize = [data_resize, 0]
    preprocess_params_code = ""
    preprocess_params_code += "#define RESIZE_HEIGHT" + "       " + str(data_resize[0]) + "\n"
    preprocess_params_code += "#define RESIZE_WIDTH" + "        " + str(data_resize[1]) + "\n"
    preprocess_params_code += (
        "#define CROP_HEGHT" + "          " + str(preprocess_params["target_shape"][0]) + "\n"
    )
    preprocess_params_code += (
        "#define CROP_WIDTH" + "          " + str(preprocess_params["target_shape"][1]) + "\n"
    )
    preprocess_params_code += (
        "#define R_MEAN" + "              " + str(preprocess_params["data_mean"][2]) + "\n"
    )
    preprocess_params_code += (
        "#define G_MEAN" + "              " + str(preprocess_params["data_mean"][1]) + "\n"
    )
    preprocess_params_code += (
        "#define B_MEAN" + "              " + str(preprocess_params["data_mean"][0]) + "\n"
    )
    preprocess_params_code += (
        "#define SCALE" + "               " + str(preprocess_params["data_scale"]) + "\n"
    )
    preprocess_str = preprocess_str.replace("#_preprocess_define_#", preprocess_params_code)
    return preprocess_str


class VisitLayers(relay.ExprVisitor):
    """get layer kinds"""

    def __init__(self, func):
        super(VisitLayers, self).__init__()
        self.layer_kinds = []
        self.visit(func)

    def visit_call(self, call):
        _ = [self.visit(arg) for arg in call.args]

        op_name = call.op.name
        if op_name == "qnn.csi.conv2d":
            in_shape = list(call.type_args[0].concrete_shape)
            kernel_shape = list(call.type_args[1].concrete_shape)
            if call.attrs.groups > 1:
                op_name = "group_conv2d"
                if call.attrs.out_layout == "NHWC":
                    # for i805 NHWC layout
                    if call.attrs.groups == in_shape[3] and kernel_shape[0] == 1:
                        op_name = "depthwise_conv2d"
                elif call.attrs.out_layout == "NCHW":
                    if call.attrs.groups == in_shape[0] and kernel_shape[1] == 1:
                        op_name = "depthwise_conv2d"
        if op_name not in self.layer_kinds:
            self.layer_kinds.append(op_name)

    def get_op_kinds(self):
        return self.layer_kinds


def generate_func_map(model, board, dump_file_path):
    def get_register_func(i805h_path):
        import re

        register_func = {}
        with open(i805h_path, "r") as f:
            for line in f:
                match_obj = re.match(r"int csi_i805_(.*)_u8", line)
                if match_obj:
                    func_name = match_obj.group(1)
                    if "init" in func_name:
                        func_name = func_name[:-5]
                    if func_name not in register_func:
                        register_func[func_name] = func_name
        return register_func

    op_kinds = VisitLayers(model["main"]).get_op_kinds()
    execute_path = get_execute_path()
    i805h_path = os.path.join(execute_path, "../../install_nn2/include/csi_i805.h")
    register_funcs = get_register_func(i805h_path)
    func_file = os.path.join(execute_path, "config", "bc_map.tp")
    with open(func_file, "r") as f:
        code_str = f.read()
    repleased_str = ""
    optimized_stwich_str = "\t\tcase CSINN_OP_{}:\n\t\t\treturn csi_#TARGET_{}_u8;\n\t\t\tbreak;\n"
    ref_stwich_str = "\t\tcase CSINN_OP_{}:\n\t\t\treturn csi_ref_{}_quant;\n\t\t\tbreak;\n"
    mem_ops = ["transpose", "resahpe", "squeeze"]
    for op in op_kinds:
        kind = op.split(".")[-1]
        if kind in register_funcs:
            repleased_str += optimized_stwich_str.format(kind.upper(), register_funcs[kind])
        else:
            tmp_str = ref_stwich_str.format(kind.upper(), kind.lower())
            if kind in mem_ops:
                tmp_str = tmp_str.replace("_quant", "")
            repleased_str += tmp_str

    code_str = code_str.replace("#OP_CASE", repleased_str)
    code_str = code_str.replace("#TARGET", board)
    with open(dump_file_path, "w+") as f:
        f.write(code_str)


def generate_c906_bc_reg(model, board, dump_file_path, q_scheme):
    c906_bc_init_map = {
        "conv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV2D", "csi_c906_conv2d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONV2D", "csi_c906_conv2d_init"],
        },
        "group_conv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GROUP_CONV2D", "csi_c906_conv2d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GROUP_CONV2D", "csi_c906_conv2d_init"],
        },
        "conv1d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV1D", "csi_c906_conv1d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONV1D", "csi_c906_conv1d_init"],
        },
        "maxpool2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MAXPOOL2D", "csi_c906_maxpool2d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MAXPOOL2D", "csi_c906_maxpool2d_init"],
        },
        "avgpool2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_AVGPOOL2D", "csi_c906_avgpool2d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_AVGPOOL2D", "csi_c906_avgpool2d_init"],
        },
        "depthwise_conv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DEPTHWISE_CONV2D", "csi_c906_depthwise_conv2d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DEPTHWISE_CONV2D", "csi_c906_depthwise_conv2d_init"],
        },
        "fullyconnected": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FULLYCONNECTED", "csi_c906_fullyconnected_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FULLYCONNECTED", "csi_c906_fullyconnected_init"],
        },
        "cache_matmul": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CACHE_MATMUL", "csi_c906_cache_matmul_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CACHE_MATMUL", "csi_c906_cache_matmul_init"],
        },
        "div": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DIV", "csi_c906_div_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DIV", "csi_c906_div_init"],
        },
        "cache_conv1d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CACHE_CONV1D", "csi_c906_cache_conv1d_init"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CACHE_CONV1D", "csi_c906_cache_conv1d_init"],
        },
    }

    c906_bc_map = {
        "abs": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ABS", "csi_c906_abs_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ABS", "csi_c906_abs_f32"],
        },
        "acos": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ACOS", "csi_ref_acos_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ACOS", "csi_ref_acos_f32"],
        },
        "acosh": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ACOSH", "csi_ref_acosh_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ACOSH", "csi_ref_acosh_f32"],
        },
        "add": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ADD", "csi_c906_add_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ADD", "csi_c906_add_f32"],
        },
        "and": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_AND", "csi_ref_and_i8"],
        },
        "arange": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ARANGE", "csi_ref_arange_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ARANGE", "csi_ref_arange_f32"],
        },
        "argmax": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ARGMAX", "csi_ref_argmax_stride_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ARGMAX", "csi_ref_argmax_stride_i32_f32"],
        },
        "argmin": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ARGMIN", "csi_ref_argmin_stride_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ARGMIN", "csi_ref_argmin_stride_i32_f32"],
        },
        "asin": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ASIN", "csi_ref_asin_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ASIN", "csi_ref_asin_f32"],
        },
        "asinh": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ASINH", "csi_ref_asinh_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ASINH", "csi_ref_asinh_f32"],
        },
        "atan": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ATAN", "csi_ref_atan_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ATAN", "csi_ref_atan_f32"],
        },
        "atanh": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ATANH", "csi_ref_atanh_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ATANH", "csi_ref_atanh_f32"],
        },
        "avgpool2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_AVGPOOL2D", "csi_ref_avgpool2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_AVGPOOL2D", "csi_ref_avgpool2d_f32"],
        },
        "avgpool3d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_AVGPOOL3D", "csi_ref_avgpool3d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_AVGPOOL3D", "csi_ref_avgpool3d_f32"],
        },
        "bn": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_BN", "csi_ref_batch_normalization_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_BN", "csi_ref_batch_normalization_f32"],
        },
        "batch_to_space": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_BATCH_TO_SPACE", "csi_ref_batch_to_space_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_BATCH_TO_SPACE", "csi_ref_batch_to_space_f32"],
        },
        "broadcast_to": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_BROADCOST", "csi_ref_broadcast_to_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_BROADCOST", "csi_ref_broadcast_to_f32"],
        },
        "cache_matmul": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CACHE_MATMUL", "csi_c906_cache_matmul_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CACHE_MATMUL", "csi_ref_cache_matmul_f32"],
        },
        "cache_conv1d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CACHE_CONV1D", "csi_c906_cache_conv1d_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CACHE_CONV1D", "csi_ref_cache_conv1d_f32"],
        },
        "ceil": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CEIL", "csi_ref_ceil_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CEIL", "csi_ref_ceil_f32"],
        },
        "clip": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CLIP", "csi_c906_clip_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CLIP", "csi_c906_clip_f32"],
        },
        "concat": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONCAT", "csi_nn_rvv_concat_int8"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONCAT", "csi_c906_concat_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONCAT", "csi_c906_concat_f32"],
        },
        "conv1d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV1D", "csi_ref_conv1d_quantv"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONV1D", "csi_ref_conv1d_f32"],
        },
        "conv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV2D", "csi_ref_conv2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONV2D", "csi_ref_conv2d_f32"],
        },
        "conv2d_relu": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV2D_RELU", "csi_ref_conv2d_relu_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONV2D_RELU", "csi_ref_conv2d_relu_f32"],
        },
        "conv2d_relu6": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV2D_RELU6", "csi_ref_conv2d_relu6_quant"],
        },
        "depthwise_conv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DEPTHWISE_CONV2D", "csi_ref_depthwise_conv2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DEPTHWISE_CONV2D", "csi_ref_depthwise_conv2d_f32"],
        },
        "depthwise_conv2d_relu": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_DEPTHWISE_CONV2D_RELU",
                "csi_ref_depthwise_conv2d_relu_quant",
            ],
        },
        "depthwise_conv2d_relu6": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_DEPTHWISE_CONV2D_RELU6",
                "csi_ref_depthwise_conv2d_relu6_quant",
            ],
        },
        "group_conv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GROUP_CONV2D", "csi_ref_group_conv2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GROUP_CONV2D", "csi_ref_group_conv2d_f32"],
        },
        "conv3d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CONV3D", "csi_ref_conv3d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CONV3D", "csi_ref_conv3d_f32"],
        },
        "deconv2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DECONV2D", "csi_ref_deconv2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DECONV2D", "csi_ref_deconv2d_f32"],
        },
        "depthwise_deconv2d": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_DEPTHWISE_DECONV2D",
                "csi_ref_depthwise_deconv2d_quant",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_DEPTHWISE_DECONV2D",
                "csi_ref_depthwise_deconv2d_f32",
            ],
        },
        "deconv3d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DECONV3D", "csi_ref_deconv3d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DECONV3D", "csi_ref_deconv3d_f32"],
        },
        "cos": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_COS", "csi_ref_cos_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_COS", "csi_ref_cos_f32"],
        },
        "cosh": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_COSH", "csi_ref_cosh_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_COSH", "csi_ref_cosh_f32"],
        },
        "cumprod": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CUMPROD", "csi_ref_cumprod_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CUMPROD", "csi_ref_cumprod_f32"],
        },
        "cumsum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_CUMSUM", "csi_ref_cumsum_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_CUMSUM", "csi_ref_cumsum_f32"],
        },
        "depth_to_space": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DEPTH_TO_SPACE", "csi_ref_depth_to_space_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DEPTH_TO_SPACE", "csi_ref_depth_to_space_f32"],
        },
        "div": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_DIV", "csi_ref_div_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_DIV", "csi_ref_div_f32"],
        },
        "elu": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ELU", "csi_ref_elu_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ELU", "csi_ref_elu_f32"],
        },
        "equal": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_EQUANL", "csi_ref_equal_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_EQUANL", "csi_ref_equal_f32"],
        },
        "erf": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ERF", "csi_ref_erf_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ERF", "csi_ref_erf_f32"],
        },
        "exp": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_EXP", "csi_ref_exp_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_EXP", "csi_ref_exp_f32"],
        },
        "expand_dims": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_EXPAND_DIMS", "csi_ref_expand_dims_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_EXPAND_DIMS", "csi_ref_expand_dims_f32"],
        },
        "expm1": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_EXPM1", "csi_ref_expm1_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_EXPM1", "csi_ref_expm1_f32"],
        },
        "flatten": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FLATTEN", "csi_ref_flatten"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FLATTEN", "csi_ref_flatten"],
        },
        "floor_divide": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FLOOR_DIVIDE", "csi_ref_floor_divide_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FLOOR_DIVIDE", "csi_ref_floor_divide_f32"],
        },
        "floor_mod": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FLOOR_MOD", "csi_ref_floor_mod_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FLOOR_MOD", "csi_ref_floor_mod_f32"],
        },
        "floor": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FLOOR", "csi_ref_floor_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FLOOR", "csi_ref_floor_f32"],
        },
        "fsmn": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FSMN", "csi_ref_fsmn_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FSMN", "csi_ref_fsmn_f32"],
        },
        "fullyconnected": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_FULLYCONNECTED", "csi_c906_fullyconnected_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_FULLYCONNECTED", "csi_c906_fullyconnected_f32"],
        },
        "gather_nd": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GATHER_ND", "csi_ref_gather_nd_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GATHER_ND", "csi_ref_gather_nd_f32"],
        },
        "gather": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GATHER", "csi_c906_gather_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GATHER", "csi_ref_gather_f32"],
        },
        "global_avgpool2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GLOBAL_AVGPOOL2D", "csi_ref_global_avgpool2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GLOBAL_AVGPOOL2D", "csi_c906_global_avgpool2d_f32"],
        },
        "global_maxpool2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GLOBAL_MAXPOOL2D", "csi_ref_global_maxpool2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GLOBAL_MAXPOOL2D", "csi_c906_global_maxpool2d_f32"],
        },
        "greater_equal_": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GREATHER_EQUAL", "csi_ref_greater_equal_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GREATHER_EQUAL", "csi_ref_greater_equal_f32"],
        },
        "greater": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_GREATHER", "csi_ref_greater_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_GREATHER", "csi_ref_greater_f32"],
        },
        "hard_sigmoid": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_HARD_SIGMOID", "csi_ref_hard_sigmoid_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_HARD_SIGMOID", "csi_ref_hard_sigmoid_f32"],
        },
        "im2col": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_IM2COL", "csi_ref_im2col_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_IM2COL", "csi_ref_im2col_f32"],
        },
        "l2_normalization": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_L2N", "csi_ref_l2_normalization_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_L2N", "csi_ref_l2_normalization_f32"],
        },
        "layer_norm": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LAYER_NORM", "csi_c906_layer_norm_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LAYER_NORM", "csi_ref_layer_norm_f32"],
        },
        "leaky_relu": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LEAKY_RELU", "csi_c906_leaky_relu_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LEAKY_RELU", "csi_c906_leaky_relu_f32"],
        },
        "less_equal": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LESS_EQUAL", "csi_ref_less_equal_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LESS_EQUAL", "csi_ref_less_equal_f32"],
        },
        "less": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LESS", "csi_ref_less_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LESS", "csi_ref_less_f32"],
        },
        "log_softmax": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOG_SOFTMAX", "csi_ref_log_softmax_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOG_SOFTMAX", "csi_ref_log_softmax_f32"],
        },
        "log": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOG", "csi_ref_log_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOG", "csi_ref_log_f32"],
        },
        "log1p": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOG1P", "csi_ref_log1p_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOG1P", "csi_ref_log1p_f32"],
        },
        "logical_and": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOGICAL_AND", "csi_ref_logical_and_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOGICAL_AND", "csi_ref_logical_and_f32"],
        },
        "logical_not": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOGICAL_NOT", "csi_ref_logical_not_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOGICAL_NOT", "csi_ref_logical_not_f32"],
        },
        "logical_or": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOGICAL_OR", "csi_ref_logical_or_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOGICAL_OR", "csi_ref_logical_or_f32"],
        },
        "logical_xor": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LOGICAL_XOR", "csi_ref_logical_xor_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LOGICAL_XOR", "csi_ref_logical_xor_f32"],
        },
        "lrn": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_LRN", "csi_ref_lrn_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_LRN", "csi_ref_lrn_f32"],
        },
        "matmul": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MATMUL", "csi_c906_matmul_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MATMUL", "csi_ref_matmul_f32"],
        },
        "max_stride": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MAX", "csi_ref_max_stride_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MAX", "csi_ref_max_stride_f32"],
        },
        "maximum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MAXIMUM", "csi_ref_maximum_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MAXIMUM", "csi_ref_maximum_f32"],
        },
        "maxpool2d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MAXPOOL2D", "csi_ref_maxpool2d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MAXPOOL2D", "csi_ref_maxpool2d_f32"],
        },
        "maxpool2d_locat": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MAXPOOL2D_LOCAT", "csi_ref_maxpool2d_locat_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MAXPOOL2D_LOCAT", "csi_ref_maxpool2d_locat_f32"],
        },
        "maxpool3d": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MAXPOOL3D", "csi_ref_maxpool3d_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MAXPOOL3D", "csi_ref_maxpool3d_f32"],
        },
        "mean": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MEAN", "csi_ref_mean_stride_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MEAN", "csi_ref_mean_stride_f32"],
        },
        "mean_stride": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MEAN_STRIDE", "csi_ref_mean_stride_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MEAN_STRIDE", "csi_ref_mean_stride_f32"],
        },
        "min": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MIN", "csi_ref_min_stride_quant"],
        },
        "minimum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MINIMUM", "csi_c906_minimum_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MINIMUM", "csi_c906_minimum_f32"],
        },
        "mod": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MOD", "csi_ref_mod_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MOD", "csi_ref_mod_f32"],
        },
        "mul": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MUL", "csi_nn_rvv_mul_int8"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_MUL", "csi_c906_mul_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_MUL", "csi_c906_mul_f32"],
        },
        "ndarray_size": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_NDARRAY_SIZE", "csi_ref_ndarray_size_i8"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_NDARRAY_SIZE", "csi_ref_ndarray_size_f32"],
        },
        "negative": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_NEGATIIVE", "csi_ref_negative_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_NEGATIIVE", "csi_ref_negative_f32"],
        },
        "not_equal": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_NOT_EQUAL", "csi_ref_not_equal_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_NOT_EQUAL", "csi_ref_not_equal_f32"],
        },
        "not": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_NOT", "csi_ref_not_i8"],
        },
        "or": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_OR", "csi_ref_or_i8"],
        },
        "pad": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_PAD", "csi_ref_pad_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_PAD", "csi_ref_pad_f32"],
        },
        "power": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_POWER", "csi_ref_power_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_POWER", "csi_ref_power_f32"],
        },
        "prelu": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_PRELU", "csi_c906_prelu_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_PRELU", "csi_c906_prelu_f32"],
        },
        "prod": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_PROD", "csi_ref_prod_stride_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_PROD", "csi_ref_prod_stride_f32"],
        },
        "proposal": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_PROPOSAL", "csi_ref_proposal_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_PROPOSAL", "csi_ref_proposal_f32"],
        },
        "psroipooling": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_PSROIPOOLING", "csi_ref_psroipooling_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_PSROIPOOLING", "csi_ref_psroipooling_f32"],
        },
        "reduce_logsumexp": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REDUCE_LOGSUMEXP", "csi_ref_reduce_logsumexp_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REDUCE_LOGSUMEXP", "csi_ref_reduce_logsumexp_f32"],
        },
        "reduce_max": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REDUCE_MAX", "csi_ref_reduce_max_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REDUCE_MAX", "csi_ref_reduce_max_f32"],
        },
        "reduce_mean": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REDUCE_MEAN", "csi_ref_reduce_mean_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REDUCE_MEAN", "csi_ref_reduce_mean_f32"],
        },
        "reduce_min": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REDUCE_MIN", "csi_ref_reduce_min_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REDUCE_MIN", "csi_ref_reduce_min_f32"],
        },
        "reduce_prod": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REDUCE_PROD", "csi_ref_reduce_prod_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REDUCE_PROD", "csi_ref_reduce_prod_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REDUCE_SUM", "csi_ref_reduce_sum_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REDUCE_SUM", "csi_ref_reduce_sum_f32"],
        },
        "relu": {
            "CSINN_DTYPE_INT8": ["CSINN_OP_RELU", "csi_nn_rvv_relu_int8"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RELU", "csi_c906_relu_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RELU", "csi_c906_relu_f32"],
        },
        "relu1": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RELU1", "csi_c906_relu1_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RELU1", "csi_c906_relu1_f32"],
        },
        "relu6": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RELU6", "csi_c906_relu6_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RELU6", "csi_c906_relu6_f32"],
        },
        "relun": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RELUN", "csi_ref_relun_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RELUN", "csi_ref_relun_f32"],
        },
        "reshape": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RESHAPE", "csi_ref_reshape"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RESHAPE", "csi_c906_reshape_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RESHAPE", "csi_ref_reshape"],
        },
        "resize": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RESIZE", "csi_ref_resize_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RESIZE", "csi_ref_resize_f32"],
        },
        "reverse": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_REVERSE", "csi_ref_reverse_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_REVERSE", "csi_ref_reverse_f32"],
        },
        "roipool": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ROIPOOL", "csi_ref_roipool_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ROIPOOL", "csi_ref_roipool_f32"],
        },
        "round": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_ROUND", "csi_ref_round_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_ROUND", "csi_ref_round_f32"],
        },
        "rsqrt": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_RSQRT", "csi_ref_rsqrt_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_RSQRT", "csi_ref_rsqrt_f32"],
        },
        "scatter_nd": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SCATTER_ND", "csi_ref_scatter_nd_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SCATTER_ND", "csi_ref_scatter_nd_f32"],
        },
        "segment_max": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SEGMENT_MAX", "csi_ref_segment_max_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SEGMENT_MAX", "csi_ref_segment_max_f32"],
        },
        "unsorted_segment_max": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_UNSORTED_SEGMENT_MAX",
                "csi_ref_unsorted_segment_max_quant",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_UNSORTED_SEGMENT_MAX",
                "csi_ref_unsorted_segment_max_f32",
            ],
        },
        "segment_mean": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SEGMENT_MEAN", "csi_ref_segment_mean_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SEGMENT_MEAN", "csi_ref_segment_mean_f32"],
        },
        "unsorted_segment_mean": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_UNSORTED_SEGMENT_MEAN",
                "csi_ref_unsorted_segment_mean_quant",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_UNSORTED_SEGMENT_MEAN",
                "csi_ref_unsorted_segment_mean_f32",
            ],
        },
        "segment_min": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SEGMENT_MIN", "csi_ref_segment_min_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SEGMENT_MIN", "csi_ref_segment_min_f32"],
        },
        "unsorted_segment_min": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_UNSORTED_SEGMENT_MIN",
                "csi_ref_unsorted_segment_min_quant",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_UNSORTED_SEGMENT_MIN",
                "csi_ref_unsorted_segment_min_f32",
            ],
        },
        "segment_prod": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SEGMENT_PROD", "csi_ref_segment_prod_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SEGMENT_PROD", "csi_ref_segment_prod_f32"],
        },
        "unsorted_segment_prod": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_UNSORTED_SEGMENT_PROD",
                "csi_ref_unsorted_segment_prod_quant",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_UNSORTED_SEGMENT_PROD",
                "csi_ref_unsorted_segment_prod_f32",
            ],
        },
        "segment_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SEGMENT_SUM", "csi_ref_segment_sum_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SEGMENT_SUM", "csi_ref_segment_sum_f32"],
        },
        "unsorted_segment_sum": {
            "CSINN_DTYPE_FLOAT16": [
                "CSINN_OP_UNSORTED_SEGMENT_SUM",
                "csi_ref_unsorted_segment_sum_quant",
            ],
            "CSINN_DTYPE_FLOAT32": [
                "CSINN_OP_UNSORTED_SEGMENT_SUM",
                "csi_ref_unsorted_segment_sum_f32",
            ],
        },
        "select": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SELECT", "csi_ref_select_i8"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SELECT", "csi_ref_select_f32"],
        },
        "shape": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SHAPE", "csi_ref_shape_i8"],
        },
        "shuffle_channel": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SHUFFLE_CHANNEL", "csi_ref_shuffle_channel_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SHUFFLE_CHANNEL", "csi_ref_shuffle_channel_f32"],
        },
        "sigmoid": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SIGMOID", "csi_nn_rvv_sigmoid_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SIGMOID", "csi_ref_sigmoid_f32"],
        },
        "sign": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SIGN", "csi_ref_sign_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SIGN", "csi_ref_sign_f32"],
        },
        "sin": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SIN", "csi_ref_sin_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SIN", "csi_ref_sin_f32"],
        },
        "sinh": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SINH", "csi_ref_sinh_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SINH", "csi_ref_sinh_f32"],
        },
        "slice": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SLICE", "csi_ref_slice_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SLICE", "csi_ref_slice_f32"],
        },
        "softmax": {
            "CSINN_DTYPE_INT8": ["CSINN_OP_SOFTMAX", "csi_ref_softmax_quant"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SOFTMAX", "csi_nn_rvv_softmax_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SOFTMAX", "csi_ref_softmax_f32"],
        },
        "softplus": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SOFTPLUS", "csi_ref_softplus_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SOFTPLUS", "csi_ref_softplus_f32"],
        },
        "softrelu": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SOFTRELU", "csi_ref_softrelu_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SOFTRELU", "csi_ref_softrelu_f32"],
        },
        "softsign": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SOFTSIGN", "csi_ref_softsign_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SOFTSIGN", "csi_ref_softsign_f32"],
        },
        "space_to_batch": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SPACE_TO_BATCH", "csi_ref_space_to_batch_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SPACE_TO_BATCH", "csi_ref_space_to_batch_f32"],
        },
        "space_to_depth": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SPACE_TO_DEPTH", "csi_ref_space_to_depth_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SPACE_TO_DEPTH", "csi_ref_space_to_depth_f32"],
        },
        "split": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SPLIT", "csi_c906_split_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SPLIT", "csi_c906_split_f32"],
        },
        "sqrt": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SQRT", "csi_ref_sqrt_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SQRT", "csi_ref_sqrt_f32"],
        },
        "squeeze": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SQUEEZE", "csi_ref_squeeze"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SQUEEZE", "csi_ref_square_f32"],
        },
        "stack": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_STACK", "csi_ref_stack_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_STACK", "csi_ref_stack_f32"],
        },
        "strided_slice": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_STRIDED_SLICE", "csi_ref_strided_slice_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_STRIDED_SLICE", "csi_ref_strided_slice_f32"],
        },
        "sub": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SUB", "csi_c906_sub_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SUB", "csi_c906_sub_f32"],
        },
        "sum": {
            "CSINN_DTYPE_INT8": ["CSINN_OP_SUM", "csi_nn_rvv_sum_stride_int8"],
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_SUM", "csi_c906_sum_stride_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_SUM", "csi_ref_sum_stride_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TAN", "csi_ref_tan_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_TAN", "csi_ref_tan_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TANH", "csi_ref_tanh_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_TANH", "csi_ref_tanh_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_THRESHOLD_RELU", "csi_ref_threshold_relu_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_THRESHOLD_RELU", "csi_ref_threshold_relu_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TILE", "csi_ref_tile_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_TILE", "csi_ref_tile_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TOPK", "csi_ref_topk_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_TOPK", "csi_ref_topk_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TRUNC", "csi_ref_trunc_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_TRUNC", "csi_ref_trunc_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_TRANSPOSE", "csi_c906_transpose_fp16"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_TRANSPOSE", "csi_ref_transpose"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_UNPOOLING", "csi_ref_unpooling_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_UNPOOLING", "csi_ref_unpooling_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_UNSTACK", "csi_ref_unstack_qunat"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_UNSTACK", "csi_ref_unstack_f32"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_XOR", "csi_ref_xor_i8"],
        },
        "reduce_sum": {
            "CSINN_DTYPE_FLOAT16": ["CSINN_OP_YUV_RGB_SCALE", "csi_ref_yuv_rgb_scale_quant"],
            "CSINN_DTYPE_FLOAT32": ["CSINN_OP_YUV_RGB_SCALE", "csi_ref_yuv_rgb_scale_f32"],
        },
    }

    op_kinds = VisitLayers(model["main"]).get_op_kinds()
    if "qnn.csi.conv2d" in op_kinds:
        op_kinds.append("depthwise_conv2d")
    execute_path = get_execute_path()
    func_file = os.path.join(execute_path, "config", "bc_reg.tp")
    with open(func_file, "r") as f:
        code_str = f.read()
    bc_init_reg_str = ""
    bc_reg_str = ""
    op_init_str = "\tcsi_nn_{}_register_op_init({}, {}, {});\n"
    op_str = "\tcsi_nn_{}_register_op({}, {}, {});\n"
    if q_scheme == "unset":
        qtype = "CSINN_DTYPE_FLOAT32"
    elif q_scheme == "float16":
        qtype = "CSINN_DTYPE_FLOAT16"
    elif q_scheme == "int8_asym_w_sym":
        qtype = "CSINN_DTYPE_INT8"
    else:
        raise HHBException("C906 unsupport\n")

    for op in op_kinds:
        kind = op.split(".")[-1]
        if kind in c906_bc_init_map:
            m = c906_bc_init_map[kind]
            bc_init_reg_str += op_init_str.format(board, qtype, m[qtype][0], m[qtype][1])
        if kind in c906_bc_map:
            m = c906_bc_map[kind]
            bc_reg_str += op_str.format(board, qtype, m[qtype][0], m[qtype][1])

    code_str = code_str.replace("#INIT_REG#", bc_init_reg_str)
    code_str = code_str.replace("#BC_REG#", bc_reg_str)
    code_str = code_str.replace("#TARGET#", board)
    with open(dump_file_path, "w+") as f:
        f.write(code_str)
