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
# pylint: disable=arguments-differ
"""
Provide support to load HHB model and convert to different HHB models.
"""
import os
import sys
import logging
import shutil
from abc import ABC
from abc import abstractmethod
import tempfile
import tarfile
import yaml

import numpy as np

import tvm
from tvm import relay
from tvm.relay import quantize as qtz
from tvm.relay.backend import graph_runtime_factory
from tvm.contrib import util
from tvm.contrib import graph_runtime
from tvm.contrib import hhb_runtime

from utils import HHBModel
from utils import HHBException
from utils import match_mod_params
from utils import get_input_info_from_relay
from utils import get_output_info_from_relay
from utils import hhb_version
from utils import save_to_file
from utils import get_target
from utils import save_relay_module
from utils import print_top5
from utils import get_file_path
from utils import ensure_dir
from model_evaluation import PreprocessParams
from model_evaluation import DatasetLoader


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


class HHBModelBase(ABC):
    """Abstract class for HHB command line interface.

    Provide a unified way to convert and save HHB model.

    """

    @staticmethod
    @abstractmethod
    def name():
        """Model name"""

    @staticmethod
    @abstractmethod
    def model_type():
        """ Model type"""

    @abstractmethod
    def load_model(self, model_path):
        """Load a model from a given path.

        Parameters
        ----------
        model_path : str
            Path to a tar file

        Returns
        -------
        mod : tvm.relay.Module
            The loaded relay ir
        params : dict
            The parameters for the relay module.
        """

    @abstractmethod
    def convert(self):
        """ Convert loaded module into current module. """


class HHBCodegenModel(HHBModelBase):
    """ Codegen Model for HHB command line tools. """

    def __init__(self):
        self.input_name_list = None
        self.input_shape_list = None
        self.input_dtype_list = None
        self.output_shape_list = None
        self.output_dtype_list = None

        self.preprocess_params = None
        self.qnn_config = None

        # self.set_env()

    @staticmethod
    def name():
        return "codegen"

    @staticmethod
    def model_type():
        return HHBModel.CODEGEN

    def load_model(self, model_path):
        """ Get module from file. """
        model_type = HHBModel.guess_model(model_path)
        if model_type is None or model_type not in (HHBModel.RELAY, HHBModel.QNN):
            raise HHBException(
                "invalid module:{}, please get valid module "
                "by executing 'import' subcommand.".format(model_path)
            )
        params = None
        model_type_str = HHBModel.TYPE2NAME[model_type]
        ir_path = os.path.join(model_path, model_type_str + ".txt")
        logger.debug("read relay ir from file: %s", ir_path)
        with open(ir_path, "r") as f:
            mod = tvm.parser.fromtext(f.read())
        if model_type == HHBModel.RELAY:
            params_path = os.path.join(model_path, model_type_str + ".params")
            logger.debug("load params from file: %s", params_path)
            with open(params_path, "rb") as f:
                params = tvm.relay.load_param_dict(f.read())
        mod, params = match_mod_params(mod, params)

        (
            self.input_name_list,
            self.input_shape_list,
            self.input_dtype_list,
        ) = get_input_info_from_relay(mod, params)
        self.output_shape_list, self.output_dtype_list = get_output_info_from_relay(mod)
        return mod, params

    def convert(self):
        """ Codegen according to current model. """

    def set_quant_env(self, input_path=None):
        """ Set environment parameters for quantization while codegen"""
        if input_path is None:
            return
        input_model_type = HHBModel.guess_model(input_path)
        if input_model_type == HHBModel.QNN:
            config_path = os.path.join(input_path, "quantize_config.yaml")
            with open(config_path, "r") as f:
                config_dict = yaml.load(f.read())
            self.preprocess_params = config_dict["preprocess"]
            self.qnn_config = config_dict["qnn_config"]
            qconfig = qtz.qconfig(**(self.qnn_config))
            with qconfig:
                logger.debug("restore quantization config from qnn_config.yaml in %s", config_path)


class HHBCodegenX86FloatModel(HHBCodegenModel):
    """ Float module in x86 """

    def __init__(self):
        self.graph_name = "codegen_x86_float.json"
        self.lib_name = "codegen_x86_float.so"
        self.params_name = "codegen_x86_float.params"
        self.info_file = "codegen_x86_float.yaml"

        self.all_included_files = [self.graph_name, self.lib_name, self.params_name, self.info_file]

    def convert(self, mod, params, board, opt_level):
        target = get_target(board)
        with tvm.transform.PassContext(opt_level=opt_level):
            logger.debug("building relay graph without quantization")
            graph_module = relay.build(mod, target=target, params=params)
            return graph_module

    def save_model(self, graph_module, output_path, save_info=True):
        """Save codegen module into tar file"""
        if not isinstance(graph_module, graph_runtime_factory.GraphRuntimeFactoryModule):
            raise HHBException("need GraphRuntimeFactoryModule")
        path_lib = os.path.join(output_path, self.lib_name)
        path_graph = os.path.join(output_path, self.graph_name)
        path_params = os.path.join(output_path, self.params_name)

        logger.debug("write lib to %s", path_lib)
        graph_module.get_lib().export_library(path_lib)

        with open(path_graph, "w") as f:
            logger.debug("write graph to %s", f.name)
            f.write(graph_module.get_json())
        with open(path_params, "wb") as f:
            logger.debug("write params to %s", f.name)
            f.write(relay.save_param_dict(graph_module.get_params()))

        if save_info:
            info_dict = {
                "input_name_list": self.input_name_list,
                "input_shape_list": self.input_shape_list,
                "input_dtype_list": self.input_dtype_list,
                "output_shape_list": self.output_shape_list,
                "output_dtype_list": self.output_dtype_list,
            }
            info_path = os.path.join(output_path, self.info_file)
            logger.debug("write extra info in %s", info_path)
            with open(info_path, "w") as f:
                yaml.dump(info_dict, f)


class HHBCodegenX86QuantModel(HHBCodegenModel):
    """ Quant module in x86 """

    def __init__(self):
        self.lib_name = "codegen_x86_quant.so"
        self.lib_source_name = "lib0.c"
        self.params_name = "codegen_x86_quant.params"
        self.info_file = "codegen_x86_quant.yaml"

        self.all_included_files = [
            self.lib_source_name,
            self.lib_name,
            self.params_name,
            self.info_file,
        ]

        # self.tmp_dir = util.tempdir()

    def convert(self, mod, params, board, opt_level, output_path):
        target = get_target(board)
        params_file = get_file_path(output_path, self.params_name)

        func = mod["main"]
        func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
        mod["main"] = func

        logger.debug("write params to %s", params_file)
        with relay.build_config(opt_level=opt_level):
            lib, _ = relay.build_hhb(mod=mod, target=target, params=params, params_path=params_file)

        return lib

    def save_model(self, lib, output_path, save_info=True):
        """Save codegen module into tar file"""
        contrib_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(contrib_dir, "..", "..")
        include_path = os.path.join(source_dir, "install_nn2", "include")

        lib_path = os.path.join(output_path, self.lib_name)
        kwargs = {}
        kwargs["options"] = ["-O2", "-g", "-I" + include_path]
        logger.debug("write lib to %s", output_path)
        lib.export_hhb_library(lib_path, fcompile=False, output_dir=output_path, **kwargs)

        if save_info:
            info_dict = {
                "input_name_list": self.input_name_list,
                "input_shape_list": self.input_shape_list,
                "input_dtype_list": self.input_dtype_list,
                "output_shape_list": self.output_shape_list,
                "output_dtype_list": self.output_dtype_list,
            }
            info_path = os.path.join(output_path, self.info_file)
            logger.debug("write extra info to %s", info_path)
            with open(info_path, "w") as f:
                yaml.dump(info_dict, f)


class HHBCodegenAnoleModel(HHBCodegenModel):
    """ Quant module for anole """

    def __init__(self):
        self.params_name = "model.params"
        self.lib_source_name = "model.c"
        self.main_source_name = "main.c"
        self.preprocess_source_name = "process.c"
        self.preprocess_header_name = "process.h"

        self.all_included_files = [
            self.params_name,
            self.lib_source_name,
            self.main_source_name,
            self.preprocess_source_name,
            self.preprocess_header_name,
        ]

    def convert(
        self, mod, params, board, opt_level, output_path, postprocess="top5", disable_nbg=False
    ):
        # self.set_env()
        target = get_target(board)
        params_file = os.path.join(output_path, self.params_name)

        func = mod["main"]
        func = func.with_attr("global_symbol", tvm.runtime.container.String("csinn"))
        mod["main"] = func

        logger.debug("write params to %s", params_file)
        with relay.build_config(opt_level=opt_level):
            lib, _ = relay.build_hhb(mod=mod, target=target, params=params, params_path=params_file)

        logger.debug("write lib source code to %s", os.path.join(output_path, self.lib_source_name))
        lib.save(os.path.join(output_path, self.lib_source_name))
        self.main_codegen(
            mod, params, board, output_path, postprocess=postprocess, disable_nbg=disable_nbg
        )

    def main_codegen(self, mod, params, board, output_path, postprocess="top5", disable_nbg=False):
        """ Generate the main.c file """
        if not hasattr(self, "preprocess_params"):
            raise HHBException("does not get preprocess params...")

        if hasattr(sys, "_MEIPASS"):
            execute_path = os.path.dirname(os.path.realpath(sys.executable))
        else:
            execute_path, _ = os.path.split(os.path.abspath(__file__))

        if board == "anole":
            template_file = "config/anole.tp"
        else:
            template_file = "config/thead.tp"

        with open(os.path.join(execute_path, template_file), "r") as f:
            code_str = f.read()
        process_c_path = os.path.join(execute_path, "config", "process", "src", "process.c")
        process_c = os.path.join(output_path, self.preprocess_source_name)
        process_h_path = os.path.join(execute_path, "config", "process", "include", "process.h")
        process_h = os.path.join(output_path, self.preprocess_header_name)

        logger.debug("write process header to %s", process_h)
        logger.debug("write process source to %s", process_c)
        shutil.copy(process_h_path, process_h)
        shutil.copy(process_c_path, process_c)

        _, input_shape_list, _ = get_input_info_from_relay(mod, params)
        output_shape_list, _ = get_output_info_from_relay(mod)

        if postprocess == "top5":
            code_str = code_str.replace("#_show_top5_#", str(1))
            code_str = code_str.replace("#_save_output_#", str(0))
        elif postprocess == "save":
            code_str = code_str.replace("#_show_top5_#", str(0))
            code_str = code_str.replace("#_save_output_#", str(1))
        else:
            code_str = code_str.replace("#_show_top5_#", str(1))
            code_str = code_str.replace("#_save_output_#", str(1))

        if disable_nbg:
            code_str = code_str.replace("#_disable_nbg_#", str(1))
        else:
            code_str = code_str.replace("#_disable_nbg_#", str(0))

        input_num_code = len(input_shape_list)
        code_str = code_str.replace("#_input_num#", str(input_num_code))
        output_num_code = len(output_shape_list)
        code_str = code_str.replace("#_output_num#", str(output_num_code))
        input_csinn_code = ""
        for i in range(input_num_code):
            input_csinn_code += "void *data" + str(i) + ", "
        code_str = code_str.replace("#_anole_csinn_args#", input_csinn_code)

        input_csinn_code = ""
        for i in range(input_num_code + output_num_code):
            input_csinn_code += "void *data" + str(i) + ", "
        code_str = code_str.replace("#_thead_csinn_args#", input_csinn_code)

        run_csinn_stats_anole = ""
        run_csinn_stats_thead = ""

        for i in range(input_num_code):
            run_csinn_stats_anole += "input[" + str(i) + "], "
            run_csinn_stats_thead += "input[" + str(i) + "], "

        for i in range(output_num_code):
            run_csinn_stats_thead += "output[" + str(i) + "], "

        if input_shape_list[0][1] == 1:
            is_rgb = 0
        else:
            is_rgb = 1
        if self.preprocess_params["pixel_format"] == "RGB":
            to_bgr = 0
        elif self.preprocess_params["pixel_format"] == "BGR":
            to_bgr = 1

        _is_rgb = str(is_rgb)
        _to_bgr = str(to_bgr)

        code_str = code_str.replace("#_is_rgb#", _is_rgb)
        code_str = code_str.replace("#_to_bgr#", _to_bgr)
        code_str = code_str.replace("#_anole_value_pass#", run_csinn_stats_anole)
        code_str = code_str.replace("#_thead_value_pass#", run_csinn_stats_thead)

        input_size_code = ""
        for ishape in input_shape_list:
            input_shape_str = list(map(str, ishape))
            input_shape_str = " * ".join(input_shape_str)
            input_size_code += input_shape_str + ", "
        code_str = code_str.replace("#_input_size_define#", input_size_code)

        output_size_code = ""
        for ishape in output_shape_list:
            output_shape_str = list(map(str, ishape))
            output_shape_str = " * ".join(output_shape_str)
            output_size_code += output_shape_str + ", "
        code_str = code_str.replace("#_output_size_define#", output_size_code)

        disable_preprocess = self.preprocess_params["disable"]
        if disable_preprocess:
            code_str = code_str.replace("#_preprocess_define_#", "#define preprocess(a, b, c)")
        else:
            code_str = self.preprocess_define(code_str, self.preprocess_params)

        code_str = code_str.replace("#_hhb_version_#", hhb_version())
        code_str = code_str.replace("#_model_name_define#", "network")
        logger.debug("save main souce to %s", os.path.join(output_path, self.main_source_name))
        save_to_file(code_str, os.path.join(output_path, self.main_source_name))

    def preprocess_define(self, code_str, preprocess_params):
        """ Generate macro definition """
        if len(preprocess_params["mean"]) not in (1, 3):
            raise HHBException(
                "do not know how to deal with mean values:{}".format(preprocess_params["mean"])
            )
        if len(preprocess_params["mean"]) == 1:
            preprocess_params["mean"] = preprocess_params["mean"] * 3
        preprocess_params_code = ""
        preprocess_params_code += (
            "#define RESIZE_HEIGHT" + "       " + str(preprocess_params["resize"][0]) + "\n"
        )
        preprocess_params_code += (
            "#define RESIZE_WIDTH" + "        " + str(preprocess_params["resize"][1]) + "\n"
        )
        preprocess_params_code += (
            "#define CROP_HEGHT" + "          " + str(preprocess_params["crop"][0]) + "\n"
        )
        preprocess_params_code += (
            "#define CROP_WIDTH" + "          " + str(preprocess_params["crop"][1]) + "\n"
        )
        preprocess_params_code += (
            "#define R_MEAN" + "              " + str(preprocess_params["mean"][2]) + "\n"
        )
        preprocess_params_code += (
            "#define G_MEAN" + "              " + str(preprocess_params["mean"][1]) + "\n"
        )
        preprocess_params_code += (
            "#define B_MEAN" + "              " + str(preprocess_params["mean"][0]) + "\n"
        )
        preprocess_params_code += (
            "#define SCALE" + "               " + str(preprocess_params["scale"]) + "\n"
        )

        preprocess_params_code += """
void preprocess(struct image_data *img, int is_rgb, int to_bgr)
{
    uint32_t new_height, new_width;
    uint32_t min_side;
    if (is_rgb) {
        im2rgb(img);
    }
    if (RESIZE_WIDTH == 0) {
        min_side = MIN(img->shape[0], img->shape[1]);
        new_height = (uint32_t) (img->shape[0] * (((float)RESIZE_HEIGHT) / (float)min_side));
        new_width = (uint32_t) (img->shape[1] * (((float)RESIZE_HEIGHT) / (float)min_side));
        imresize(img, new_height, new_width);
    } else {
        imresize(img, RESIZE_HEIGHT, RESIZE_WIDTH);
    }
    data_crop(img, CROP_HEGHT, CROP_WIDTH);
    sub_mean(img, R_MEAN, G_MEAN, B_MEAN);
    data_scale(img, SCALE);
    if(to_bgr) {
        imrgb2bgr(img);
    }
    imhwc2chw(img);
}
    """

        code_str = code_str.replace("#_preprocess_define_#", preprocess_params_code)
        return code_str


def driver_main_command(args):
    """Driver main commands"""
    command_level = 0
    if args.E:
        command_level = 10
    elif args.Q:
        command_level = 20
    elif args.C:
        command_level = 30
    elif args.simulate:
        command_level = 40

    if command_level == 0:
        raise HHBException("No command to execute...")

    # execute '-E' command
    from importer import import_model

    # args.output = ensure_dir(args.output)
    mod, params = import_model(
        args.model_file,
        args.model_format_main,
        args.input_name_main,
        args.input_shape_main,
        args.output_name_main,
    )
    if command_level == 10:
        if args.opt_level_main != -1:
            target = get_target(args.board_main)
            with tvm.transform.PassContext(opt_level=args.opt_level_main):
                mod, params = relay.optimize(mod, target=target, params=params)
    if (args.save_temps and command_level > 10) or command_level == 10:
        args.output = ensure_dir(args.output)
        save_relay_module(mod, params, args.output, HHBModel.RELAY)
    command_level -= 10
    if command_level == 0:
        return 0

    # execute '-Q' command
    if command_level == 10 and args.no_quantize:
        raise HHBException("can not set '-Q' and '--no_quantize' at the same time.")
    input_name_list, input_shape_list, _ = get_input_info_from_relay(mod, params)
    input_shape_dict = {name: shape for name, shape in zip(input_name_list, input_shape_list)}
    preprocess_params = PreprocessParams(
        mean=args.data_mean_main,
        scale=args.data_scale_main,
        resize_base=args.data_resize_main,
        pixel_format=args.pixel_format_main,
    )
    if not args.no_quantize:
        logger.debug("get calibrate dataset from %s", args.calibrate_dataset_main)
        dl = DatasetLoader(args.calibrate_dataset_main, input_shape_dict, preprocess_params)
        dataset = dl.get_data()
        dataset_list = []
        for d in dataset:
            dataset_list.append(d)
        from quantizer import get_quantize_config, quantize_model

        qconfig, _ = get_quantize_config(
            args.board_main,
            args.num_bit_activation_main,
            args.dtype_input_main,
            args.dtype_weight_main,
            args.calibrate_mode_main,
            args.quantized_type_main,
            args.weight_scale_main,
            args.fuse_relu_main,
            args.channel_quantization_main,
            args.broadcast_quantization_main,
        )

        qfunc = quantize_model(mod, params, qconfig, dataset_list)
        if (args.save_temps and command_level > 10) or command_level == 10:
            args.output = ensure_dir(args.output)
            save_relay_module(qfunc, None, args.output, HHBModel.QNN)
    command_level -= 10
    if command_level == 0:
        return 0

    # execute '-C' command
    if args.board_main == "x86_ref":
        if args.no_quantize:
            hhb_model = HHBCodegenX86FloatModel()
            graph_module = hhb_model.convert(mod, params, args.board_main, args.opt_level_main)
            if (args.save_temps and command_level > 10) or command_level == 10:
                args.output = ensure_dir(args.output)
                hhb_model.save_model(graph_module, args.output, save_info=False)
        else:
            hhb_model = HHBCodegenX86QuantModel()
            lib = hhb_model.convert(qfunc, None, args.board_main, args.opt_level_main, args.output)
            if (args.save_temps and command_level > 10) or command_level == 10:
                args.output = ensure_dir(args.output)
                hhb_model.save_model(lib, args.output, save_info=False)
    elif args.board_main in ("anole", "c860"):
        if args.no_quantize:
            raise HHBException("can not set '--board anole' and '--no-quantize' at the same time.")
        hhb_model = HHBCodegenAnoleModel()
        hhb_model.preprocess_params = preprocess_params.__dict__
        args.output = ensure_dir(args.output)
        hhb_model.convert(
            qfunc,
            None,
            args.board_main,
            args.opt_level_main,
            args.output,
            args.postprocess_main,
            args.disable_binary_graph_main,
        )

        # save part data in calibrate dataset into tensor file
        data_count = 0
        for k, v in dataset_list[0].items():
            safe_k = k.replace("/", "_")
            v.tofile(os.path.join(args.output, safe_k + ".{}.tensor".format(data_count)), "\n")
            data_count += 1
    else:
        raise HHBException("unsupport for board: {}".format(args.board_main))
    command_level -= 10
    if command_level == 0:
        return 0

    # execute '--simulate' command
    if args.board_main in ("anole", "c860"):
        raise HHBException("can not simulate anole or c860.")
    ctx = tvm.cpu(0)
    if args.no_quantize:
        m = graph_runtime.GraphModule(graph_module["default"](ctx))
    else:
        m = hhb_runtime.create(lib, qfunc, ctx, output_dir=args.output)
        m.set_params(os.path.join(args.output, hhb_model.params_name))
    dl = DatasetLoader(args.simulate_data_main, input_shape_dict, preprocess_params)
    dataset = dl.get_data()
    index = 0
    for data in dataset:
        m.run(**data)
        for i in range(m.get_num_outputs()):
            output = m.get_output(i).asnumpy()
            out = np.reshape(output, [np.prod(output.size)])
            output_prefix = os.path.basename(dl.file_path[index]) + "_output_" + str(i) + ".tensor"
            default_output_path = os.path.join(args.output, output_prefix)
            if args.postprocess_main == "top5":
                print_top5(out, str(i))
            elif args.postprocess_main == "save":
                ensure_dir(args.output)
                np.savetxt(default_output_path, out, delimiter="\n", newline="\n")
            else:
                print_top5(out, str(i))
                ensure_dir(args.output)
                np.savetxt(default_output_path, out, delimiter="\n", newline="\n")
        index += 1
