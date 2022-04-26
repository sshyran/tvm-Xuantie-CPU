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
"""
Codegen the imported model.
"""
import logging
import os

from core.arguments_manage import (
    add_optimize_argument,
    add_common_argument,
    add_postprocess_argument,
    add_codegen_argument,
    ArgumentFilter,
)
from core.common import (
    hhb_register_parse,
    HHBException,
    ensure_dir,
    generate_config_file,
    ALL_ARGUMENTS_DESC,
    collect_arguments_info,
    AttributeDict,
)
from core.hhbir_manage import (
    guess_ir_type,
    HHBIRType,
    HHBRelayIR,
    HHBQNNIR,
    HHBFloatCodegenIR,
    HHBX86QnnCodegenIR,
    HHBBoardQnnCodegenIR,
    get_input_info_from_relay,
    get_output_info_from_relay,
)
from core.codegen_manage import collect_codegen_config, set_codegen_config
from core.preprocess_manage import (
    collect_preprocess_config,
    set_preprocess_params,
)
from core.quantization_manage import (
    collect_quantization_config,
    set_quantize_params_by_board,
    get_quantize_config,
)

# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


@hhb_register_parse
def add_codegen_parser(subparsers):
    """ Include parser for 'codegen' subcommand """

    parser = subparsers.add_parser("codegen", help="Codegen the imported model")
    parser.set_defaults(func=driver_codegen)

    add_optimize_argument(parser)
    add_postprocess_argument(parser)
    add_codegen_argument(parser)
    add_common_argument(parser)

    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        default="hhb_out",
        help="The directory that holds the codegen files.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("FILE", help="Directory to the model file")

    ALL_ARGUMENTS_DESC["codegen"] = collect_arguments_info(parser._actions)


def driver_codegen(args_filter: ArgumentFilter):
    """Driver codegen command"""
    # filter arguments and prepare all needed args
    all_filters = [collect_codegen_config]
    args_filter.filter_argument(all_filters)

    args = args_filter.filtered_args
    if not os.path.exists(args.FILE) or not os.path.isdir(args.FILE):
        raise HHBException("The directory is not exists: {}".format(args.FILE))
    model_type = guess_ir_type(args.FILE)
    logger.debug("Infer the ir type: %s in %s" % (HHBIRType.TYPE2NAME[model_type], args.FILE))

    args.output = ensure_dir(args.output)

    if args.generate_config:
        generate_config_file(os.path.join(args.output, "cmd_codegen_params.yml"))

    if args.board == "x86_ref":
        if model_type == HHBIRType.RELAY:
            # get relay ir
            relay_ir = HHBRelayIR()
            relay_ir.load_model(args.FILE)
            input_module = relay_ir.get_model()
            # convert to float codegen ir
            float_codegen_ir = HHBFloatCodegenIR()
            float_codegen_ir.convert(input_module, args.board, args.opt_level)
            float_codegen_ir.save_model(args.output)
        elif model_type == HHBIRType.QNN:
            # get qnn ir
            qnn_ir = HHBQNNIR()
            qnn_ir.load_model(args.FILE)
            input_module = qnn_ir.get_model()
            # convert to x86 qnn codegen ir
            x86_qnn_codegen_ir = HHBX86QnnCodegenIR()
            quantize_config = x86_qnn_codegen_ir.get_quant_env(
                os.path.join(args.FILE, qnn_ir.info_file)
            )
            x86_qnn_codegen_ir.convert(
                input_module, args.board, args.opt_level, args.output, quantize_config
            )
            x86_qnn_codegen_ir.save_model(args.output)
        else:
            raise HHBException("unsupport for IR type: {}".format(HHBIRType.TYPE2NAME[model_type]))
    elif args.board in (
        "anole",
        "light",
        "hlight",
        "asp",
        "i805",
        "c860",
        "c906",
        "ch8601",
        "dp1k",
    ):
        if model_type != HHBIRType.QNN:
            raise HHBException(
                "IR type: {} can not be converted to board codegen".format(
                    HHBIRType.TYPE2NAME[model_type]
                )
            )
        # get qnn ir
        qnn_ir = HHBQNNIR()
        qnn_ir.load_model(args.FILE)
        input_module = qnn_ir.get_model()

        _, input_shape, _ = get_input_info_from_relay(*input_module)
        output_shape, _ = get_output_info_from_relay(input_module[0], input_module[1])

        all_filters = [
            collect_codegen_config,
            set_codegen_config,
        ]
        extra_args = AttributeDict()
        extra_args.input_shape = input_shape
        extra_args.input_num = len(input_shape)
        extra_args.output_num = len(output_shape)
        extra_args.model_save = args.model_save
        args_filter.filter_argument(all_filters, extra=extra_args)
        args = args_filter.filtered_args

        light_input_fix_size = qnn_ir.info_dict["preprocess"]["light_input_fix_size"]

        # convert to board qnn codegen ir
        board_qnn_codegen_ir = HHBBoardQnnCodegenIR()
        quantize_config = board_qnn_codegen_ir.get_quant_env(
            os.path.join(args.FILE, qnn_ir.info_file)
        )

        if len(light_input_fix_size) == 2:
            quantize_config["light_input_fix_height"] = light_input_fix_size[0]
            quantize_config["light_input_fix_width"] = light_input_fix_size[1]
        quantize_config["trace_strategy"] = args.codegen_config.trace_strategy
        quantize_config["input_memory_type"] = args.codegen_config.input_memory_type
        quantize_config["output_memory_type"] = args.codegen_config.output_memory_type

        board_qnn_codegen_ir.convert(
            input_module,
            args.board,
            args.opt_level,
            args.output,
            args.verbose,
            quantize_config,
            args.codegen_config.multithread,
            args.codegen_config.model_save,
        )

        board_qnn_codegen_ir.save_model(
            input_shape,
            output_shape,
            args.board,
            args.output,
            args.postprocess,
            args.codegen_config.model_save,
            args.codegen_config.without_preprocess,
            qnn_ir.info_dict["preprocess"],
            args.codegen_config.multithread,
            args.codegen_config.input_memory_type,
            quantize_config["quantization_scheme"],
            args.codegen_config,
        )
    else:
        raise HHBException("unsupport for board: {}".format(args.board))
