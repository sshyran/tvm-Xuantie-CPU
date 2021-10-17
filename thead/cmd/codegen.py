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

from utils import hhb_register_parse
from utils import HHBException
from utils import HHBModel
from utils import add_optimize_argument
from utils import add_common_argument
from utils import add_postprocess_argument
from utils import add_codegen_argument
from utils import ensure_dir
from hhb_models import HHBCodegenX86FloatModel
from hhb_models import HHBCodegenX86QuantModel
from hhb_models import HHBCodegenAnoleModel


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


@hhb_register_parse
def add_codegen_parser(subparsers):
    """ Include parser for 'codegen' subcommand """

    parser = subparsers.add_parser("codegen", help="Codegen the imported model")
    parser.set_defaults(func=driver_codegen)

    add_optimize_argument(parser)
    add_postprocess_argument(parser)
    add_common_argument(parser)

    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        default="model_codegen",
        help="The directory that holds the codegen files.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("FILE", help="Directory to the model file")


def driver_codegen(args):
    """Driver codegen command"""
    if not os.path.exists(args.FILE) or not os.path.isdir(args.FILE):
        raise HHBException("The directory is not exists: {}".format(args.FILE))
    model_type = HHBModel.guess_model(args.FILE)

    args.output = ensure_dir(args.output)
    if args.board == "x86_ref":
        if model_type == HHBModel.RELAY:
            hhb_model = HHBCodegenX86FloatModel()
            mod, params = hhb_model.load_model(args.FILE)
            graph_module = hhb_model.convert(mod, params, args.board, args.opt_level)
            hhb_model.save_model(graph_module, args.output)
        elif model_type == HHBModel.QNN:
            hhb_model = HHBCodegenX86QuantModel()
            mod, params = hhb_model.load_model(args.FILE)
            hhb_model.set_quant_env(args.FILE)
            lib = hhb_model.convert(mod, params, args.board, args.opt_level, args.output)
            hhb_model.save_model(lib, args.output)
        else:
            raise HHBException("unsupport for {} module".format(HHBModel.TYPE2NAME[model_type]))
    elif args.board in ("anole", "c860"):
        hhb_model = HHBCodegenAnoleModel()
        mod, params = hhb_model.load_model(args.FILE)
        hhb_model.set_quant_env(args.FILE)
        hhb_model.convert(
            mod,
            params,
            args.board,
            args.opt_level,
            args.output,
            args.postprocess,
            args.disable_binary_graph,
        )
    else:
        raise HHBException("unsupport for board: {}".format(args.board))
