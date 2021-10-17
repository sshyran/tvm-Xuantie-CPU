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
Import networks into relay IR.
"""
import logging

import tvm
from tvm import relay

from utils import hhb_register_parse
from utils import parse_node_name
from utils import parse_node_shape
from utils import HHBModel
from utils import add_import_argument
from utils import add_optimize_argument
from utils import add_common_argument
from utils import save_relay_module
from utils import get_target
from frontends import get_frontend_by_name
from frontends import guess_frontend


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


@hhb_register_parse
def add_import_parser(subparsers):
    """ Include parser for 'import' subcommand """

    parser = subparsers.add_parser("import", help="Import a model into relay ir")
    parser.set_defaults(func=driver_import)

    add_import_argument(parser)
    add_optimize_argument(parser)
    add_common_argument(parser)

    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        default="model_relay",
        help="The directory that holds the relay ir.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument(
        "FILE", nargs="+", help="Path to the input model file, can pass multi files"
    )


def driver_import(args):
    """Driver import command"""
    mod, params = import_model(
        args.FILE,
        args.model_format,
        args.input_name,
        args.input_shape,
        args.output_name,
    )
    if args.opt_level != -1:
        target = get_target(args.board)
        with tvm.transform.PassContext(opt_level=args.opt_level):
            mod, params = relay.optimize(mod, target=target, params=params)

    save_relay_module(mod, params, args.output, HHBModel.RELAY)
    return 0


def import_model(path, model_format=None, input_name=None, input_shape=None, output_name=None):
    """Import a model from a supported framework into relay ir.

    Parameters
    ----------
    path : list[str]
        Path to a model file. There may be two files(.caffemodel, .prototxt) for Caffe model
    model_format : str, optional
        A string representing a name of a frontend to be used
    input_name : str, optional
        The names of input node in the graph, which is seperated by semicolon(;) if multi-input
    input_shape : str, optional
        The shape of input node in the graph, which is seperated by semicolon(;) if multi-input
    output_name : str, optional
        The name of output node in the graph, which is seperated by semicolon(;) if multi-output

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    input_name = parse_node_name(input_name)
    input_shape = parse_node_shape(input_shape)
    output_name = parse_node_name(output_name)

    if model_format is not None:
        frontend = get_frontend_by_name(model_format)
    else:
        frontend = guess_frontend(path)
    if input_name and input_shape:
        assert len(input_name) == len(
            input_shape
        ), "The length of \
                input_name must be equal to that of input_shape."
    mod, params = frontend.load(path, input_name, input_shape, output_name)

    return mod, params
