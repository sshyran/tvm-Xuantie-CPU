#!/usr/bin/env python

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
""" HHB Command Line Tools """
import argparse
import logging
import sys
import os

# pylint: disable=wrong-import-position
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from utils import hhb_version, HHBException
from utils import HHB_REGISTERED_PARSER
from utils import import_module_for_register
from utils import add_main_argument
from utils import add_import_argument
from utils import add_optimize_argument
from utils import add_preprocess_argument
from utils import add_quantize_argument
from utils import add_common_argument
from utils import add_simulate_argument
from utils import add_postprocess_argument
from utils import add_codegen_argument
from utils import wraper_for_parameters
from utils import generate_config_file
from utils import read_params_from_file
from utils import get_set_arguments
from utils import update_cmd_params
from utils import check_cmd_arguments
from utils import check_subcommand
from hhb_models import driver_main_command


def _main(argv):
    """ HHB commmand line interface. """

    parser = argparse.ArgumentParser(
        prog="HHB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="HHB command line tools",
        epilog=__doc__,
        allow_abbrev=False,
        add_help=False,
    )

    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity")
    parser.add_argument("-h", "--help", action="store_true", help="Show this help information")
    parser.add_argument("--version", action="store_true", help="Print the version and exit")
    parser.add_argument(
        "-f",
        "--model-file",
        nargs="+",
        metavar="",
        help="Path to the input model file, can pass multi files",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="",
        default="hhb_output",
        help="The directory that holds the outputs.",
    )

    parameter_dict = {}
    wraper_for_parameters(add_main_argument, parser, None, "main", parameter_dict)
    wraper_for_parameters(add_import_argument, parser, False, "import", parameter_dict)
    wraper_for_parameters(add_optimize_argument, parser, False, "optimize", parameter_dict)
    wraper_for_parameters(add_quantize_argument, parser, False, "quantize", parameter_dict)
    wraper_for_parameters(add_preprocess_argument, parser, False, "preprocess", parameter_dict)
    wraper_for_parameters(add_common_argument, parser, False, "common", parameter_dict)
    wraper_for_parameters(add_simulate_argument, parser, False, "simulate", parameter_dict)
    wraper_for_parameters(add_postprocess_argument, parser, False, "postprocess", parameter_dict)
    wraper_for_parameters(add_codegen_argument, parser, False, "codegen", parameter_dict)

    if check_subcommand(argv):
        subparser = parser.add_subparsers(title="commands")
        for make_subparser in HHB_REGISTERED_PARSER:
            make_subparser(subparser)

    args = parser.parse_args(argv)

    if args.verbose > 4:
        args.verbose = 4

    logging.basicConfig()
    logger = logging.getLogger("HHB")
    logger.setLevel(40 - args.verbose * 10)

    if args.help:
        # print subcommand info
        subparser = parser.add_subparsers(title="commands")
        for make_subparser in HHB_REGISTERED_PARSER:
            make_subparser(subparser)
        parser.print_help()
        return 0

    if args.version:
        sys.stdout.write("{}\n".format(hhb_version()))
        return 0

    if args.config_file_main:
        update_cmd_params(args.config_file_main, parser, args, is_subcommand=False)

    if args.generate_config:
        generate_config_file(parameter_dict, args.output)
        return 0

    if args.E or args.Q or args.C or args.simulate:
        # execute prgram using main arguments
        driver_main_command(args)
        return 0

    if not hasattr(args, "config_file"):
        raise HHBException("do not execute any command...")
    if args.config_file:
        curr_command = args.func.__name__.split("_")[-1]
        update_cmd_params(
            args.config_file, parser, args, is_subcommand=True, commands=[curr_command, "fakefile"]
        )

    assert hasattr(args, "func"), "Error: missing 'func' attribute for subcommand {}".format(argv)

    try:
        return args.func(args)
    except HHBException as err:
        sys.stderr.write("Error: {}".format(err))
        return 4


def main():
    import_module_for_register()

    check_cmd_arguments(sys.argv[1:])
    sys.exit(_main(sys.argv[1:]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl-C detected.")
