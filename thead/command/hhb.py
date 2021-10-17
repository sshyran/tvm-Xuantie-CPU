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
from argparse import ArgumentParser
import logging

import utils as _utils


def get_command_args():
    """ Parse the command line arguments """
    options = ArgumentParser(description="HHB command line tools.", allow_abbrev=False)

    # main commands
    group1 = options.add_mutually_exclusive_group()
    group1.add_argument("-E", action="store_true", help="Convert model into relay ir(mod, params).")
    group1.add_argument(
        "-Q", action="store_true", help="Convert model into relay ir which has been quantized."
    )
    group1.add_argument(
        "-S",
        action="store_true",
        help="Convert model into (graph, lib, params) which has "
        "been optimized but is not generated into binary libs or C files.",
    )
    group1.add_argument("-C", action="store_true", help="Convert model into (graph, lib, parmas).")
    group1.add_argument(
        "--simulate", action="store_true", help="Simulate model on x86 llvm device."
    )
    group1.add_argument(
        "--deploy", action="store_true", help="Generate C code running on the specified target."
    )
    group1.add_argument(
        "--generate-input", action="store_true", help="Generate input data after preprocessing."
    )

    options.add_argument("-v", action="store_true", help="Print complte commands into stderr.")
    options.add_argument(
        "--version", action="version", version=_utils.hhb_version(), help="Show the version info"
    )

    options.add_argument(
        "-P",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
        #                        help="Print model profile info into file which is specified"
        #                        " by '-P' when '-P' is not none."
    )

    options.add_argument(
        "--save-temps", action="store_true", help="Save temp files if '--save-temps' is set."
    )
    options.add_argument(
        "-o", type=str, default=None, help="Place outputs into specific directory."
    )
    options.add_argument(
        "--no-quantize", action="store_true", help="If set, don't quantize the model."
    )
    # model args
    group3 = options.add_mutually_exclusive_group()
    group3.add_argument(
        "--tf-pb", type=str, default=None, help="The path of Tensorflow model file(.pb)."
    )
    group3.add_argument(
        "--caffe-proto", type=str, default=None, help="The path of Caffe model file(.protofile)."
    )
    options.add_argument(
        "--caffe-blobs", type=str, default=None, help="The path of Caffe model file(.caffemodel)."
    )
    options.add_argument(
        "--input-name",
        type=str,
        default=None,
        help="Set the name of input node. If '--input-name' "
        "is None, default value is 'Placeholder'. Multiple values "
        "are separated by semicolon(;).",
    )
    options.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Set the shape of input nodes. Multiple shapes are "
        "separated by semicolon(;) and the dims between shape are "
        "separated by space.",
    )
    options.add_argument(
        "--output-name",
        type=str,
        help="Set the name of output nodes. Multiple shapes are " "separated by semicolon(;).",
    )
    options.add_argument(
        "--quantized-type",
        type=str,
        help=argparse.SUPPRESS,
        #                       help="Select the algorithm of quantization (asym, sym)."
    )
    options.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Provide with dataset for calibration in quantization step. "
        "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
        "of images. Note: only one image path in one line if .txt.",
    )
    options.add_argument(
        "--input-dataset",
        type=str,
        default=None,
        help="Provide with dataset for the input of model in reference step. "
        "Support dir or .npz .jpg .png .JPEG or .txt in which there are path "
        "of images. Note: only one image path in one line if .txt.",
    )
    group4 = options.add_mutually_exclusive_group()
    group4.add_argument(
        "--target",
        type=str,
        default="llvm -mtriple=csky -mcpu=c860 -mfloat-abi=hard -device=c860 --system-lib",
        help=argparse.SUPPRESS,
    )
    group4.add_argument(
        "--board",
        type=str,
        default="anole",
        choices=["anole", "c860", "x86_ref"],
        help="Set target device which codes generated. Defualt is 'anole'",
    )
    options.add_argument(
        "--input-normal", type=float, default="1", help="divide number for inputs normalization."
    )
    options.add_argument(
        "--input-mean",
        type=str,
        default="0",
        help="Set the mean value of input. Multiple values are separated by space.",
    )
    options.add_argument(
        "--resize-base",
        type=float,
        default=None,
        help="Resize base size for input image to resize.",
    )
    options.add_argument(
        "--postprocess",
        type=str,
        default="save",
        choices=["top5", "save", "save_and_top5"],
        help="Set the mode of postprocess: "
        "'top5' show top5 of output; "
        "'save' save output to file."
        "'save_and_top5' show top5 and save output to file.",
    )
    options.add_argument(
        "--debug",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    options.add_argument("--accuracy-test", action="store_true", help=argparse.SUPPRESS)

    args = options.parse_args()
    return args


def main():
    """ Entry point """
    cmd_args = get_command_args()
    if cmd_args.v:
        logging.basicConfig(style="{", format="{levelname}: {message}", level=logging.INFO)
        print("The command line args before executing the script:")
        print(cmd_args)
    elif cmd_args.debug:
        logging.basicConfig(
            style="{", format="{asctime} {levelname}: {message}", level=logging.DEBUG
        )
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(style="{", format="{levelname}: {message}", level=logging.WARNING)

    _utils.check_cmd_args(cmd_args)
    wksp_args = _utils.WorkspaceArgs(cmd_args)

    if cmd_args.E:
        _utils.command_E(wksp_args)
    elif cmd_args.Q:
        _utils.command_Q(wksp_args)
    elif cmd_args.S:
        _utils.command_S(wksp_args)
    elif cmd_args.C:
        _utils.command_C(wksp_args)
    elif cmd_args.simulate:
        _utils.command_simulate(wksp_args)
    elif cmd_args.deploy:
        _utils.command_deploy(wksp_args)
    elif cmd_args.generate_input:
        _utils.command_generate_input(wksp_args)
    else:
        _utils.command_deploy(wksp_args)
    if cmd_args.P:
        print("Print model profile info into {}".format(cmd_args.P))
        ################################################
        #                   (todo) generate profile
        ################################################

    if cmd_args.v:
        print("The command line args after executing the script:")
        wksp_args.print_cmd_args()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCtrl-C detected.")
