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
Show the imported model.
"""
import logging
import tarfile

import tvm

from utils import hhb_register_parse
from utils import HHBException
from utils import HHBModel


# pylint: disable=invalid-name
logger = logging.getLogger("HHB")


# @hhb_register_parse
def add_visualize_parser(subparsers):
    """ Include parser for 'show' subcommand """

    parser = subparsers.add_parser("show", help="Show the imported model")
    parser.set_defaults(func=driver_show)
    parser.add_argument(
        "-o", "--output", default=None, help="Output the visualized model into text."
    )
    parser.add_argument(
        "--no-print", action="store_true", help="Do not print the structure in the console."
    )
    parser.add_argument("FILE", help="Path to the model file")


def driver_show(args):
    """Driver show command"""
    model_type = HHBModel.guess_model(args.FILE)
    if model_type is None:
        raise HHBException("unspport for showing {}".format(args.FILE))

    logger.debug("extracting module file %s", args.FILE)
    t = tarfile.open(args.FILE)
    model_type_str = HHBModel.TYPE2NAME[model_type]
    module_file = t.extractfile(model_type_str + ".txt")
    mod = tvm.parser.fromtext(module_file.read())
    if args.output:
        with open(args.output, "w") as f:
            f.write(mod.astext())
    if not args.no_print:
        print(mod["main"])
