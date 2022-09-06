/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/csinn/gref.cc
 * \brief Implementation of CSINN gref codegen APIs.
 */

#include "gref.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

string CodegenGref::EmitGraph(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  EmitSessionRun();
  DumpConstant();
  return func_def_.str();
}

bool CodegenGref::IsIntegralOrNot(string const_kind) {
  std::vector<string> per_channel = {"conv_kernel", "dense_kernel", "depthwise_kernel",
                                     "conv_bias",   "dense_bias",   "depthwise_bias"};
  if ((cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM" ||
       cfg->quantization_scheme == "CSINN_QUANT_INT4_ASYM_W_SYM") &&
      !is_contain_item<string>(per_channel, const_kind)) {
    return true;
  }
  return false;
}

#if 0
void CodegenGref::GetAsymScale(float min_value, float max_value, int bits, Qinfo* qinfo) {
  int valid_range = std::pow(2, bits) - 1;
  qinfo->scale = (max_value - min_value) / valid_range;
  if (qinfo->scale == 0) {
    qinfo->scale = std::abs(max_value);
  }
  qinfo->zero_point =
      std::min(valid_range,
               static_cast<int>(std::max(-127.0f, std::round(-127.0f - min_value / qinfo->scale))));
}
#endif
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
