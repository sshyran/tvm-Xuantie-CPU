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
 * \file src/relay/backend/contrib/csinn/light.h
 * \brief The base class for light.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_ASP_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_ASP_H_

#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {

class CodegenASP : public CodegenCSINN {
 public:
  CodegenASP() : CodegenCSINN() {
    base_dtype_ = "CSINN_DTYPE_INT8";
    target_op_list = {"qnn.csi.conv2d", "qnn.csi.dense", "qnn.csi.avgpool2d", "qnn.csi.maxpool2d"};
  }
  virtual ~CodegenASP() {}

  virtual void CreateTensor(string name, string data, std::vector<int> shape,
                            QuantParams quant_params, string dtype);
  virtual void params_common_setup(std::ostringstream& decl, const CallNode* call, string op_name,
                                   string params_name, string layer_name, string layout);

  void malloc_buf(string out, int out_size) {}
  void CreateMallocBuf(string name, std::vector<int> shape, string dtype) {}
  void CreateTensorSessData() {}
  void CreateHybridTensorSessData(std::vector<int> shape, string dtype) {}
  void FreeTensor(const Expr& expr, string name) {}
  void EmitSessionSetup();
  void ModelBinarySave();
  void SessionRunMode() { PrintOneLine(code_stream_, "sess->base_run_mode = CSINN_RM_CPU_GRAPH;"); }
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_ASP_H_
