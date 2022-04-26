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
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_HLIGHT_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_HLIGHT_H_

#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {

class CodegenHLight : public CodegenCSINN {
 public:
  CodegenHLight() : CodegenCSINN() {
    auto qs = cfg->quantization_scheme;
    if (qs == "CSINN_QUANT_UINT8_ASYM") {
      base_dtype_ = "CSINN_DTYPE_UINT8";
    } else if (qs == "CSINN_QUANT_INT8_SYM") {
      base_dtype_ = "CSINN_DTYPE_INT8";
    } else if (qs == "CSINN_QUANT_INT8_ASYM") {
      base_dtype_ = "CSINN_DTYPE_INT8";
    } else if (qs == "CSINN_QUANT_INT16_SYM") {
      base_dtype_ = "CSINN_DTYPE_INT16";
    } else if (qs == "CSINN_QUANT_FLOAT16") {
      base_dtype_ = "CSINN_DTYPE_FLOAT16";
    } else if (qs == "CSINN_QUANT_BFLOAT16") {
      base_dtype_ = "CSINN_DTYPE_BFLOAT16";
    } else {
      base_dtype_ = "CSINN_DTYPE_FLOAT32";
    }
    target_op_list = {"qnn.csi.conv2d",
                      "qnn.csi.concatenate",
                      "qnn.csi.relu",
                      "qnn.csi.dense",
                      "qnn.csi.avgpool2d",
                      "qnn.csi.maxpool2d",
                      "qnn.csi.add",
                      "qnn.csi.clip",
                      "qnn.csi.upsampling",
                      "qnn.csi.mean",
                      "qnn.csi.mul",
                      "qnn.csi.bias_add",
                      "qnn.csi.deconv2d",
                      "qnn.csi.global_avgpool2d",
                      "qnn.csi.global_maxpool2d",
                      "qnn.csi.leaky_relu",
                      "qnn.csi.sigmoid",
                      "qnn.csi.split",
                      "qnn.csi.strided_slice",
                      "qnn.csi.transpose",
                      "qnn.csi.reshape"};
  }
  virtual ~CodegenHLight() {}

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
  virtual void GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo);
  void SessionRunMode() { PrintOneLine(code_stream_, "sess->base_run_mode = CSINN_RM_CPU_GRAPH;"); }
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_HLIGHT_H_
