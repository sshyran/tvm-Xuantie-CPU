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
 * \file src/relay/backend/contrib/csinn/light.cc
 * \brief Implementation of CSINN light codegen APIs.
 */

#include "hlight.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

void CodegenHLight::CreateTensor(string name, string data, std::vector<int> shape,
                                 QuantParams quant_params, string dtype) {
  std::ostringstream t0;
  t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
  PushDeclLine(t0);
  t0 << name << "->name = "
     << "\"" << name << "\"";
  PushDeclLine(t0);
  t0 << name << "->layout = " << GetCSINNActLayout(shape);
  PushDeclLine(t0);
  SetDim(name, shape);
  t0 << name << "->qinfo = (struct csi_quant_info *)(params_base + " << to_string(constant_offset)
     << ")";
  PushDeclLine(t0);
  t0 << name << "->quant_channel = " << to_string(quant_params.q_size);
  PushDeclLine(t0);
  constant_offset += quant_params.q_size * sizeof(Qinfo);
  qinfo_list_.push_back(quant_params);
  flag_list_.push_back(QINFO);
}

void CodegenHLight::params_common_setup(std::ostringstream& decl, const CallNode* call,
                                        string op_name, string params_name, string layer_name,
                                        string layout = "") {
  std::ostringstream t0;
  t0 << params_name << "->base.layout = CSINN_LAYOUT_NCHW";
  PushDeclLine(t0);
  t0 << params_name << "->base.name = "
     << "\"" << op_name + "_" + layer_name << "_" << params_idx_ << "\"";
  PushDeclLine(t0);
  params_idx_++;
  if (InOpList(call)) {
    t0 << params_name << "->base.api = CSINN_LIGHT";
    PushDeclLine(t0);
  }
  t0 << "csi_" << op_name << "_init" << decl.str();
  PushDeclLine(t0);
}

void CodegenHLight::GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo) {
  int valid_range = std::pow(2, bits - 1) - 1;
  float abs_max = std::max(std::abs(min_value), std::abs(max_value));
  float scale = valid_range / abs_max;
  int exponent;
  frexp(scale, &exponent);
  qinfo->scale = 1.0f / std::pow(2, exponent - 1);
  qinfo->zero_point = 1;
  qinfo->max = 127 * qinfo->scale;
  qinfo->min = -128 * qinfo->scale;
}

void CodegenHLight::EmitSessionSetup(void) {
  std::ostringstream t0;
  t0 << "void *" << ext_func_id_ << "_(";
  t0 << "char *params_base) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
  SessionRunMode();
  ModelBinarySave();
  t0 << "sess->base_api = " << target_name_ << ";";
  PrintOneLine(code_stream_, t0);
  t0 << "sess->base_dtype = " << base_dtype_ << ";";
  PrintOneLine(code_stream_, t0);
  if (debug_level_ == "INFO") {
    PrintOneLine(code_stream_, "sess->debug_level = CSI_DEBUG_LEVEL_INFO;");
  }
  PrintOneLine(code_stream_, "csi_session_init(sess);");

  t0 << "csi_set_input_number(" << ext_func_args_.size() << ", sess);";
  PrintOneLine(code_stream_, t0);
  t0 << "csi_set_output_number(" << output_list_.size() << ", sess);";
  PrintOneLine(code_stream_, t0);
  // Function body
  PrintNewLine(code_stream_);
  for (auto decl : buf_decl_) {
    PrintOneLine(code_stream_, decl);
  }
  PrintNewLine(code_stream_);
  for (uint32_t i = 0; i < ext_func_args_.size(); i++) {
    std::string new_name = CodegenCSINN::replace(ext_func_args_[i]->name_hint());
    auto iter = io_nodes.find(new_name);
    if (iter == io_nodes.end()) {
      CHECK(0);
    }
    QuantParams q_params = iter->second;
    string in_name = q_params.name;
    std::ostringstream t1;
    t1 << "csi_set_tensor_entry(" << in_name << ", sess);\n";
    PrintIndents(t1);
    t1 << "csi_set_input(" << i << ", " << in_name << ", sess);";
    PrintOneLine(code_stream_, t1);
  }

  PrintNewLine(code_stream_);
  for (auto stmt : ext_func_body) {
    PrintOneLine(code_stream_, stmt);
  }

  int output_index = 0;
  // emit normal outputs
  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (!output_list_[i].is_const) {
      string output_name = output_list_[i].name;
      t0 << "csi_set_output(" << output_index++ << ", " << output_name << ", sess);";
      PrintOneLine(code_stream_, t0);
    }
  }

  // emit constant outputs
  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].is_const) {
      t0 << output_list_[i].name << "->name = "
         << "\"" << output_list_[i].name << "\";";
      PrintOneLine(code_stream_, t0);
      t0 << output_list_[i].name << "->dtype = CSINN_DTYPE_FLOAT32;";
      PrintOneLine(code_stream_, t0);
      t0 << output_list_[i].name << "->is_const = 1;";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_set_output(" << output_index++ << ", " << output_list_[i].name << ", sess);";
      PrintOneLine(code_stream_, t0);
    }
  }

  auto ctx = transform::PassContext::Current();
  auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
  auto opt_cfg = opt.value();

  double fix_height = opt_cfg->light_input_fix_height;
  double fix_width = opt_cfg->light_input_fix_width;
  if (fix_height != 0) {
    t0 << "csi_pnna_set_input_strides(sess, 1, " << fix_height << " ," << fix_width << ");";
    PrintOneLine(code_stream_, t0);
  }

  PrintNewLine(code_stream_);
  PrintOneLine(code_stream_, "csi_session_setup(sess);");
  PrintOneLine(code_stream_, "return sess;");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

void CodegenHLight::ModelBinarySave() {
  std::ostringstream t0;
  t0 << "sess->model_name = \"csi.mbs.bin\";";
  PrintOneLine(code_stream_, t0);
  t0 << "sess->base_quant_type = " << cfg->quantization_scheme << ";";
  PrintOneLine(code_stream_, t0);
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
