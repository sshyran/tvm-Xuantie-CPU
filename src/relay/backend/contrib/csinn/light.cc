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

#include "light.h"

#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

void CodegenLight::VisitExpr_(const CallNode* call) {
  /* Get the arguments for various CSINN kernels. */
  /* QNN op */
  if (first_visit_expr) {
    first_visit_expr = false;
    Output output;
    output.call = call;
    output_list_.push_back(output);
  }
  if (IsOp(call, "qnn.csi.add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.argmax")) {
    Reduce(call, "argmax", "int32_t");
  } else if (IsOp(call, "qnn.csi.avgpool2d")) {
    AvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.batch_to_space_nd")) {
    BatchToSpaceND(call);
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.clip")) {
    Clip(call);
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    Concat(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2d(call, "conv2d");
  } else if (IsOp(call, "qnn.csi.conv2d_relu")) {
    Conv2d(call, "conv2d_relu");
  } else if (IsOp(call, "qnn.csi.conv2d_relu6")) {
    Conv2d(call, "conv2d_relu6");
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2d(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    Dense(call);
  } else if (IsOp(call, "qnn.csi.depth_to_space")) {
    DepthToSpace(call);
  } else if (IsOp(call, "qnn.csi.exp")) {
    Unary(call, "exp");
  } else if (IsOp(call, "qnn.csi.flatten")) {
    Flatten(call);
  } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
    GlobalAvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.global_maxpool2d")) {
    GlobalMaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    LeakyRelu(call);
  } else if (IsOp(call, "qnn.csi.lrn")) {
    LRN(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d")) {
    MaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_locat")) {
    MaxPool2dLocat(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_with_argmax")) {
    Maxpool2dWithArgmax(call);
  } else if (IsOp(call, "qnn.csi.mean")) {
    Reduce(call, "mean", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOp(call, "mul");
  } else if (IsOp(call, "qnn.csi.div")) {
    DisoOp(call, "div");
  } else if (IsOp(call, "qnn.csi.minimum")) {
    DisoOp(call, "minimum");
  } else if (IsOp(call, "qnn.csi.maximum")) {
    DisoOp(call, "maximum");
  } else if (IsOp(call, "qnn.csi.subtract")) {
    DisoOp(call, "sub");
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PRelu(call);
  } else if (IsOp(call, "qnn.csi.pad")) {
    Pad(call);
  } else if (IsOp(call, "qnn.csi.proposal")) {
    Proposal(call);
  } else if (IsOp(call, "qnn.csi.psroipooling")) {
    PSROIPool(call);
  } else if (IsOp(call, "qnn.csi.relu")) {
    Relu(call);
  } else if (IsOp(call, "qnn.csi.relu6")) {
    Relu6(call);
  } else if (IsOp(call, "qnn.csi.reshape")) {
    Reshape(call);
  } else if (IsOp(call, "qnn.csi.roipooling")) {
    ROIPool(call);
  } else if (IsOp(call, "qnn.csi.sigmoid")) {
    Sigmoid(call);
  } else if (IsOp(call, "qnn.csi.softmax")) {
    Softmax(call);
  } else if (IsOp(call, "qnn.csi.space_to_batch_nd")) {
    SpaceToBatchND(call);
  } else if (IsOp(call, "qnn.csi.split")) {
    Split(call);
  } else if (IsOp(call, "qnn.csi.squeeze")) {
    Squeeze(call);
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSlice(call);
  } else if (IsOp(call, "qnn.csi.transpose")) {
    Transpose(call);
  } else if (IsOp(call, "qnn.csi.unpooling")) {
    UnPool2d(call);
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    UpSampling(call);
  } else {
    std::cerr << "light unsupported op: " << AsText(call->op, false) << "\n";
    exit(-1);
  }
}
void CodegenLight::EmitHeader(void) {
  std::ostringstream t0;
  PrintOneLine(code_stream_, "#include <csi_pnna.h>");
  PrintNewLine(code_stream_);
}

void CodegenLight::EmitSessionSetup(void) {
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
    t1 << in_name << "->mtype = " << GetCSINNMemoryType(input_memory_type[i]) << ";\n";
    PrintIndents(t1);
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

void CodegenLight::GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo) {
  int valid_range = std::pow(2, bits - 1) - 1;
  float abs_max = std::max(std::abs(min_value), std::abs(max_value));
  float scale = valid_range / abs_max;
  int exponent;
  frexp(scale, &exponent);
  qinfo->scale = 1.0f / std::pow(2, exponent - 1);
  qinfo->zero_point = 0;
  qinfo->max = abs_max;
  qinfo->min = -abs_max;
}

void CodegenLight::EmitJitWrapper() {
  PrintNewLine(code_stream_);
  std::ostringstream t0;
  string in_dtype = cfg->dtype_input;
  string weight_dtype = cfg->dtype_weight;
  PrintNewLine(code_stream_);
  t0 << "int csinn_runtime_wrapper_(";
  t0 << "int64_t* arg_value, ";
  t0 << "int64_t* arg_type, ";
  t0 << "int64_t* arg_size, ";
  t0 << "int64_t* ret_vale, int64_t* ret_type_code) {";
  PrintOneLine(code_stream_, t0);

  EnterScope();
  PrintOneLine(code_stream_, "char *params_base = (char *)arg_value[2];");

  t0 << ext_func_id_ << "_(params_base);\n";

  PrintOneLine(code_stream_, t0);
  PrintOneLine(code_stream_, "return 0;");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

void CodegenLight::EmitNBGSetup(void) {
  std::ostringstream t0;
  std::vector<string> nbg_func_;
  int output_index = 0;
  for (uint i = 0; i < output_list_.size(); i++) {
    if (!output_list_[i].is_const) {
      string output_name = output_list_[i].name;
      auto iter = io_nodes.find(output_name);
      if (iter == io_nodes.end()) {
        CHECK(0);
      }
      QuantParams q_params = iter->second;
      std::ostringstream t0;
      t0 << "csi_set_tensor_entry(" << output_name << ", sess);";
      nbg_func_.push_back(t0.str());
      t0.str("");
      t0 << "csi_set_output(" << output_index++ << ", " << output_name << ", sess);";
      nbg_func_.push_back(t0.str());
    }
  }
  for (uint i = 0; i < ext_func_args_.size(); i++) {
    std::string new_name = CodegenCSINN::replace(ext_func_args_[i]->name_hint());
    auto iter = io_nodes.find(new_name);
    QuantParams q_params = iter->second;
    string in_name = q_params.name;
    std::ostringstream t0;
    t0 << in_name << "->mtype = " << GetCSINNMemoryType(input_memory_type[i]) << ";";
    nbg_func_.push_back(t0.str());
    t0.str("");
    t0 << "csi_set_tensor_entry(" << in_name << ", sess);";
    nbg_func_.push_back(t0.str());

    t0.str("");
    t0 << "csi_set_input(" << i << ", " << in_name << ", sess);";
    nbg_func_.push_back(t0.str());
  }
  // codegen for binary graph function
  PrintNewLine(code_stream_);
  t0 << "void *csinn_nbg(char *path) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  // function body
  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
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
  t0 << "csi_set_output_number(" << output_index << ", sess);";
  PrintOneLine(code_stream_, t0);

  PrintNewLine(code_stream_);
  std::map<string, QuantParams>::iterator iter;
  for (iter = io_nodes.begin(); iter != io_nodes.end(); iter++) {
    CreateGraphTensor(iter->second);
  }

  for (auto decl : nbg_func_) {
    PrintOneLine(code_stream_, decl);
  }

  t0 << "csi_load_binary_model(path, sess);";
  PrintOneLine(code_stream_, t0);
  PrintOneLine(code_stream_, "return sess;");

  ExitScope();
  PrintOneLine(code_stream_, "}");
}

string CodegenLight::EmitGraph(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  if (model_save == "save_only") {
    EmitJitWrapper();
  } else {
    EmitSessionRun();
    EmitNBGSetup();
  }
  DumpConstant();
  return code_stream_.str();
}

void CodegenLight::ModelBinarySave() {
  std::ostringstream t0;
  t0 << "sess->model_name = \"csi.mbs.bin\";";
  PrintOneLine(code_stream_, t0);
  t0 << "sess->base_quant_type = " << cfg->quantization_scheme << ";";
  PrintOneLine(code_stream_, t0);
  if (model_save == "save_only") {
    t0 << "sess->model_save = CSINN_SAVE_ONLY;";
    PrintOneLine(code_stream_, t0);
  }
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
