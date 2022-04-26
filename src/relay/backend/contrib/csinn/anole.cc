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
 * \file src/relay/backend/contrib/csinn/anole.cc
 * \brief Implementation of CSINN anole codegen APIs.
 */

#include "anole.h"

#include <map>
#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

void CodegenAnole::EmitHeader(void) {
  PrintOneLine(code_stream_, "#include <csi_ovx.h>");
  PrintNewLine(code_stream_);
}

void CodegenAnole::EmitSessionSetup(void) {
  std::ostringstream t0;
  t0 << "void *" << ext_func_id_ << "_(";
  if (multithread) {
    t0 << "char *params_base, int deviceIndex) {";
  } else {
    t0 << "char *params_base) {";
  }
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

  if (multithread) {
    PrintOneLine(code_stream_, "csi_ovx_set_graph_attribute(sess, deviceIndex);");
  }

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

  for (uint32_t i = 0; i < output_list_.size(); i++) {
    if (!output_list_[i].is_const) {
      string output_name = output_list_[i].name;
      t0 << "csi_set_output(" << i << ", " << output_name << ", sess);";
      PrintOneLine(code_stream_, t0);
    } else {
      t0 << output_list_[i].name << "->name = "
         << "\"" << output_list_[i].name << "\";";
      PrintOneLine(code_stream_, t0);
      t0 << output_list_[i].name << "->dtype = CSINN_DTYPE_FLOAT32;";
      PrintOneLine(code_stream_, t0);
      t0 << output_list_[i].name << "->is_const = 1;";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_set_tensor_entry(" << output_list_[i].name << ", sess);";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_set_output(" << i << ", " << output_list_[i].name << ", sess);";
      PrintOneLine(code_stream_, t0);
    }
  }

  PrintNewLine(code_stream_);
  PrintOneLine(code_stream_, "csi_session_setup(sess);");
  PrintOneLine(code_stream_, "return sess;");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

void CodegenAnole::EmitNBGSetup(void) {
  std::ostringstream t0;
  std::vector<string> nbg_func_;
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
      t0 << "csi_set_output(" << i << ", " << output_name << ", sess);";
      nbg_func_.push_back(t0.str());
    }
  }
  for (uint i = 0; i < ext_func_args_.size(); i++) {
    std::string new_name = CodegenCSINN::replace(ext_func_args_[i]->name_hint());
    auto iter = io_nodes.find(new_name);
    QuantParams q_params = iter->second;
    string in_name = q_params.name;
    std::ostringstream t0;
    t0 << "csi_set_tensor_entry(" << in_name << ", sess);";
    nbg_func_.push_back(t0.str());

    t0.str("");
    t0 << "csi_set_input(" << i << ", " << in_name << ", sess);";
    nbg_func_.push_back(t0.str());
  }
  // codegen for binary graph function
  PrintNewLine(code_stream_);
  if (multithread) {
    t0 << "void *csinn_nbg(char *path, int deviceIndex) {";
  } else {
    t0 << "void *csinn_nbg(char *path) {";
  }
  PrintOneLine(code_stream_, t0);
  EnterScope();

  // function body
  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
  t0 << "sess->base_api = " << target_name_ << ";";
  PrintOneLine(code_stream_, t0);
  t0 << "sess->base_dtype = " << base_dtype_ << ";";
  PrintOneLine(code_stream_, t0);
  PrintOneLine(code_stream_, "csi_session_init(sess);");

  if (multithread) {
    PrintOneLine(code_stream_, "csi_ovx_set_graph_attribute(sess, deviceIndex);");
  }

  t0 << "csi_set_input_number(" << ext_func_args_.size() << ", sess);";
  PrintOneLine(code_stream_, t0);
  t0 << "csi_set_output_number(" << output_list_.size() << ", sess);";
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

void CodegenAnole::VisitExpr_(const CallNode* call) {
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
  } else if (IsOp(call, "qnn.csi.avgpool2d")) {
    AvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.clip")) {
    Clip(call);
  } else if (IsOp(call, "qnn.csi.subtract")) {
    DisoOp(call, "sub");
  } else if (IsOp(call, "qnn.csi.div")) {
    DisoOp(call, "div");
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
  } else if (IsOp(call, "qnn.csi.equal")) {
    DisoOp(call, "equal");
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
  } else if (IsOp(call, "qnn.csi.minimum")) {
    DisoOp(call, "minimum");
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOp(call, "mul");
  } else if (IsOp(call, "qnn.csi.pad")) {
    Pad(call);
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PRelu(call);
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
    std::cerr << "Anole NPU unsupported op: " << AsText(call->op, false) << "\n";
    exit(-1);
  }
}

void CodegenAnole::DisoOp(const CallNode* call, string op_name) {
  std::ostringstream decl_stream;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnBinaryOpAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl_stream << "(";

  string lhs_name, rhs_name;
  /* Emit input0 tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  auto lhs_input = out_[0];
  lhs_name = CodegenCSINN::InputTensor(decl_stream, call, 0, q_params[0], cfg->dtype_weight);
  decl_stream << ", ";

  /* Emit input1 tensor */
  if (call->args[1].as<tvm::relay::CallNode>() || call->args[1].as<tvm::relay::VarNode>()) {
    VisitExpr(call->args[1]);
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    auto rhs_input = out_[0];
    rhs_name = CodegenCSINN::InputTensor(decl_stream, call, 1, q_params[1], cfg->dtype_weight);

  } else {
    // add constant arg
    VisitExpr(call->args[1]);
    CHECK(constant_.size() == 1) << "Every args expects a single out_";
    auto rhs = constant_[0];
    auto lhs_shape = GetShape(call->args[0]->checked_type());
    auto rhs_shape = GetShape(call->args[1]->checked_type());

    rhs_name = "rhs_" + to_string(buf_idx_);
    CreateConstantTensor(&rhs, rhs_name, rhs_shape, cfg->dtype_weight, q_params[1]);
    t0 << rhs_name << "->dtype = CSINN_DTYPE_UINT8";
    PushDeclLine(t0);
    t0 << "csi_set_tensor_entry(" << rhs_name << ", sess)";
    PushDeclLine(t0);
    decl_stream << rhs_name;
  }

  /* Emit output tensor */
  string output_name = OutputTensor(decl_stream, call, q_params[2], cfg->dtype_weight);

  string params_name = "params_" + to_string(buf_idx_);
  decl_stream << ", " << params_name << ")";

  malloc_params("diso_params", params_name);
  PushOutput(output_name, call);
  buf_idx_++;
  params_common_setup(decl_stream, call, op_name, params_name, attr->layer_name.c_str(),
                      "CSINN_LAYOUT_NCHW");
  end_stream(decl_stream, op_name);
}

void CodegenAnole::Flatten(const CallNode* call) {
  std::ostringstream decl_stream;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();

  string callback;
  if (CheckOutput(call) != -1) {
    callback = "csi_ovx_flatten_tail";
  } else {
    callback = "csi_ovx_flatten";
  }

  SisoOp<QnnCSIUnaryAttrs>(decl_stream, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl_stream << ", " << params_name << ")";

  malloc_params("flatten_params", params_name);
  t0 << params_name << "->base.layout = CSINN_LAYOUT_NCHW";
  PushDeclLine(t0);
  t0 << params_name << "->base.api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << "->base.bc = " << callback;
  PushDeclLine(t0);

  end_stream(decl_stream, "flatten");
}

void CodegenAnole::Squeeze(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISqueezeAttrs>();
  string callback;
  if (CheckOutput(call) != -1) {
    callback = "csi_ovx_squeeze_tail";
  } else {
    callback = "csi_ovx_squeeze";
  }

  SisoOp<QnnCSISqueezeAttrs>(decl, call, attr);

  string squeeze_axis_name = "squeeze_aixs_" + to_string(buf_idx_);
  int32_t squeeze_axis_dim_num = attr->axis.size();
  t0 << "int32_t " << squeeze_axis_name << "[" << squeeze_axis_dim_num << "] = {";
  for (int i = 0; i < squeeze_axis_dim_num; i++) {
    t0 << to_string(attr->axis[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("squeeze_params", params_name);
  t0 << params_name << "->base.layout = CSINN_LAYOUT_NCHW";
  PushDeclLine(t0);
  t0 << params_name << "->base.api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << "->base.bc = " << callback;
  PushDeclLine(t0);
  t0 << params_name << "->axis = " << squeeze_axis_name;
  PushDeclLine(t0);
  t0 << params_name << "->axis_num = " << squeeze_axis_dim_num;
  PushDeclLine(t0);
  end_stream(decl, "squeeze");
}

void CodegenAnole::Reshape(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIReshapeAttrs>();
  string callback;
  if (CheckOutput(call) != -1) {
    callback = "csi_ovx_reshape_tail";
  } else {
    callback = "csi_ovx_reshape";
  }

  SisoOp<QnnCSIReshapeAttrs>(decl, call, attr);

  auto out_shape = GetShape(call->checked_type());
  string new_shape_name = "shape_" + to_string(buf_idx_);
  int32_t new_shape_dim_num = out_shape.size();
  t0 << "int32_t " << new_shape_name << "[" << new_shape_dim_num << "] = {";
  for (int i = 0; i < new_shape_dim_num; i++) {
    t0 << to_string(out_shape[i]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("reshape_params", params_name);
  t0 << params_name << "->base.layout = CSINN_LAYOUT_NCHW";
  PushDeclLine(t0);
  t0 << params_name << "->base.api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << "->base.bc = " << callback;
  PushDeclLine(t0);
  t0 << params_name << "->shape = " << new_shape_name;
  PushDeclLine(t0);
  t0 << params_name << "->shape_num = " << new_shape_dim_num;
  PushDeclLine(t0);

  end_stream(decl, "reshape");
}

void CodegenAnole::ModelBinarySave() {
  std::ostringstream t0;
  t0 << "sess->model_name = \"network.nb\";";
  PrintOneLine(code_stream_, t0);
  if (model_save == "run_only") {
    t0 << "sess->model_save = CSINN_RUN_ONLY;";
  } else if (model_save == "save_only") {
    t0 << "sess->model_save = CSINN_SAVE_ONLY;";
  } else if (model_save == "save_and_run") {
    t0 << "sess->model_save = CSINN_SAVE_AND_RUN;";
  } else {
    std::cerr << "Unsupport for model_save type: " << model_save << "\n";
    exit(-1);
  }
  PrintOneLine(code_stream_, t0);
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
