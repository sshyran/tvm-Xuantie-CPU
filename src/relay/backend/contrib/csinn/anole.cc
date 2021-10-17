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

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "../../utils.h"
#include "codegen.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

/*!
 * \brief A common interface that is used by various external runtime to
 * generate the wrapper to invoke external kernels.
 *
 * \param ext_func_id The unique id of an external function. It will be used
 * during runtime to pick the correct external function.
 * \param args The arguments used by the external function.
 * \param buf_decl The declaration of temporary buffers that used to store the
 * intermeidate of each external kernel.
 * \param body The statements of the external function.
 * \param out The name and id pairs for output.
 *
 * \return The emitted code string.
 */
string CodegenAnole::JitImpl(const string& ext_func_id, const Array<Var>& args,
                             const std::vector<string>& buf_decl, const std::vector<string>& body,
                             const std::vector<string>& ovx, const std::vector<Output>& out) {
  std::ostringstream t0;
  PrintOneLine(code_stream_, "#include <csi_ovx.h>");
  PrintNewLine(code_stream_);
  t0 << "void *" << ext_func_id << "_(";

  t0 << "char *params_base) {";
  PrintOneLine(code_stream_, t0);

  EnterScope();

  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
  PrintOneLine(code_stream_, "sess->base_api = CSINN_ANOLE;");
  PrintOneLine(code_stream_, "sess->base_dtype = CSINN_DTYPE_UINT8;");
  PrintOneLine(code_stream_, "csi_session_init(sess);");

  t0 << "csi_set_input_number(" << args.size() << ", sess);";
  PrintOneLine(code_stream_, t0);
  t0 << "csi_set_output_number(" << output_list_.size() << ", sess);";
  PrintOneLine(code_stream_, t0);

  // Function body
  PrintNewLine(code_stream_);
  for (auto decl : buf_decl) {
    PrintOneLine(code_stream_, decl);
  }

  PrintNewLine(code_stream_);
  for (auto stmt : body) {
    PrintOneLine(code_stream_, stmt);
  }

  PrintNewLine(code_stream_);
  for (auto stmt : ovx) {
    PrintOneLine(code_stream_, stmt);
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].is_const) {
      CodegenCSINN::CreateConstantTensor(output_list_[i].name, output_list_[i].size,
                                         output_list_[i].shape);
      t0 << output_list_[i].name << "->dtype = CSINN_DTYPE_FLOAT32;";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_ovx_set_const_tensor(" << output_list_[i].name << ", sess);";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_set_output(" << i << ", " << output_list_[i].name << ", sess);";
      PrintOneLine(code_stream_, t0);
    }
  }

  PrintOneLine(code_stream_, "csi_session_setup(sess);");
  PrintOneLine(code_stream_, "return sess;");
  ExitScope();
  PrintOneLine(code_stream_, "}");

  t0 << "void csinn_run(";
  for (uint i = 0; i < args.size(); i++) {
    // const auto& dtype_str = GetDtypeString(args[i]);
    t0 << "void* "
       << "data" << to_string(i);
    if (i != args.size() - 1) {
      t0 << ", ";
    }
  }
  t0 << ", void *sess) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  PrintOneLine(code_stream_, "struct csi_tensor input_tensor;");
  for (uint i = 0; i < args.size(); i++) {
    t0 << "input_tensor.data = data" << to_string(i) << ";";
    PrintOneLine(code_stream_, t0);
    t0 << "csi_update_input(" << to_string(i) << ", "
       << "&input_tensor, sess);";
    PrintOneLine(code_stream_, t0);
  }
  PrintOneLine(code_stream_, "csi_session_run(sess);");
  ExitScope();
  PrintOneLine(code_stream_, "}");

  // codegen for binary graph function
  PrintNewLine(code_stream_);
  t0 << "void *csinn_nbg(const char *nbg_file_name) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  // function body
  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
  PrintOneLine(code_stream_, "sess->base_api = CSINN_ANOLE;");
  PrintOneLine(code_stream_, "csi_session_init(sess);");

  t0 << "csi_set_input_number(" << args.size() << ", sess);";
  PrintOneLine(code_stream_, t0);
  t0 << "csi_set_output_number(" << output_list_.size() << ", sess);";
  PrintOneLine(code_stream_, t0);

  PrintNewLine(code_stream_);
  for (auto decl : nbg_buf_decl_) {
    PrintOneLine(code_stream_, decl);
  }

  for (auto decl : nbg_func_) {
    PrintOneLine(code_stream_, decl);
  }

  t0 << "struct csi_tensor *inputs[" << nbg_input_tensor_name_.size() << "];";
  PrintOneLine(code_stream_, t0);
  t0 << "struct csi_tensor *outputs[" << nbg_output_tensor_name_.size() << "];";
  PrintOneLine(code_stream_, t0);
  for (uint32_t i = 0; i < nbg_input_tensor_name_.size(); i++) {
    t0 << "inputs[" << i << "] = " << nbg_input_tensor_name_[i] << ";";
    PrintOneLine(code_stream_, t0);
  }

  for (uint32_t i = 0; i < nbg_output_tensor_name_.size(); i++) {
    t0 << "outputs[" << i << "] = " << nbg_output_tensor_name_[i] << ";";
    PrintOneLine(code_stream_, t0);
  }

  // PrintOneLine(code_stream_, "csi_nbg_init(inputs, outputs, &params);");
  t0 << "csi_ovx_nbg(inputs, outputs, " << nbg_input_tensor_name_.size() << ", "
     << nbg_output_tensor_name_.size() << ", nbg_file_name);";
  PrintOneLine(code_stream_, t0);

  PrintNewLine(code_stream_);
  PrintOneLine(code_stream_, "csi_session_setup(sess);");
  PrintOneLine(code_stream_, "return sess;");

  ExitScope();
  PrintOneLine(code_stream_, "}");

  DumpConstant();

  return code_stream_.str();
}

string CodegenAnole::JIT(const std::vector<Output>& out) {
  return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, ovx_body_, out_);
}

string CodegenAnole::JIT(void) { return JIT(out_); }

void CodegenAnole::VisitExpr_(const CallNode* call) {
  call_list_.push_back(call);
  /* Get the arguments for various CSINN kernels. */
  /* QNN op */
  if (IsOp(call, "qnn.csi.nn_deinit")) {
    CSINNDeinit(call);
  } else if (IsOp(call, "qnn.csi.nn_init")) {
    CSINNInit(call);
  } else if (IsOp(call, "qnn.csi.softmax")) {
    SoftmaxU8(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    DenseU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2dU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_relu")) {
    Conv2dReluU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_relu6")) {
    Conv2dRelu6U8(call);
  } else if (IsOp(call, "qnn.csi.relu")) {
    ReluU8(call);
  } else if (IsOp(call, "qnn.csi.relu6")) {
    Relu6U8(call);
  } else if (IsOp(call, "qnn.csi.global_avgpool")) {
    GlobalAvgPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.global_maxpool")) {
    GlobalMaxPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.add")) {
    DisoOpU8(call, "add");
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOpU8(call, "add");
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOpU8(call, "mul");
  } else if (IsOp(call, "qnn.csi.maxpool")) {
    MaxPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.avg_pool")) {
    AvgPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    ConcatU8(call);
  } else if (IsOp(call, "qnn.csi.lrn")) {
    LRNU8(call);
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2dU8(call);
  } else if (IsOp(call, "qnn.csi.unpooling")) {
    UnPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PReluU8(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_locat")) {
    MaxPool2dLocatU8(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_with_argmax")) {
    Maxpool2dWithArgmaxU8(call);
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    LeakyReluU8(call);
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    UpSamplingU8(call);
  } else if (IsOp(call, "qnn.csi.flatten")) {
    FlattenU8(call);
  } else if (IsOp(call, "qnn.csi.transpose")) {
    TransposeU8(call);
  } else if (IsOp(call, "qnn.csi.squeeze")) {
    SqueezeU8(call);
  } else if (IsOp(call, "qnn.csi.reshape")) {
    ReshapeU8(call);
  } else if (IsOp(call, "qnn.csi.sigmoid")) {
    SigmoidU8(call);
  } else if (IsOp(call, "qnn.csi.psroipooling")) {
    PSROIPoolU8(call);
  } else if (IsOp(call, "qnn.csi.roipooling")) {
    ROIPoolU8(call);
  } else if (IsOp(call, "qnn.csi.proposal")) {
    ProposalU8(call);
  } else if (IsOp(call, "qnn.csi.mean")) {
    ReduceU8(call, "mean");
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSliceU8(call);
  } else if (IsOp(call, "qnn.csi.split")) {
    SplitU8(call);
  } else if (IsOp(call, "qnn.csi.exp")) {
    UnaryU8(call, "exp");
  } else {
    std::cerr << "Anole NPU unsupported op: " << AsText(call->op, false) << "\n";
    exit(-1);
  }
}

void CodegenAnole::CreateTensor(string name, string data, std::vector<int> shape) {
  std::ostringstream t0;
  t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
  PushDeclLine(t0);
  SetDim(name, shape);
}

void CodegenAnole::CreateTensor(string name, string data, std::vector<int> shape,
                                int32_t zero_point, double scale) {
  std::ostringstream t0;
  CreateTensor(name, data, shape);
  t0 << name << "->zero_point = " << to_string(zero_point);
  PushDeclLine(t0);
  t0 << name << "->scale = " << double_to_string(scale);
  PushDeclLine(t0);
}

void CodegenAnole::CreateConstantTensor(string name, size_t size, std::vector<int> shape,
                                        int32_t zero_point, double scale) {
  std::ostringstream t0;
  CodegenCSINN::CreateConstantTensor(name, size, shape);
  t0 << name << "->zero_point = " << to_string(zero_point);
  PushDeclLine(t0);
  t0 << name << "->scale = " << double_to_string(scale);
  PushDeclLine(t0);
}

string CodegenAnole::OutputTensor(std::ostringstream& decl, const CallNode* call,
                                  int32_t zero_point, double scale) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  auto out_shape = GetShape(call->checked_type());
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  string output_name = "output_" + to_string(buf_idx_);
  CreateTensor(output_name, "", out_shape, zero_point, scale);
  decl << ", " << output_name;
  return output_name;
}

string CodegenAnole::OutputTensor(std::ostringstream& decl, const CallNode* call,
                                  int32_t zero_point, double scale, double fix_scale) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  auto out_shape = GetShape(call->checked_type());
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  string output_name = "output_" + to_string(buf_idx_);
  CreateTensor(output_name, "", out_shape, zero_point, scale);
  decl << ", " << output_name;
  return output_name;
}

void CodegenAnole::CSINNInit(const CallNode* call) {
  std::ostringstream decl_stream;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<NNInitAttrs>();
  CHECK(attr);
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 1) << "csi_nn_init expects 1 args";

  decl_stream << "csi_ovx_set_tensor";

  // Make function call with input buffers when visiting arguments
  decl_stream << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  // const VarNode* input_var = call->args[0].as<tvm::relay::VarNode>();
  // PushInput(input_var->name_hint(), call);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  /* Emit output tensor */
  string output_name = OutputTensor(t0, call, output_zero_point, output_scale);

  decl_stream << output_name << ", sess);";

  ext_func_body.push_back(decl_stream.str());

  nbg_func_.push_back(decl_stream.str());
  nbg_input_tensor_name_.push_back(output_name);

  decl_stream.str("");
  decl_stream << "csi_set_input(" << to_string(ext_func_args_.size() - 1) << ", " << output_name
              << ", sess);";
  ext_func_body.push_back(decl_stream.str());

  auto out_shape = GetShape(call->checked_type());
  nbg_output_tensor(output_name, out_shape, output_zero_point, output_scale);
  nbg_func_.push_back(decl_stream.str());

  PushOutput(output_name, call);
  layer_index_--;

  buf_idx_++;
}

void CodegenAnole::CSINNDeinit(const CallNode* call) {
  const auto* attr = call->attrs.as<NNDeinitAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;

  CHECK(call->args.size() == 1) << "csi_nn_deinit expects 1 args";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  std::ostringstream ovx;

  auto input = out_[0];
  auto pre_call = call->args[0].as<tvm::relay::CallNode>();
  CHECK(pre_call);
  if (input.call != pre_call) {
    input = GetRealInput(pre_call);
    CHECK_NE(input.size, -1);
  }

  string output_name = input.name;

  auto out_shape = GetShape(pre_call->checked_type());
  nbg_output_tensor(output_name, out_shape, input_zero_point, input_scale);

  std::ostringstream t0;
  t0 << "csi_ovx_set_tensor(" << output_name << ", sess);";
  nbg_func_.push_back(t0.str());

  ovx << "csi_set_output(" << output_list_.size() << ", " << output_name << ", sess);\n";
  ovx_body_.push_back(ovx.str());

  nbg_func_.push_back(ovx.str());
  nbg_output_tensor_name_.push_back(output_name);

  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  const auto& dtype = GetDtypeString(type_node);
  Output output;
  output.dtype = dtype;
  output.is_const = false;
  output_list_.push_back(output);
}

void CodegenAnole::DisoOpU8(const CallNode* call, string op_name) {
  std::ostringstream decl_stream;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnBinaryOpAttrs>();
  CHECK(attr);
  int32_t lhs_zero_point = attr->lhs_zero_point;
  double lhs_scale = attr->lhs_scale;
  int32_t rhs_zero_point = attr->rhs_zero_point;
  double rhs_scale = attr->rhs_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl_stream << "(";

  string lhs_name, rhs_name;
  /* Emit input0 tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  auto lhs_input = out_[0];
  lhs_name = CodegenCSINN::InputTensor(decl_stream, call, 0, lhs_zero_point, lhs_scale);
  decl_stream << ", ";

  /* Emit input1 tensor */
  if (call->args[1].as<tvm::relay::CallNode>()) {
    VisitExpr(call->args[1]);
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    auto rhs_input = out_[0];
    rhs_name = CodegenCSINN::InputTensor(decl_stream, call, 1, rhs_zero_point, rhs_scale);

  } else {
    // add constant arg
    VisitExpr(call->args[1]);
    CHECK(constant_.size() == 1) << "Every args expects a single out_";
    auto rhs = constant_[0];
    auto lhs_shape = GetShape(call->args[0]->checked_type());
    auto rhs_shape = GetShape(call->args[1]->checked_type());
    rhs_name = "rhs_" + to_string(buf_idx_);
    CreateConstantTensor(rhs_name, rhs.size, rhs_shape, rhs_zero_point, rhs_scale);
    t0 << rhs_name << "->dtype = CSINN_DTYPE_UINT8";
    PushDeclLine(t0);
    t0 << "csi_ovx_set_const_tensor(" << rhs_name << ", sess)";
    PushDeclLine(t0);
    decl_stream << rhs_name;
  }

  /* Emit output tensor */
  string output_name = OutputTensor(decl_stream, call, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl_stream << ", &" << params_name << ")";

  t0 << "struct diso_params " << params_name;
  PushDeclLine(t0);

  PushOutput(output_name, call);
  buf_idx_++;
  params_common_setup(decl_stream, op_name, params_name);
  end_stream(decl_stream, op_name);
}

void CodegenAnole::FlattenU8(const CallNode* call) {
  std::ostringstream decl_stream;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();

  string callback;
  auto next_call = call_list_[call_list_.size() - 2];
  if (IsOp(next_call, "qnn.csi.nn_deinit")) {
    callback = "csi_ovx_flatten_tail";
  } else {
    callback = "csi_ovx_flatten";
  }

  SisoOpU8<QnnCSIUnaryAttrs>(decl_stream, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl_stream << ", &" << params_name << ")";

  t0 << "struct flatten_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".layout = CSINN_NCHW";
  PushDeclLine(t0);
  t0 << params_name << ".api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << ".bc = " << callback;
  PushDeclLine(t0);

  end_stream(decl_stream, "flatten");
}

void CodegenAnole::SqueezeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISqueezeAttrs>();
  string callback;
  auto next_call = call_list_[call_list_.size() - 2];
  if (IsOp(next_call, "qnn.csi.nn_deinit")) {
    callback = "csi_ovx_squeeze_tail";
  } else {
    callback = "csi_ovx_squeeze";
  }

  SisoOpU8<QnnCSISqueezeAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct squeeze_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".layout = CSINN_NCHW";
  PushDeclLine(t0);
  t0 << params_name << ".api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << ".bc = " << callback;
  PushDeclLine(t0);
  end_stream(decl, "squeeze");
}

void CodegenAnole::ReshapeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIReshapeAttrs>();
  string callback;
  auto next_call = call_list_[call_list_.size() - 2];
  if (IsOp(next_call, "qnn.csi.nn_deinit")) {
    callback = "csi_ovx_reshape_tail";
  } else {
    callback = "csi_ovx_reshape";
  }

  SisoOpU8<QnnCSIReshapeAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct reshape_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".layout = CSINN_NCHW";
  PushDeclLine(t0);
  t0 << params_name << ".api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << ".bc = " << callback;
  PushDeclLine(t0);

  end_stream(decl, "reshape");
}

void CodegenAnole::StridedSliceU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0, t1;
  const auto* attr = call->attrs.as<QnnCSIStridedSliceAttrs>();
  CHECK(attr);

  string callback;
  auto next_call = call_list_[call_list_.size() - 2];
  if (IsOp(next_call, "qnn.csi.nn_deinit")) {
    callback = "csi_ovx_slice_tail";
  } else {
    callback = "csi_ovx_slice";
  }

  // x86 reference
  auto begin = attr->begin;
  auto end = attr->end;
  auto strides = attr->strides;

  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 1) << "strided slic expects 1 args";
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string input_name;
  decl << "(";
  input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  t0 << "int32_t begin_" << buf_idx_ << "[" << begin.size() << "] = {";
  t1 << "int32_t end_" << buf_idx_ << "[" << end.size() << "] = {";
  for (uint i = 0; i < begin.size(); i++) {
    t0 << to_string(begin[i]) << ", ";
    t1 << to_string(end[i]) << ", ";
  }
  t0 << "}";
  t1 << "}";
  PushDeclLine(t0);
  PushDeclLine(t1);

  t0 << "int32_t strides_" << buf_idx_ << "[" << strides.size() << "] = {";
  for (uint i = 0; i < strides.size(); i++) {
    t0 << to_string(strides[i]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct slice_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".layout = CSINN_NCHW";
  PushDeclLine(t0);
  t0 << params_name << ".begin = begin_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".end = end_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".strides = strides_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".api = CSINN_ANOLE";
  PushDeclLine(t0);
  t0 << params_name << ".bc = " << callback;
  PushDeclLine(t0);

  PushOutput(output_name, call);
  end_stream(decl, "slice");
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
