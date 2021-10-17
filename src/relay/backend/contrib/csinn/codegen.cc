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
 * \file src/relay/backend/contrib/csinn/codegen.cc
 * \brief Implementation of CSINN codegen APIs.
 */

#include "codegen.h"

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

#include "../../../quantize/quantize.h"
#include "../../utils.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;
using namespace quantize;

void CodegenCSINN::VisitExpr_(const VarNode* node) {
  first_visit_expr = false;
  ext_func_args_.push_back(GetRef<Var>(node));
  out_.clear();
  Output output;
  output.name = node->name_hint();
  output.need_copy = false;
  auto output_shape = GetShape(node->checked_type());
  output.shape = output_shape;
  out_.push_back(output);
  out_list_.push_back(output);
}

void CodegenCSINN::VisitExpr_(const ConstantNode* node) {
  first_visit_expr = false;
  constant_.clear();
  CSIConstant constant;
  constant.name = "constant_" + to_string(const_idx_++);
  constant.size = node->data.Length();
  constant.data_buf = reinterpret_cast<uint8_t*>(malloc(constant.size));
  node->data.CopyToBytes(constant.data_buf, constant.size);

  constant_.push_back(constant);
  constant_list_.push_back(constant);
}

void CodegenCSINN::VisitExpr_(const TupleNode* op) {
  if (first_visit_expr) {
    // output expr
    first_visit_expr = false;
    for (auto field : op->fields) {
      if (auto const_node = field.as<tvm::relay::ConstantNode>()) {
        // const output node
        CHECK(const_node);
        VisitExpr(field);
        CHECK(constant_.size() == 1) << "Every args expects a single constant_";
        auto const_out = constant_[0];
        auto const_shape = GetShape(field->checked_type());
        string const_out_name = "const_output_" + to_string(buf_idx_);

        std::ostringstream buf;
        buf << "struct csi_tensor *" << const_out_name << " = csi_alloc_tensor(sess)";
        PushDeclLine(buf);
        buf << const_out_name << "->data = params_base + " << to_string(constant_offset);
        PushDeclLine(buf);
        constant_offset += const_out.size;
        SetDim(const_out_name, const_shape);

        buf_decl_.push_back(buf.str());

        auto type_node = field->checked_type().as<TensorTypeNode>();
        CHECK(type_node);
        const auto& dtype = GetDtypeString(type_node);

        int out_size = 1;
        for (size_t i = 0; i < const_shape.size(); i++) {
          out_size *= const_shape[i];
        }
        Output output;
        output.dtype = dtype;
        output.name = const_out_name;
        output.size = out_size;
        output.is_const = true;
        output.shape = const_shape;
        output_list_.push_back(output);
      } else {
        VisitExpr(field);
      }
    }
  } else {
    // other expr
    for (auto field : op->fields) {
      VisitExpr(field);
    }
  }
}

void CodegenCSINN::VisitExpr_(const CallNode* call) {
  call_list_.push_back(call);
  /* Get the arguments for various CSINN kernels. */
  /* QNN op */
  if (IsOp(call, "qnn.csi.nn_deinit")) {
    CSINNDeinit(call);
  } else if (IsOp(call, "qnn.csi.nn_init")) {
    CSINNInit(call);
  } else if (IsOp(call, "qnn.csi.softmax")) {
    SoftmaxU8(call);
  } else if (IsOp(call, "qnn.csi.reverse")) {
    ReverseU8(call);
  } else if (IsOp(call, "qnn.csi.log_softmax")) {
    LogSoftmaxU8(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    DenseU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2dU8(call);
  } else if (IsOp(call, "qnn.csi.conv3d")) {
    Conv3dU8(call);
  } else if (IsOp(call, "qnn.csi.dilation2d")) {
    Dilation2dU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_relu")) {
    Conv2dReluU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_relu6")) {
    Conv2dRelu6U8(call);
  } else if (IsOp(call, "qnn.csi.relu")) {
    ReluU8(call);
  } else if (IsOp(call, "qnn.csi.relu6")) {
    Relu6U8(call);
  } else if (IsOp(call, "qnn.csi.sin")) {
    UnaryU8(call, "sin");
  } else if (IsOp(call, "qnn.csi.asin")) {
    UnaryU8(call, "asin");
  } else if (IsOp(call, "qnn.csi.sinh")) {
    UnaryU8(call, "sinh");
  } else if (IsOp(call, "qnn.csi.asinh")) {
    UnaryU8(call, "asinh");
  } else if (IsOp(call, "qnn.csi.cos")) {
    UnaryU8(call, "cos");
  } else if (IsOp(call, "qnn.csi.acos")) {
    UnaryU8(call, "acos");
  } else if (IsOp(call, "qnn.csi.cosh")) {
    UnaryU8(call, "cosh");
  } else if (IsOp(call, "qnn.csi.acosh")) {
    UnaryU8(call, "acosh");
  } else if (IsOp(call, "qnn.csi.tan")) {
    UnaryU8(call, "tan");
  } else if (IsOp(call, "qnn.csi.tanh")) {
    UnaryU8(call, "tanh");
  } else if (IsOp(call, "qnn.csi.atan")) {
    UnaryU8(call, "atan");
  } else if (IsOp(call, "qnn.csi.atanh")) {
    UnaryU8(call, "atanh");
  } else if (IsOp(call, "qnn.csi.global_avgpool")) {
    GlobalAvgPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.global_maxpool")) {
    GlobalMaxPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.add")) {
    DisoOpU8(call, "add");
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOpU8(call, "add");
  } else if (IsOp(call, "qnn.csi.maximum")) {
    DisoOpU8(call, "maximun");
  } else if (IsOp(call, "qnn.csi.minimum")) {
    DisoOpU8(call, "minimum");
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOpU8(call, "mul");
  } else if (IsOp(call, "qnn.csi.div")) {
    DisoOpU8(call, "div");
  } else if (IsOp(call, "qnn.csi.power")) {
    DisoOpU8(call, "power");
  } else if (IsOp(call, "qnn.csi.subtract")) {
    DisoOpU8(call, "sub");
  } else if (IsOp(call, "qnn.csi.floor_mod")) {
    DisoOpU8(call, "floor_mod");
  } else if (IsOp(call, "qnn.csi.floor_div")) {
    DisoOpU8(call, "floor_div");
  } else if (IsOp(call, "qnn.csi.left_shift")) {
    DisoOpU8(call, "left_shift");
  } else if (IsOp(call, "qnn.csi.right_shift")) {
    DisoOpU8(call, "right_shift");
  } else if (IsOp(call, "qnn.csi.mod")) {
    DisoOpU8(call, "mod");
  } else if (IsOp(call, "qnn.csi.maxpool")) {
    MaxPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.avg_pool")) {
    AvgPool2dU8(call);
  } else if (IsOp(call, "qnn.csi.avg_pool3d")) {
    AvgPool3dU8(call);
  } else if (IsOp(call, "qnn.csi.max_pool3d")) {
    MaxPool3dU8(call);
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    ConcatU8(call);
  } else if (IsOp(call, "qnn.csi.lrn")) {
    LRNU8(call);
  } else if (IsOp(call, "qnn.csi.negative")) {
    UnaryU8(call, "negative");
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2dU8(call);
  } else if (IsOp(call, "qnn.csi.deconv3d")) {
    DeConv3dU8(call);
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
  } else if (IsOp(call, "qnn.csi.prod")) {
    ReduceU8(call, "prod");
  } else if (IsOp(call, "qnn.csi.max")) {
    ReduceU8(call, "max");
  } else if (IsOp(call, "qnn.csi.min")) {
    ReduceU8(call, "min");
  } else if (IsOp(call, "qnn.csi.sum")) {
    ReduceU8(call, "sum");
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSliceU8(call);
  } else if (IsOp(call, "qnn.csi.split")) {
    SplitU8(call);
  } else if (IsOp(call, "qnn.csi.exp")) {
    UnaryU8(call, "exp");
  } else if (IsOp(call, "qnn.csi.segment_max")) {
    SegmentU8(call, "max");
  } else if (IsOp(call, "qnn.csi.segment_min")) {
    SegmentU8(call, "min");
  } else if (IsOp(call, "qnn.csi.segment_mean")) {
    SegmentU8(call, "mean");
  } else if (IsOp(call, "qnn.csi.segment_prod")) {
    SegmentU8(call, "prob");
  } else if (IsOp(call, "qnn.csi.segment_sum")) {
    SegmentU8(call, "sum");
  } else if (IsOp(call, "qnn.csi.erf")) {
    UnaryU8(call, "erf");
  } else if (IsOp(call, "qnn.csi.abs")) {
    UnaryU8(call, "abs");
  } else if (IsOp(call, "qnn.csi.argmax")) {
    ReduceU8(call, "argmax");
  } else if (IsOp(call, "qnn.csi.argmin")) {
    ReduceU8(call, "argmin");
  } else if (IsOp(call, "qnn.csi.expand_dims")) {
    ExpandDimsU8(call);
  } else if (IsOp(call, "qnn.csi.broadcast_to")) {
    BroadCastToU8(call);
  } else if (IsOp(call, "qnn.csi.cast")) {
    UnaryU8(call, "cast");
  } else if (IsOp(call, "qnn.csi.ceil")) {
    UnaryU8(call, "ceil");
  } else if (IsOp(call, "qnn.csi.floor")) {
    UnaryU8(call, "floor");
  } else if (IsOp(call, "qnn.csi.round")) {
    UnaryU8(call, "round");
  } else if (IsOp(call, "qnn.csi.crop_resize")) {
    CropResizeU8(call);
  } else if (IsOp(call, "qnn.csi.depth_to_space")) {
    DepthToSpaceU8(call);
  } else if (IsOp(call, "qnn.csi.space_to_depth")) {
    SpaceToDepthU8(call);
  } else if (IsOp(call, "qnn.csi.clip")) {
    ClipU8(call);
  } else if (IsOp(call, "qnn.csi.pad")) {
    PadU8(call);
  } else if (IsOp(call, "qnn.csi.sqrt")) {
    UnaryU8(call, "sqrt");
  } else if (IsOp(call, "qnn.csi.full")) {
    FullU8(call);
  } else if (IsOp(call, "qnn.csi.bn")) {
    BNU8(call);
  } else if (IsOp(call, "qnn.csi.take")) {
    TakeU8(call);
  } else if (IsOp(call, "qnn.csi.log")) {
    UnaryU8(call, "log");
  } else if (IsOp(call, "qnn.csi.sign")) {
    UnaryU8(call, "sign");
  } else if (IsOp(call, "qnn.csi.tile")) {
    TileU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_relu_channel")) {
    Conv2dReluChannelU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_relu6_channel")) {
    Conv2dRelu6ChannelU8(call);
  } else if (IsOp(call, "qnn.csi.conv2d_channel")) {
    Conv2dChannelU8(call);
  } else {
    LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
  }
}

string replace(string a) {
  std::string new_name = a;
  int pos;
  pos = new_name.find("/");
  while (pos != -1) {
    new_name.replace(pos, string("/").length(), "_");
    pos = new_name.find("/");
  }
  return new_name;
}

void CodegenCSINN::GenerateBackendCFunc(const string& func_name, const Array<Var>& args,
                                        const Output& out) {
  PrintNewLine(code_stream_);
  std::ostringstream t0;
  t0 << "int " << func_name << "_runtime_wrapper_(";
  t0 << "int64_t* arg_value, ";
  t0 << "int64_t* arg_type, ";
  t0 << "int64_t* arg_size, ";
  t0 << "int64_t* ret_vale, int64_t* ret_type_code" << args.size() << ") {";
  PrintOneLine(code_stream_, t0);

  EnterScope();
  PrintOneLine(code_stream_, "char** inputs = (char**)arg_value[0];");
  PrintOneLine(code_stream_, "char** outputs = (char**)arg_value[1];");
  PrintOneLine(code_stream_, "char *params_base = (char *)arg_value[2];");

  for (uint i = 0; i < args.size(); i++) {
    const auto& dtype_str = GetDtypeString(args[i]);
    std::string new_name = replace(args[i]->name_hint());
    t0 << dtype_str << "* " << new_name << " = (" << dtype_str << "*)inputs[" << i << "];\n";
    PrintOneLine(code_stream_, t0);
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    t0 << output_list_[i].dtype << "* out_" << i << " = (" << output_list_[i].dtype << "*)outputs["
       << i << "];\n";
    PrintOneLine(code_stream_, t0);
  }

  t0 << func_name << "_(";
  for (const auto& arg : args) {
    std::string new_name = replace(arg->name_hint());
    t0 << new_name << ", ";
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    t0 << "out_" << i << ", ";
  }
  t0 << "params_base);";
  PrintOneLine(code_stream_, t0);
  PrintOneLine(code_stream_, "return 0;");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

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
string CodegenCSINN::JitImpl(const string& ext_func_id, const Array<Var>& args,
                             const std::vector<string>& buf_decl, const std::vector<string>& body,
                             const std::vector<Output>& out) {
  const QConfig& cfg = QConfig::CSIConfig();
  string base_dtype;
  if (cfg->dtype_input == DataType::UInt(8)) {
    base_dtype = "CSINN_DTYPE_UINT8";
  } else if (cfg->dtype_input == DataType::Int(8)) {
    base_dtype = "CSINN_DTYPE_INT8";
  }
  std::ostringstream t0;
  t0 << "void *" << ext_func_id << "_(";

  CHECK_EQ(out.size(), 1U) << "Internal error: only single output is support.";

  for (const auto& arg : args) {
    const auto& dtype_str = GetDtypeString(arg);
    std::string new_name = replace(arg->name_hint());
    t0 << dtype_str << "* " << new_name << ", ";
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    t0 << output_list_[i].dtype << "* out_" << i << ", ";
  }

  t0 << "char *params_base) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
  std::ostringstream sess_dtype;
  sess_dtype << "sess->base_dtype = " << base_dtype << ";";
  PrintOneLine(code_stream_, sess_dtype);
  PrintOneLine(code_stream_, "sess->base_layout = CSINN_NCHW;");

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
  for (uint i = 0; i < output_list_.size(); i++) {
    t0 << "memcpy("
       << "out_" << i << ", " << output_list_[i].name << "->data, 4 * " << output_list_[i].size
       << ");";
    PrintOneLine(code_stream_, t0);
  }

  // Free buffers
  PrintNewLine(code_stream_);
  for (int i = 0; i < alloc_idx_; i++) {
    t0 << "free(alloc_" << i << ");";
    PrintOneLine(code_stream_, t0);
  }
  ExitScope();
  PrintOneLine(code_stream_, "}");

  this->GenerateBackendCFunc(ext_func_id, args, out[0]);

  DumpConstant();

  return code_stream_.str();
}

string CodegenCSINN::JIT(const std::vector<Output>& out) {
  return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out);
}

string CodegenCSINN::JIT(void) { return JIT(out_); }

void CodegenCSINN::SetDim(string name, std::vector<int> shape) {
  std::ostringstream t0;
  for (size_t i = 0; i < shape.size(); i++) {
    t0 << name << "->dim[" << i << "] = " << shape[i];
    PushDeclLine(t0);
  }
  t0 << name << "->dim_count = " << shape.size();
  PushDeclLine(t0);
}

void CodegenCSINN::CreateConstantTensor(string name, size_t size, std::vector<int> shape) {
  std::ostringstream t0;
  t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
  PushDeclLine(t0);
  t0 << name << "->data = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += size;

  SetDim(name, shape);
}

void CodegenCSINN::CreateConstantTensor(string name, size_t size, std::vector<int> shape,
                                        int32_t zero_point, double scale) {
  std::ostringstream t0;
  CreateConstantTensor(name, size, shape);
  t0 << name << "->zero_point = " << to_string(zero_point);
  PushDeclLine(t0);
  t0 << name << "->scale = " << double_to_string(scale);
  PushDeclLine(t0);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(scale, &multiplier, &shift);
  t0 << name << "->multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << name << "->shift = " << to_string(shift);
  PushDeclLine(t0);
}

void CodegenCSINN::CreateTensor(string name, string data, std::vector<int> shape) {
  std::ostringstream t0;
  string new_name = replace(data);
  t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
  PushDeclLine(t0);
  t0 << name << "->data = " << new_name;
  PushDeclLine(t0);
  SetDim(name, shape);
}

void CodegenCSINN::CreateTensor(string name, string data, std::vector<int> shape,
                                int32_t zero_point, double scale) {
  std::ostringstream t0;
  CreateTensor(name, data, shape);
  t0 << name << "->zero_point = " << to_string(zero_point);
  PushDeclLine(t0);
  t0 << name << "->scale = " << double_to_string(scale);
  PushDeclLine(t0);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(scale, &multiplier, &shift);
  t0 << name << "->multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << name << "->shift = " << to_string(shift);
  PushDeclLine(t0);
}

void CodegenCSINN::CreateTensor(string name, string data, std::vector<int> shape,
                                int32_t zero_point, double scale, double fix_scale) {
  std::ostringstream t0;
  CreateTensor(name, data, shape);
  t0 << name << "->zero_point = " << to_string(zero_point);
  PushDeclLine(t0);
  t0 << name << "->scale = " << double_to_string(scale);
  PushDeclLine(t0);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(fix_scale, &multiplier, &shift);
  t0 << name << "->multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << name << "->shift = " << to_string(shift);
  PushDeclLine(t0);
}

Output CodegenCSINN::GetRealInput(const CallNode* call) {
  Output ret;
  ret.size = -1;
  for (auto out : out_list_) {
    if (out.call == call) {
      return out;
    }
  }
  return ret;
}

Output CodegenCSINN::GetRealInput(const VarNode* var) {
  Output ret;
  ret.size = -1;
  for (auto out : out_list_) {
    if (out.name == var->name_hint()) {
      return out;
    }
  }
  return ret;
}

void CodegenCSINN::PushInput(string name, const CallNode* call) {
  for (uint i = 0; i < out_list_.size(); i++) {
    if (out_list_[i].name == name) {
      out_list_[i].call = call;
    }
  }
}

void CodegenCSINN::InputTensor(std::ostringstream& decl, const CallNode* call, int input_index) {
  auto ishape = GetShape(call->args[input_index]->checked_type());
  auto input = out_[0];

  auto pre_var = call->args[input_index].as<tvm::relay::VarNode>();
  if (input.name != pre_var->name_hint()) {
    input = GetRealInput(pre_var);
    CHECK_NE(input.size, -1);
  }

  if (input.need_copy == true) {
    decl << input.name;
  } else {
    string input_name = "input" + to_string(input_index) + "_" + to_string(buf_idx_);
    CreateTensor(input_name, input.name, ishape);
    decl << input_name;
  }
}

string CodegenCSINN::InputTensor(std::ostringstream& decl, const CallNode* call, int input_index,
                                 int32_t zero_point, double scale) {
  auto ishape = GetShape(call->args[input_index]->checked_type());
  auto input = out_[0];
  auto shape = input.shape;

  auto pre_call = call->args[input_index].as<tvm::relay::CallNode>();
  if (pre_call) {
    if (input.call != pre_call) {
      input = GetRealInput(pre_call);
      CHECK_NE(input.size, -1);
    }

    if (input.need_copy == true) {
      decl << input.name;
      return input.name;
    } else {
      string input_name = "input" + to_string(input_index) + "_" + to_string(buf_idx_);
      CreateTensor(input_name, input.name, ishape, zero_point, scale);
      decl << input_name;
      return input_name;
    }
  } else {
    auto pre_call = call->args[input_index].as<tvm::relay::TupleGetItemNode>();
    CHECK(pre_call);
    auto pre_tuple = pre_call->tuple.as<tvm::relay::CallNode>();
    if (input.call != pre_tuple) {
      input = GetRealInput(pre_tuple);
      CHECK_NE(input.size, -1);
    }
    auto input_name = input.names[pre_call->index];
    if (input.need_copy == true) {
      decl << input_name;
    }
    return input_name;
  }
}

void CodegenCSINN::malloc_buf(string out, int out_size) {
  std::ostringstream t0;
  const QConfig& cfg = QConfig::CSIConfig();
  string base_dtype;
  if (cfg->dtype_input == DataType::UInt(8)) {
    base_dtype = "uint8_t";
  } else if (cfg->dtype_input == DataType::Int(8)) {
    base_dtype = "int8_t";
  }
  t0 << base_dtype << " *" << out << " = (" << base_dtype << " *)malloc(" << out_size << ")";
  PushDeclLine(t0);
}

void CodegenCSINN::setup_callback(std::ostringstream& decl, string op_name, string params_name) {
  std::ostringstream t0;
  t0 << "csi_" << op_name << "_init" << decl.str();
  PushDeclLine(t0);
}

void CodegenCSINN::params_common_setup(std::ostringstream& decl, string op_name,
                                       string params_name) {
  std::ostringstream t0;
  t0 << params_name << ".layout = CSINN_NCHW";
  PushDeclLine(t0);
  t0 << params_name << ".api = CSINN_REF";
  PushDeclLine(t0);
  setup_callback(decl, op_name, params_name);
}

string CodegenCSINN::OutputTensor(std::ostringstream& decl, const CallNode* call) {
  std::ostringstream t0;
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  string out = "alloc_" + to_string(alloc_idx_);
  auto out_shape = GetShape(call->checked_type());
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  t0 << "float *" << out << " = (float *)malloc(4 * " << out_size << ")";
  PushDeclLine(t0);
  alloc_idx_++;
  string output_name = "output_" + to_string(buf_idx_);
  CreateTensor(output_name, out, out_shape);
  decl << ", " << output_name;
  return output_name;
}

template <typename T>
string CodegenCSINN::OutputTensor(std::ostringstream& decl, const CallNode* call, const T* attr,
                                  int32_t zero_point, double scale) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  string out = "alloc_" + to_string(alloc_idx_);
  auto out_shape = GetShape(call->checked_type());
  int out_size = attr->out_dtype.bytes();
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  malloc_buf(out, out_size);
  alloc_idx_++;
  string output_name = "output_" + to_string(buf_idx_);
  CreateTensor(output_name, out, out_shape, zero_point, scale);
  decl << ", " << output_name;
  return output_name;
}

string CodegenCSINN::OutputTensor(std::ostringstream& decl, const CallNode* call,
                                  int32_t zero_point, double scale) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  string out = "alloc_" + to_string(alloc_idx_);
  auto out_shape = GetShape(call->checked_type());
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  malloc_buf(out, out_size);
  alloc_idx_++;
  string output_name = "output_" + to_string(buf_idx_);
  CreateTensor(output_name, out, out_shape, zero_point, scale);
  decl << ", " << output_name;
  return output_name;
}

string CodegenCSINN::OutputTensor(std::ostringstream& decl, const CallNode* call,
                                  int32_t zero_point, double scale, double fix_scale) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  string out = "alloc_" + to_string(alloc_idx_);
  auto out_shape = GetShape(call->checked_type());
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  malloc_buf(out, out_size);
  alloc_idx_++;
  string output_name = "output_" + to_string(buf_idx_);
  CreateTensor(output_name, out, out_shape, zero_point, scale, fix_scale);
  decl << ", " << output_name;
  return output_name;
}

void CodegenCSINN::DumpConstant() {
  std::ofstream params;
  params.open(params_path_, std::ios::out | std::ios::binary);
  for (auto constant : constant_list_) {
    params.write(reinterpret_cast<char*>(constant.data_buf), constant.size);
  }
  params.close();
}

void CodegenCSINN::InputMultiplier(string input, double scale) {
  std::ostringstream stream;
  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(scale, &multiplier, &shift);
  stream << input << "->multiplier = " << to_string(multiplier) << ";\n";
  stream << "  " << input << "->shift = " << to_string(shift) << ";";
  ext_func_body.push_back(stream.str());
}

void CodegenCSINN::PushOutput(string name, const CallNode* call, bool push_output) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  const auto& dtype = GetDtypeString(type_node);
  auto out_shape = GetShape(call->checked_type());
  int out_size = 1;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    out_size *= out_shape[i];
  }

  out_.clear();
  Output output;
  output.dtype = dtype;
  output.name = name;
  output.size = out_size;
  output.need_copy = true;
  output.call = call;
  output.shape = out_shape;
  output.index = layer_index_;
  output.is_const = false;
  layer_index_++;
  out_.push_back(output);
  out_list_.push_back(output);
  if (push_output) {
    output_list_.push_back(output);
  }
}

void CodegenCSINN::PushOutput(std::vector<string> names, const CallNode* call) {
  auto type_node = call->checked_type().as<TupleTypeNode>();
  CHECK(type_node);

  const auto& dtype = "uint8_t";

  out_.clear();
  Output output;
  output.dtype = dtype;
  output.need_copy = true;
  output.call = call;
  output.index = layer_index_;
  output.is_const = false;
  output.names = names;
  layer_index_++;
  out_.push_back(output);
  out_list_.push_back(output);
}

template <typename T>
void CodegenCSINN::SisoOpU8(std::ostringstream& decl, const CallNode* call, const T* attr) {
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 1) << "op expects 1 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  InputMultiplier(input_name, input_scale);

  PushOutput(output_name, call);
}

void CodegenCSINN::UnaryU8(const CallNode* call, string op_name) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOpU8<QnnCSIUnaryAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct siso_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, op_name, params_name);
  end_stream(decl, op_name);
}

void CodegenCSINN::CSINNInit(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<NNInitAttrs>();
  CHECK(attr);
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 1) << "csi_nn_init expects 1 args";

  decl << "csi_nn_init";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  // const VarNode* input_var = call->args[0].as<tvm::relay::VarNode>();
  // string name = replace(input_var->name_hint());
  // PushInput(name, call);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  InputTensor(decl, call, 0);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  decl << ");";

  ext_func_body.push_back(decl.str());

  PushOutput(output_name, call);

  buf_idx_++;
}

void CodegenCSINN::CSINNDeinit(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<NNDeinitAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;

  CHECK(call->args.size() == 1) << "csi_nn_deinit expects 1 args";

  decl << "csi_nn_deinit";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call);
  buf_idx_++;

  decl << ");";

  InputMultiplier(input_name, input_scale);
  ext_func_body.push_back(decl.str());

  PushOutput(output_name, call, true);
}

void CodegenCSINN::DisoOpU8(const CallNode* call, string op_name) {
  std::ostringstream decl;
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
  decl << "(";

  string lhs_name, rhs_name;
  /* Emit input0 tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  lhs_name = InputTensor(decl, call, 0, lhs_zero_point, lhs_scale);
  decl << ", ";

  /* Emit input1 tensor */
  if (call->args[1].as<tvm::relay::CallNode>()) {
    VisitExpr(call->args[1]);
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    rhs_name = InputTensor(decl, call, 1, rhs_zero_point, rhs_scale);
  } else {
    // add constant arg
    VisitExpr(call->args[1]);
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto rhs = constant_[0];
    auto rhs_shape = GetShape(call->args[1]->checked_type());

    rhs_name = "rhs_" + to_string(buf_idx_);
    CreateConstantTensor(rhs_name, rhs.size, rhs_shape, rhs_zero_point, rhs_scale);
    decl << rhs_name;
  }

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct diso_params " << params_name;
  PushDeclLine(t0);

  InputMultiplier(lhs_name, lhs_scale);
  InputMultiplier(rhs_name, rhs_scale);
  PushOutput(output_name, call);
  buf_idx_++;
  params_common_setup(decl, op_name, params_name);
  end_stream(decl, op_name);
}

template <typename T>
void CodegenCSINN::SetupPadding(string name, const T* attr) {
  Array<IndexExpr> pad = attr->padding;
  std::ostringstream t0;
  if (pad.size() == 4) {
    t0 << name << ".pad_top = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_left = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_down = " << to_string(pad[2].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_right = " << to_string(pad[3].as<IntImmNode>()->value);
    PushDeclLine(t0);
  } else if (pad.size() == 6) {
    t0 << name << ".pad_before = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_after = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_top = " << to_string(pad[2].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_left = " << to_string(pad[3].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_down = " << to_string(pad[4].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_right = " << to_string(pad[5].as<IntImmNode>()->value);
    PushDeclLine(t0);
  } else {
    t0 << name << ".pad_top = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_left = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_down = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << ".pad_right = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
  }
}

template <typename T>
void CodegenCSINN::SetupConv2dParams(string name, const T* attr) {
  std::ostringstream t0;
  t0 << "struct conv2d_params " << name;
  PushDeclLine(t0);
  t0 << name << ".group = " << to_string(attr->groups);
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << ".stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << ".dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".wscales = NULL";
  PushDeclLine(t0);
  t0 << name << ".wzps = NULL";
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupDilation2dParams(string name, const T* attr) {
  std::ostringstream t0;
  t0 << "struct dilation2d_params " << name;
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << ".stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilations;
  t0 << name << ".dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupConv3dParams(string name, const T* attr) {
  std::ostringstream t0;
  t0 << "struct conv3d_params " << name;
  PushDeclLine(t0);
  t0 << name << ".group = " << to_string(attr->groups);
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << ".stride_depth = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_height = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_width = " << to_string(strides[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << ".dilation_depth = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".dilation_height = " << to_string(dilation[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".dilation_width = " << to_string(dilation[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupPoolParams(string name, const T* attr) {
  std::ostringstream t0;
  t0 << "struct pool_params " << name;
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << ".stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> pool_size = attr->pool_size;
  t0 << name << ".filter_height = " << to_string(pool_size[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".filter_width = " << to_string(pool_size[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupPool3DParams(string name, const T* attr) {
  std::ostringstream t0;
  t0 << "struct pool3d_params " << name;
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << ".stride_depth = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_height = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".stride_width = " << to_string(strides[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> pool_size = attr->pool_size;
  t0 << name << ".filter_depth = " << to_string(pool_size[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".filter_height = " << to_string(pool_size[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << ".filter_width = " << to_string(pool_size[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

void CodegenCSINN::Conv2dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIConv2DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "Conv2d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupConv2dParams<QnnCSIConv2DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv2d", params_name);
  end_stream(decl, "conv2d");
}

void CodegenCSINN::Conv2dChannelU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIConv2DChannelAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 5) << "Conv2d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, 0, 1);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, 1);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ");";

  /* set channel quantization params */
  VisitExpr(call->args[3]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto wscales = constant_[0];
  string wscales_name = "wscales_" + to_string(buf_idx_);
  t0 << "float *" << wscales_name << " = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += wscales.size;

  VisitExpr(call->args[4]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto wzps = constant_[0];

  string wzps_name = "wzps_" + to_string(buf_idx_);
  t0 << "int32_t *" << wzps_name << " = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += wzps.size;
  SetupConv2dParams<QnnCSIConv2DChannelAttrs>(params_name, attr);

  t0 << params_name << ".wscales = " << wscales_name;
  PushDeclLine(t0);
  t0 << params_name << ".wzps = " << wzps_name;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv2d", params_name);
  end_stream(decl, "conv2d");
}

void CodegenCSINN::Conv3dU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  const auto* attr = call->attrs.as<QnnCSIConv3DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "Conv3d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupConv3dParams<QnnCSIConv3DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv3d", params_name);
  end_stream(decl, "conv3d");
}

void CodegenCSINN::Dilation2dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIDilation2DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 2) << "Dilation2D expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupDilation2dParams<QnnCSIDilation2DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "dilation2d", params_name);
  end_stream(decl, "dilation2d");
}

void CodegenCSINN::Conv2dReluU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIConv2DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "Conv2d_relu expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupConv2dParams<QnnCSIConv2DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv2d_relu", params_name);
  end_stream(decl, "conv2d_relu");
}

void CodegenCSINN::Conv2dReluChannelU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIConv2DChannelAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  // int32_t kernel_zero_point = attr->kernel_zero_point;
  // double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 5) << "Conv2d_relu_channel expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, 0, 1);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, 1);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  /* set channel quantization params */
  VisitExpr(call->args[3]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto wscales = constant_[0];
  string wscales_name = "wscales_" + to_string(buf_idx_);
  t0 << "float *" << wscales_name << " = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += wscales.size;

  VisitExpr(call->args[4]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto wzps = constant_[0];

  string wzps_name = "wzps_" + to_string(buf_idx_);
  t0 << "int32_t *" << wzps_name << " = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += wzps.size;

  SetupConv2dParams<QnnCSIConv2DChannelAttrs>(params_name, attr);

  t0 << params_name << ".wscales = " << wscales_name;
  PushDeclLine(t0);
  t0 << params_name << ".wzps = " << wzps_name;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv2d_relu", params_name);
  end_stream(decl, "conv2d_relu");
}

void CodegenCSINN::Conv2dRelu6ChannelU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIConv2DChannelAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 5) << "Conv2d_relu6_channel expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, 0, 1);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, 1);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  /* set channel quantization params */
  VisitExpr(call->args[3]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto wscales = constant_[0];
  string wscales_name = "wscales_" + to_string(buf_idx_);
  t0 << "float *" << wscales_name << " = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += wscales.size;

  VisitExpr(call->args[4]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto wzps = constant_[0];

  string wzps_name = "wzps_" + to_string(buf_idx_);
  t0 << "int32_t *" << wzps_name << " = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  constant_offset += wzps.size;

  SetupConv2dParams<QnnCSIConv2DChannelAttrs>(params_name, attr);

  t0 << params_name << ".wscales = " << wscales_name;
  PushDeclLine(t0);
  t0 << params_name << ".wzps = " << wzps_name;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv2d_relu6", params_name);
  end_stream(decl, "conv2d_relu6");
}

void CodegenCSINN::Conv2dRelu6U8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIConv2DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "Conv2d_relu expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupConv2dParams<QnnCSIConv2DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "conv2d_relu6", params_name);
  end_stream(decl, "conv2d_relu6");
}

void CodegenCSINN::DeConv2dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIDeConv2DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "DeConv2d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupConv2dParams<QnnCSIDeConv2DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "deconv2d", params_name);
  end_stream(decl, "deconv2d");
}

void CodegenCSINN::DeConv3dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIDeConv3DAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t kernel_zero_point = attr->kernel_zero_point;
  double kernel_scale = attr->kernel_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "DeConv3d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  SetupConv3dParams<QnnCSIDeConv3DAttrs>(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "deconv3d", params_name);
  end_stream(decl, "deconv3d");
}

void CodegenCSINN::DenseU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* dense_attr = call->attrs.as<QnnCSIDenseAttrs>();
  CHECK(dense_attr);
  int32_t input_zero_point = dense_attr->input_zero_point;
  double input_scale = dense_attr->input_scale;
  int32_t kernel_zero_point = dense_attr->kernel_zero_point;
  double kernel_scale = dense_attr->kernel_scale;
  int32_t output_zero_point = dense_attr->output_zero_point;
  double output_scale = dense_attr->output_scale;
  double fix_scale = input_scale * kernel_scale / output_scale;

  CHECK(call->args.size() == 3) << "Dense expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale, fix_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());
  auto ishape = GetShape(call->args[0]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(kernel_name, kernel.size, wshape, kernel_zero_point, kernel_scale);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(bias_name, bias.size, bshape, 0, input_scale * kernel_scale);
  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";
  t0 << "struct fc_params " << params_name;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);

  params_common_setup(decl, "fullyconnected", params_name);
  end_stream(decl, "fullyconnected");
}

void CodegenCSINN::SoftmaxU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOpU8<QnnCSIAxisAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct softmax_params " << params_name;
  PushDeclLine(t0);
  int actual_aixs = attr->axis;
  auto ishape = GetShape(call->args[0]->checked_type());
  if (attr->axis < 0) {
    actual_aixs += ishape.size();
  }
  t0 << params_name << ".axis = " << to_string(actual_aixs);
  PushDeclLine(t0);

  params_common_setup(decl, "softmax", params_name);
  end_stream(decl, "softmax");
}

void CodegenCSINN::ReverseU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOpU8<QnnCSIAxisAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct reverse_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << to_string(attr->axis);
  PushDeclLine(t0);

  params_common_setup(decl, "reverse", params_name);
  end_stream(decl, "reverse");
}

void CodegenCSINN::LogSoftmaxU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOpU8<QnnCSIAxisAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct log_softmax_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << to_string(attr->axis);
  PushDeclLine(t0);

  params_common_setup(decl, "log_softmax", params_name);
  end_stream(decl, "log_softmax");
}

void CodegenCSINN::ExpandDimsU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIExpandDimsAttrs>();
  SisoOpU8<QnnCSIExpandDimsAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct expand_dims_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << to_string(attr->axis);
  PushDeclLine(t0);
  t0 << params_name << ".num_newaxis = " << to_string(attr->num_newaxis);
  PushDeclLine(t0);

  params_common_setup(decl, "expand_dims", params_name);
  end_stream(decl, "expand_dims");
}

void CodegenCSINN::MaxPool2dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DAttrs>();
  SisoOpU8<QnnCSIMaxPool2DAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  SetupPoolParams(params_name, attr);

  params_common_setup(decl, "maxpool", params_name);
  end_stream(decl, "maxpool");
}

void CodegenCSINN::AvgPool2dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIAvgPool2DAttrs>();
  SisoOpU8<QnnCSIAvgPool2DAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  SetupPoolParams(params_name, attr);

  params_common_setup(decl, "averagepool", params_name);
  end_stream(decl, "averagepool");
}

void CodegenCSINN::AvgPool3dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIAvgPool3DAttrs>();
  SisoOpU8<QnnCSIAvgPool3DAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  SetupPool3DParams(params_name, attr);

  params_common_setup(decl, "averagepool3d", params_name);
  end_stream(decl, "averagepool3d");
}

void CodegenCSINN::MaxPool3dU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool3DAttrs>();
  SisoOpU8<QnnCSIMaxPool3DAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  SetupPool3DParams(params_name, attr);

  params_common_setup(decl, "max_pool3d", params_name);
  end_stream(decl, "max_pool3d");
}

void CodegenCSINN::GlobalAvgPool2dU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIGlobalAvgPoolAttrs>();
  SisoOpU8<QnnCSIGlobalAvgPoolAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct pool_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "global_averagepool", params_name);
  end_stream(decl, "global_averagepool");
}

void CodegenCSINN::GlobalMaxPool2dU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIGlobalMaxPoolAttrs>();
  SisoOpU8<QnnCSIGlobalMaxPoolAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct pool_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "global_maxpool", params_name);
  end_stream(decl, "global_maxpool");
}

void CodegenCSINN::Maxpool2dWithArgmaxU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DAttrs>();
  SisoOpU8<QnnCSIMaxPool2DAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  SetupPoolParams(params_name, attr);

  params_common_setup(decl, "maxpool", params_name);
  end_stream(decl, "maxpool");
}

void CodegenCSINN::MaxPool2dLocatU8(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DLocatAttrs>();
  CHECK(attr);
  int32_t input_zero_point = 0;
  double input_scale = 1;
  int32_t output_zero_point = 0;
  double output_scale = 1;

  CHECK(call->args.size() == 1) << "MaxPool2dLocat expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name =
      OutputTensor<QnnCSIMaxPool2DLocatAttrs>(decl, call, attr, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  SetupPoolParams(params_name, attr);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "maxpool2d_locat", params_name);
  end_stream(decl, "maxpool2d_locat");
}

void CodegenCSINN::UnPool2dU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnPoolingAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 2) << "Unpool2d expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);
  decl << ", ";

  /* Emit_ mask tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string mask_name = InputTensor(decl, call, 1, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct unpooling_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name
     << ".pad_out_height = " << to_string(attr->out_padding[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name
     << ".pad_out_width = " << to_string(attr->out_padding[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << ".scale_height = " << to_string(attr->scales[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << ".scale_width = " << to_string(attr->scales[1].as<IntImmNode>()->value);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "unpooling", params_name);
  end_stream(decl, "unpooling");
}

void CodegenCSINN::PSROIPoolU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIPSROIPoolingAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_points[0].as<IntImmNode>()->value;
  double input_scale = attr->input_scales[0].as<FloatImmNode>()->value;
  int32_t roi_zero_point = attr->input_zero_points[1].as<IntImmNode>()->value;
  double roi_scale = attr->input_scales[1].as<FloatImmNode>()->value;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 2) << "PSROIPooling expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);
  decl << ", ";

  /* Emit_ roi tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string roi_name = InputTensor(decl, call, 1, roi_zero_point, roi_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(attr->spatial_scale, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct psroipooling_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".output_dim = " << to_string(attr->output_dim);
  PushDeclLine(t0);
  t0 << params_name << ".group_size = " << to_string(attr->group_size);
  PushDeclLine(t0);
  t0 << params_name << ".spatial_scale = " << to_string(attr->spatial_scale);
  PushDeclLine(t0);
  t0 << params_name << ".spatial_scale_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".spatial_scale_shift = " << to_string(shift);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  InputMultiplier(roi_name, roi_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "psroipooling", params_name);
  end_stream(decl, "psroipooling");
}

void CodegenCSINN::ROIPoolU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIROIPoolingAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_points[0].as<IntImmNode>()->value;
  double input_scale = attr->input_scales[0].as<FloatImmNode>()->value;
  int32_t roi_zero_point = attr->input_zero_points[1].as<IntImmNode>()->value;
  double roi_scale = attr->input_scales[1].as<FloatImmNode>()->value;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 2) << "ROIPooling expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);
  decl << ", ";

  /* Emit_ roi tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string roi_name = InputTensor(decl, call, 1, roi_zero_point, roi_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(attr->spatial_scale, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  Array<IndexExpr> pooled_size = attr->pooled_size;

  t0 << "struct roi_pool_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".pooled_size_h = " << to_string(pooled_size[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << ".pooled_size_w = " << to_string(pooled_size[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << ".spatial_scale = " << to_string(attr->spatial_scale);
  PushDeclLine(t0);
  t0 << params_name << ".spatial_scale_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".spatial_scale_shift = " << to_string(shift);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  InputMultiplier(roi_name, roi_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "roipool", params_name);
  end_stream(decl, "roipool");
}

void CodegenCSINN::ProposalU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  std::ostringstream mstream, sstream, fstream;
  const auto* attr = call->attrs.as<QnnCSIProposalAttrs>();
  CHECK(attr);
  int32_t cls_zero_point = attr->input_zero_points[0].as<IntImmNode>()->value;
  double cls_scale = attr->input_scales[0].as<FloatImmNode>()->value;
  int32_t bbox_zero_point = attr->input_zero_points[1].as<IntImmNode>()->value;
  double bbox_scale = attr->input_scales[1].as<FloatImmNode>()->value;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 3) << "Proposal expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ cls tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string cls_name = InputTensor(decl, call, 0, cls_zero_point, cls_scale);
  decl << ", ";

  /* Emit_ bbox tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string bbox_name = InputTensor(decl, call, 1, bbox_zero_point, bbox_scale);

  /* Emit_ im_info tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto im_info = constant_[0];
  auto im_info_shape = GetShape(call->args[2]->checked_type());
  string im_info_name = "im_info_" + to_string(buf_idx_);
  CreateConstantTensor(im_info_name, im_info.size, im_info_shape);
  t0 << im_info_name << "->dtype = CSINN_DTYPE_FLOAT32";
  PushDeclLine(t0);

  decl << "," << im_info_name;

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  int32_t scales_num = attr->scales.size();
  int32_t ratios_num = attr->ratios.size();

  int32_t multiplier;
  int32_t shift;
  mstream << "int32_t scale_multipliers_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  sstream << "int32_t scale_shifts_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  fstream << "float scale_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  for (int i = 0; i < scales_num; i++) {
    float scale = attr->scales[i].as<FloatImmNode>()->value;
    GetMultiplierAndShift(scale, &multiplier, &shift);
    mstream << to_string(multiplier) << ", ";
    sstream << to_string(shift) << ", ";
    fstream << to_string(scale) << ", ";
  }
  mstream << "}";
  PushDeclLine(mstream);
  sstream << "}";
  PushDeclLine(sstream);
  fstream << "}";
  PushDeclLine(fstream);

  mstream << "int32_t ratio_multipliers_" << to_string(buf_idx_) << "[" << ratios_num << "] = {";
  sstream << "int32_t ratio_shifts_" << to_string(buf_idx_) << "[" << ratios_num << "] = {";
  fstream << "float ratios_" << to_string(buf_idx_) << "[" << scales_num << "] = {";
  for (int i = 0; i < ratios_num; i++) {
    float ratios = attr->ratios[i].as<FloatImmNode>()->value;
    GetMultiplierAndShift(ratios, &multiplier, &shift);
    mstream << to_string(multiplier) << ", ";
    sstream << to_string(shift) << ", ";
    fstream << to_string(ratios) << ", ";
  }
  mstream << "}";
  PushDeclLine(mstream);
  sstream << "}";
  PushDeclLine(sstream);
  fstream << "}";
  PushDeclLine(fstream);

  GetMultiplierAndShift(attr->threshold, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct proposal_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".scales = scale_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << ".scale_multipliers = scale_multipliers_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << ".scale_shifts = scale_shifts_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << ".scales_num = " << to_string(scales_num);
  PushDeclLine(t0);
  t0 << params_name << ".ratios = ratios_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << ".ratio_multipliers = ratio_multipliers_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << ".ratio_shifts = ratio_shifts_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << ".ratios_num = " << to_string(ratios_num);
  PushDeclLine(t0);
  t0 << params_name << ".feature_stride = " << to_string(attr->feature_stride);
  PushDeclLine(t0);
  t0 << params_name << ".threshold = " << to_string(attr->threshold);
  PushDeclLine(t0);
  t0 << params_name << ".threshold_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".threshold_shift = " << to_string(shift);
  PushDeclLine(t0);
  t0 << params_name << ".rpn_pre_nms_top_n = " << to_string(attr->rpn_pre_nms_top_n);
  PushDeclLine(t0);
  t0 << params_name << ".rpn_post_nms_top_n = " << to_string(attr->rpn_post_nms_top_n);
  PushDeclLine(t0);
  t0 << params_name << ".rpn_min_size = " << to_string(attr->rpn_min_size);
  PushDeclLine(t0);
  t0 << params_name << ".iou_loss = " << to_string(attr->iou_loss);
  PushDeclLine(t0);

  InputMultiplier(cls_name, cls_scale);
  InputMultiplier(bbox_name, bbox_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "proposal", params_name);
  end_stream(decl, "proposal");
}

void CodegenCSINN::UpSamplingU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUpSamplingAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 1) << "UpSampling expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct resize_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".resize_mode = ";
  if (attr->method == "bilinear") {
    t0 << "CSINN_RESIZE_BILINEAR";
  } else if (attr->method == "nearest_neighbor") {
    t0 << "CSINN_RESIZE_NEAREST_NEIGHBOR";
  } else if (attr->method == "nearest_bicubic") {
    t0 << "CSINN_RESIZE_NEAREST_BICUBIC";
  } else {
    CHECK(0);
  }
  PushDeclLine(t0);
  t0 << params_name << ".align_corners = " << to_string(attr->align_corners);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "resize", params_name);
  end_stream(decl, "resize");
}

void CodegenCSINN::ReluU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOpU8<QnnCSIUnaryAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct relu_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "relu", params_name);
  end_stream(decl, "relu");
}

void CodegenCSINN::FullU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIFullAttrs>();
  auto shape = attr->shape;
  SisoOpU8<QnnCSIFullAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "int32_t shape_" << buf_idx_ << "[" << shape.size() << "] = {";
  for (uint k = 0; k < shape.size(); k++) {
    t0 << Downcast<IndexExpr>(shape[k]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  t0 << "struct full_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".shape = shape_" << buf_idx_;
  PushDeclLine(t0);

  params_common_setup(decl, "full", params_name);
  end_stream(decl, "full");
}

void CodegenCSINN::BNU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIBatchNormAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  int32_t axis = attr->axis;
  double epsilon = attr->epsilon;
  bool center = attr->center;
  bool scale = attr->scale;

  CHECK(call->args.size() == 5) << "bn expects 5 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  /* Emit gamma tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto gamma = constant_[0];
  auto gshape = GetShape(call->args[1]->checked_type());
  string gamma_name = "gamma_" + to_string(buf_idx_);
  CreateConstantTensor(gamma_name, gamma.size, gshape, 0, 1);
  decl << ", &" << gamma_name;

  /* Emit beta tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto beta = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string beta_name = "beta_" + to_string(buf_idx_);
  CreateConstantTensor(beta_name, beta.size, bshape, 0, 1);
  decl << ", " << beta_name;

  /* Emit moving_mean tensor */
  VisitExpr(call->args[3]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto mean = constant_[0];
  auto mean_shape = GetShape(call->args[3]->checked_type());
  string mean_name = "mean_" + to_string(buf_idx_);
  CreateConstantTensor(mean_name, mean.size, mean_shape, 0, 1);
  decl << ", " << mean_name;

  /* Emit moving_var tensor */
  VisitExpr(call->args[4]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto var = constant_[0];
  auto var_shape = GetShape(call->args[4]->checked_type());
  string var_name = "var_" + to_string(buf_idx_);
  CreateConstantTensor(var_name, var.size, var_shape, 0, 1);
  decl << ", " << var_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  t0 << "struct bn_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << axis;
  PushDeclLine(t0);
  t0 << params_name << ".epsilon = " << epsilon;
  PushDeclLine(t0);
  t0 << params_name << ".center = " << to_string(center);
  PushDeclLine(t0);
  t0 << params_name << ".scale = " << to_string(scale);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "bn", params_name);
  end_stream(decl, "bn");
}

void CodegenCSINN::TakeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSITakeAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  int axis = static_cast<int>(attr->axis->value);
  String mode = attr->mode;

  CHECK(call->args.size() == 2) << "take expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  /* Emit indices tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto indices = constant_[0];
  auto indices_shape = GetShape(call->args[1]->checked_type());
  string indices_name = "indices_" + to_string(buf_idx_);
  CreateConstantTensor(indices_name, indices.size, indices_shape, 0, 1);
  decl << ", " << indices_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  t0 << "struct take_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << axis;
  PushDeclLine(t0);
  t0 << params_name << ".mode = " << mode;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "take", params_name);
  end_stream(decl, "take");
}

void CodegenCSINN::ClipU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIClipAttrs>();
  double min = attr->a_min;
  double max = attr->a_max;

  SisoOpU8<QnnCSIClipAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct clip_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".min_value = " << to_string(min);
  PushDeclLine(t0);
  t0 << params_name << ".max_value = " << to_string(max);
  PushDeclLine(t0);

  params_common_setup(decl, "clip", params_name);
  end_stream(decl, "clip");
}

void CodegenCSINN::PadU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIPadAttrs>();
  auto pad_width = attr->pad_width;
  double pad_value = attr->pad_value;
  string pad_mode = attr->pad_mode;
  SisoOpU8<QnnCSIPadAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "  int32_t pad_before_" << buf_idx_ << "[" << pad_width.size() << "]"
     << " = {";
  for (uint k = 0; k < pad_width.size(); k++) {
    t0 << Downcast<IndexExpr>(pad_width[k][0]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);
  t0 << "  int32_t pad_after_" << buf_idx_ << "[" << pad_width.size() << "]"
     << " = {";
  for (uint k = 0; k < pad_width.size(); k++) {
    t0 << Downcast<IndexExpr>(pad_width[k][1]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);
  t0 << "struct pad_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".pad_before = pad_before_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".pad_after = pad_after_" << buf_idx_;
  PushDeclLine(t0);
  if (pad_mode == "constant") {
    t0 << params_name << ".pad_mode = CSINN_PAD_CONSTANT";
  } else {
    t0 << params_name << ".pad_mode = CSINN_PAD_EDGE";
  }

  PushDeclLine(t0);
  t0 << params_name << ".pad_value = " << to_string(pad_value);
  PushDeclLine(t0);

  params_common_setup(decl, "pad", params_name);
  end_stream(decl, "pad");
}

void CodegenCSINN::TileU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSITileAttrs>();
  auto reps = attr->reps;
  SisoOpU8<QnnCSITileAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "  int32_t reps_" << buf_idx_ << "[" << reps.size() << "] = {";
  for (uint k = 0; k < reps.size(); k++) {
    t0 << Downcast<IndexExpr>(reps[k]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  t0 << "struct reps_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".reps = reps_" << buf_idx_;
  PushDeclLine(t0);

  params_common_setup(decl, "tile", params_name + ".tile");
  end_stream(decl, "tile");
}

void CodegenCSINN::DepthToSpaceU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISubPixelAttrs>();
  int block_size = attr->block_size;
  string mode = attr->mode;
  SisoOpU8<QnnCSISubPixelAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct depth_to_space_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".block_size = " << to_string(block_size);
  PushDeclLine(t0);
  t0 << params_name << ".mode = " << mode;
  PushDeclLine(t0);

  params_common_setup(decl, "depth_to_space", params_name);
  end_stream(decl, "depth_to_space");
}

void CodegenCSINN::SpaceToDepthU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISubPixelAttrs>();
  int block_size = attr->block_size;
  SisoOpU8<QnnCSISubPixelAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct space_to_depth_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".block_size = " << to_string(block_size);
  PushDeclLine(t0);

  params_common_setup(decl, "space_to_depth", params_name);
  end_stream(decl, "space_to_depth");
}

void CodegenCSINN::Relu6U8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOpU8<QnnCSIUnaryAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct relu_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "relu6", params_name);
  end_stream(decl, "relu6");
}

void CodegenCSINN::PReluU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIPReluAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t alpha_zero_point = attr->alpha_zero_point;
  double alpha_scale = attr->alpha_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 2) << "PRelu expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto alpha = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string alpha_name = "alpha_" + to_string(buf_idx_);
  CreateConstantTensor(alpha_name, alpha.size, wshape, alpha_zero_point, alpha_scale);
  decl << ", " << alpha_name;

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";
  t0 << "struct prelu_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << to_string(attr->axis);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "prelu", params_name);
  end_stream(decl, "prelu");
}

void CodegenCSINN::LeakyReluU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSILeakyReluAttrs>();
  CHECK(attr);
  double alpha = attr->alpha;
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 1) << "LeakyRelu expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  buf_idx_++;

  int32_t alpha_multiplier;
  int32_t alpha_shift;
  GetMultiplierAndShift(alpha, &alpha_multiplier, &alpha_shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "  struct relu_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".n = " << to_string(attr->alpha);
  PushDeclLine(t0);
  t0 << params_name << ".n_multiplier = " << to_string(alpha_multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".n_shift = " << to_string(alpha_shift);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "leaky_relu", params_name);
  end_stream(decl, "leaky_relu");
}

void CodegenCSINN::ConcatU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnConcatenateAttrs>();
  CHECK(attr);
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  /* Make function call with input buffers when visiting arguments */
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  auto tuple = call->args[0].as<tvm::relay::TupleNode>();
  CHECK(tuple);
  int32_t input_num = tuple->fields.size();

  string input_name = "input_" + to_string(buf_idx_);
  t0 << "struct csi_tensor *" << input_name << "[" << input_num << "]";
  PushDeclLine(t0);

  std::vector<int> index;
  for (int i = 0; i < input_num; i++) {
    auto sub_input_node = tuple->fields[i].as<tvm::relay::CallNode>();
    CHECK(sub_input_node);
    auto sub_input = GetRealInput(sub_input_node);
    CHECK(sub_input.need_copy == true);
    double sub_input_scale = attr->input_scales[i].as<FloatImmNode>()->value;
    InputMultiplier(sub_input.name, sub_input_scale);
    std::ostringstream mem_stream;
    mem_stream << input_name << "[" << i << "] = " << sub_input.name << ";";
    ext_func_body.push_back(mem_stream.str());
    index.push_back(sub_input.index);
  }
  decl << input_name;

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct concat_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".inputs_count = " << to_string(input_num);
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << to_string(attr->axis);
  PushDeclLine(t0);
  PushOutput(output_name, call);

  params_common_setup(decl, "concat", params_name);
  end_stream(decl, "concat");
}

void CodegenCSINN::LRNU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSILRNAttrs>();
  SisoOpU8<QnnCSILRNAttrs>(decl, call, attr);
  int32_t multiplier, shift;

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";
  t0 << "struct lrn_params " << params_name;
  PushDeclLine(t0);

  /* range */
  t0 << params_name << ".range = " << to_string(attr->size);
  PushDeclLine(t0);
  /* bias */
  GetMultiplierAndShift(attr->bias, &multiplier, &shift);
  t0 << params_name << ".bias = " << to_string(attr->bias);
  PushDeclLine(t0);
  t0 << params_name << ".bias_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".bias_shift = " << to_string(shift);
  PushDeclLine(t0);

  /* alpha */
  GetMultiplierAndShift(attr->alpha, &multiplier, &shift);
  t0 << params_name << ".alpha = " << to_string(attr->alpha);
  PushDeclLine(t0);
  t0 << params_name << ".alpha_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".alpha_shift = " << to_string(shift);
  PushDeclLine(t0);

  /* beta */
  GetMultiplierAndShift(attr->beta, &multiplier, &shift);
  t0 << params_name << ".beta = " << to_string(attr->beta);
  PushDeclLine(t0);
  t0 << params_name << ".beta_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << ".beta_shift = " << to_string(shift);
  PushDeclLine(t0);

  params_common_setup(decl, "lrn", params_name);
  end_stream(decl, "lrn");
}

void CodegenCSINN::FlattenU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  SisoOpU8<QnnCSIUnaryAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct flatten_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "flatten", params_name);
  end_stream(decl, "flatten");
}

void CodegenCSINN::SigmoidU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  SisoOpU8<QnnCSIUnaryAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct sigmoid_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "sigmoid", params_name);
  end_stream(decl, "sigmoid");
}

void CodegenCSINN::TransposeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSITransposeAttrs>();
  CHECK(attrs);

  int32_t input_zero_point = attrs->input_zero_point;
  double input_scale = attrs->input_scale;
  int32_t output_zero_point = attrs->output_zero_point;
  double output_scale = attrs->output_scale;

  CHECK(call->args.size() == 1) << "Transpose expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  string perm_name = "permute_" + to_string(buf_idx_);
  int32_t perm_size = attrs->axes.size();

  t0 << "int32_t " << perm_name << "[" << perm_size << "] = {";
  for (int i = 0; i < perm_size; i++) {
    t0 << to_string(attrs->axes[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";
  t0 << "struct transpose_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".permute = " << perm_name;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "transpose", params_name);
  end_stream(decl, "transpose");
}

void CodegenCSINN::ReshapeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIReshapeAttrs>();
  SisoOpU8<QnnCSIReshapeAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct reshape_params " << params_name;
  PushDeclLine(t0);
  params_common_setup(decl, "reshape", params_name);
  end_stream(decl, "reshape");
}

void CodegenCSINN::BroadCastToU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIBroadCastToAttrs>();
  SisoOpU8<QnnCSIBroadCastToAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct broadcast_params " << params_name << ";\n";
  PushDeclLine(t0);

  params_common_setup(decl, "broadcast", params_name);
  end_stream(decl, "broadcast");
}

void CodegenCSINN::SqueezeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISqueezeAttrs>();
  SisoOpU8<QnnCSISqueezeAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct squeeze_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "squeeze", params_name);
  end_stream(decl, "squeeze");
}

void CodegenCSINN::SegmentU8(const CallNode* call, string name) {
  const auto* attr = call->attrs.as<QnnCSISegmentAttrs>();
  std::ostringstream decl;
  std::ostringstream t0;
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit idx tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto idx = constant_[0];
  auto ishape = GetShape(call->args[1]->checked_type());
  string idx_name = "idx_" + to_string(buf_idx_);
  CreateConstantTensor(idx_name, idx.size, ishape);
  decl << ", " << idx_name;

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  InputMultiplier(input_name, input_scale);

  PushOutput(output_name, call);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct segment_params " << params_name;
  PushDeclLine(t0);

  params_common_setup(decl, "segment_" + name, params_name);
  end_stream(decl, "segment_" + name);
}

void CodegenCSINN::ReduceU8(const CallNode* call, string name) {
  std::ostringstream decl;
  std::ostringstream t0, t1;

  const auto* attr = call->attrs.as<QnnCSIReduceAttrs>();
  CHECK(attr);
  // x86 reference
  auto axis = attr->axis;
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;

  auto input_shape = GetShape(call->args[0]->checked_type());

  CHECK(call->args.size() == 1) << name << " expects 1 args";
  auto out_shape = GetShape(call->checked_type());
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name;
  decl << "(";
  input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  std::vector<int> out_extents;
  std::vector<int> out_strides;
  std::vector<int> inner_extents;
  std::vector<int> inner_strides;

  auto reduce_axes = __get_real_axis(input_shape.size(), axis);
  for (uint i = 0; i < input_shape.size(); i++) {
    int flag = 0;
    for (uint j = 0; j < reduce_axes.size(); j++) {
      uint tmp = reduce_axes[j];
      if (i == tmp) {
        flag = 1;
      }
    }
    if (flag) {
      inner_extents.push_back(input_shape[i]);
      int stride = __get_stride(i, input_shape);
      inner_strides.push_back(stride);
    } else {
      out_extents.push_back(input_shape[i]);
      int stride = __get_stride(i, input_shape);
      out_strides.push_back(stride);
    }
  }

  t0 << "int32_t out_strides_" << buf_idx_ << "[" << out_strides.size() << "] = {";
  t1 << "int32_t out_extents_" << buf_idx_ << "[" << out_extents.size() << "] = {";
  for (uint i = 0; i < out_strides.size(); i++) {
    t0 << to_string(out_strides[i]) << ", ";
    t1 << to_string(out_extents[i]) << ", ";
  }
  t0 << "}";
  t1 << "}";
  PushDeclLine(t0);
  PushDeclLine(t1);

  t0 << "int32_t inner_strides_" << buf_idx_ << "[" << inner_strides.size() << "] = {";
  t1 << "int32_t inner_extents_" << buf_idx_ << "[" << inner_extents.size() << "] = {";
  for (uint i = 0; i < inner_strides.size(); i++) {
    t0 << to_string(inner_strides[i]) << ", ";
    t1 << to_string(inner_extents[i]) << ", ";
  }
  t0 << "}";
  t1 << "}";
  PushDeclLine(t0);
  PushDeclLine(t1);

  t0 << "int32_t aixs_" << buf_idx_ << "[" << axis.size() << "] = {";
  for (uint i = 0; i < axis.size(); i++) {
    t0 << to_string(axis[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", &" << params_name << ")";

  t0 << "struct reduce_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".out_strides = out_strides_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".out_extents = out_extents_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".n = " << to_string(out_extents.size());
  PushDeclLine(t0);
  t0 << params_name << ".inner_strides = inner_strides_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".inner_extents = inner_extents_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".m = " << to_string(out_extents.size());
  PushDeclLine(t0);
  t0 << params_name << ".axis = aixs_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".axis_count = " << axis.size();
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, name, params_name);
  end_stream(decl, name);
}

void CodegenCSINN::CropResizeU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSICropResizeAttrs>();
  CHECK(attr);
  int32_t input_zero_point = attr->input_zero_point;
  double input_scale = attr->input_scale;
  int32_t output_zero_point = attr->output_zero_point;
  double output_scale = attr->output_scale;
  auto crop_size = attr->crop_size;
  auto method = attr->method;
  auto extrapolation_value = attr->extrapolation_value;
  CHECK(call->args.size() == 3) << "CropResize expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, output_zero_point, output_scale);

  /* Emit boxes tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto boxes = constant_[0];
  auto bshape = GetShape(call->args[1]->checked_type());

  string boxes_name = "boxes_" + to_string(buf_idx_);
  CreateConstantTensor(boxes_name, boxes.size, bshape, 0, 1);
  decl << ", " << boxes_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto box_indices = constant_[0];
  auto index_shape = GetShape(call->args[2]->checked_type());
  string index_name = "index_" + to_string(buf_idx_);
  CreateConstantTensor(index_name, box_indices.size, index_shape, 0, 1);
  decl << ", " << index_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", &" << params_name << ")";

  t0 << "  int32_t crop_size_" << buf_idx_ << "[" << crop_size.size() << "] = {";
  for (uint i = 0; i < crop_size.size(); i++) {
    t0 << Downcast<IndexExpr>(crop_size[i]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);
  t0 << "  struct crop_resize_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".method = " << method;
  PushDeclLine(t0);
  t0 << params_name << ".extrapolation_value = " << extrapolation_value;
  PushDeclLine(t0);
  t0 << params_name << ".crop_size = crop_size_" << buf_idx_;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "crop_resize", params_name);
  end_stream(decl, "crop_resize");
}

void CodegenCSINN::StridedSliceU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0, t1;
  const auto* attr = call->attrs.as<QnnCSIStridedSliceAttrs>();
  CHECK(attr);
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
  t0 << params_name << ".begin = begin_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".end = end_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".strides = strides_" << buf_idx_;
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(output_name, call);
  params_common_setup(decl, "slice", params_name);
  end_stream(decl, "slice");
}

void CodegenCSINN::SplitU8(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;

  const auto* attr = call->attrs.as<QnnCSISplitAttrs>();
  CHECK(attr);
  // x86 reference
  auto axis = attr->axis;

  double input_scale = attr->input_scale;
  int32_t input_zero_point = attr->input_zero_point;
  Array<tvm::PrimExpr> output_scales = attr->output_scales;
  Array<Integer> output_zero_points = attr->output_zero_points;

  CHECK(call->args.size() == 1) << "strided slic expects 1 args";
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string out_name = "output_" + to_string(buf_idx_);
  t0 << "struct csi_tensor *" << out_name << "[" << output_zero_points.size() << "]";
  PushDeclLine(t0);
  decl << "(";
  string input_name = InputTensor(decl, call, 0, input_zero_point, input_scale);

  decl << ", " << out_name << ", ";
  string params_name = "params_" + to_string(buf_idx_);
  decl << "&" << params_name << ")";

  std::vector<string> out_names;
  for (uint i = 0; i < output_zero_points.size(); i++) {
    double output_scale = output_scales[i].as<FloatImmNode>()->value;
    int32_t output_zero_point = output_zero_points[i].as<IntImmNode>()->value;
    auto type_node = call->checked_type().as<TupleTypeNode>();
    CHECK(type_node);
    uint32_t out_cnt = type_node->fields.size();
    CHECK(i < out_cnt);
    string out = "alloc_" + to_string(alloc_idx_);
    auto out_shape = GetShape(type_node->fields[i]);
    int out_size = 1;
    for (size_t j = 0; j < out_shape.size(); ++j) {
      out_size *= out_shape[j];
    }
    malloc_buf(out, out_size);
    alloc_idx_++;
    string output_name = "output_" + to_string(buf_idx_) + "_" + to_string(i);
    CreateTensor(output_name, out, out_shape, output_zero_point, output_scale);

    t0 << out_name << "[" << to_string(i) << "] = " << output_name;
    PushDeclLine(t0);

    out_names.push_back(output_name);
  }

  Array<Integer> indices_or_sections;
  if (const IntImmNode* sections = attr->indices_or_sections.as<IntImmNode>()) {
    t0 << "int32_t axis_len" << buf_idx_ << " = ";
    t0 << input_name << "->dim[" << axis << "]";
    PushDeclLine(t0);

    t0 << "int32_t index_" << buf_idx_ << " = ";
    t0 << "axis_len" << buf_idx_ << " / " << sections->value;
    PushDeclLine(t0);

    t0 << "int32_t indices_or_sections_" << buf_idx_ << "[";
    t0 << sections->value - 1 << "] = {";
    for (int x = 1; x < sections->value; x++) {
      t0 << "index_" << buf_idx_ << " * " << x << ",";
    }
    t0 << "}";
    PushDeclLine(t0);
  } else {
    auto indices_ = Downcast<Array<ObjectRef>>(attr->indices_or_sections);
    t0 << "int32_t indices_or_sections_" << buf_idx_ << "[" << indices_.size() << "] = {";
    for (uint k = 0; k < indices_.size(); k++) {
      auto idx = Downcast<IndexExpr>(indices_[k]);
      t0 << to_string(*tir::as_const_int(idx)) << ", ";
    }
    t0 << "}";
    PushDeclLine(t0);
  }

  t0 << "struct split_params " << params_name;
  PushDeclLine(t0);
  t0 << params_name << ".split_index = indices_or_sections_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << ".output_num = " << output_zero_points.size();
  PushDeclLine(t0);
  t0 << params_name << ".axis = " << to_string(axis);
  PushDeclLine(t0);

  InputMultiplier(input_name, input_scale);
  PushOutput(out_names, call);
  params_common_setup(decl, "split", params_name);
  end_stream(decl, "split");
}

/*!
 * \brief The CSINN codegen helper to generate wrapepr function calls of CSINN
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class CSINNModuleCodegen : public CSourceModuleCodegenBase {
 public:
  explicit CSINNModuleCodegen(const tvm::Target& target, const string& path) {
    this->target_ = target;
    this->params_path_ = path;
  }
  // Create a corresponding CSINN function for the given relay Function.
  void GenCSINNFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    // const auto* call = func->body.as<CallNode>();
    // CHECK(call) << "CSINN expects a single convolution or dense op";

    // Record the external symbol for runtime lookup.
    auto sid = GetExtSymbol(func);

    string layout;
    String device = target_->GetAttr<String>("device", "").value();
    if (device == "anole") {
      layout = "NCHW";
      CodegenAnole builder(sid, layout, device, params_path_);
      builder.VisitExpr(func->body);
      code_stream_ << builder.JIT();
    } else {
      layout = "NHWC";
      CodegenCSINN builder(sid, layout, device, params_path_);
      builder.VisitExpr(func->body);
      code_stream_ << builder.JIT();
    }
  }

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and csinn specific ones. To make
   * linking simpiler, the CSINN kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/csinn folder.
   *
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    // Create headers
    code_stream_ << "#include <csi_nn.h>\n";
    code_stream_ << "\n";

    if (ref->IsInstance<FunctionNode>()) {
      GenCSINNFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenCSINNFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    String sym = "";
    Array<String> variables = {};
    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc", sym, variables);
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
  tvm::Target target_;
  string params_path_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module CSINNCompiler(const ObjectRef& ref, const tvm::Target& target,
                              const string& params_path) {
  CSINNModuleCodegen csinn(target, params_path);
  return csinn.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.csinn").set_body_typed(CSINNCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
