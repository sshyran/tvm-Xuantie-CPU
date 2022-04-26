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

#include "csinn.h"

#include "anole.h"
#include "ch8601.h"
#include "dp1k.h"
#include "gref.h"
#include "light.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

void CodegenCSINN::VisitExpr_(const VarNode* node) {
  first_visit_expr = false;
  ext_func_args_.push_back(GetRef<Var>(node));
  out_.clear();
  Output output;
  output.name = node->name_hint();
  output.need_copy = false;
  auto output_shape = GetShape(node->checked_type());
  output.shape = output_shape;
  output.size = -1;
  out_.push_back(output);
  out_list_.push_back(output);
}

void CodegenCSINN::VisitExpr(const Expr& expr) {
  auto it = visit_counter_.find(expr.get());
  if (it != visit_counter_.end()) {
    if (auto const_node = expr.as<ConstantNode>()) {
      constant_.clear();
      CSIConstant* constant = new CSIConstant();
      constant->name = "constant_" + to_string(const_idx_++);
      constant->dtype = GetDtypeString(const_node->data.DataType());
      constant->size = const_node->data.Length();
      constant->data_buf = reinterpret_cast<float*>(malloc(constant->size));
      const_node->data.CopyToBytes(constant->data_buf, constant->size);
      constant_.push_back(*constant);
    }
    ++it->second;
  } else {
    using TParent = ExprFunctor<void(const Expr&)>;
    TParent::VisitExpr(expr);
    visit_counter_.insert({expr.get(), 1});
  }
}

void CodegenCSINN::VisitExpr_(const ConstantNode* node) {
  first_visit_expr = false;
  constant_.clear();
  CSIConstant* constant = new CSIConstant();
  constant->name = "constant_" + to_string(const_idx_++);
  constant->dtype = GetDtypeString(node->data.DataType());
  constant->size = node->data.Length();
  constant->data_buf = reinterpret_cast<float*>(malloc(constant->size));
  node->data.CopyToBytes(constant->data_buf, constant->size);
  constant_.push_back(*constant);
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
        string const_out_name = "const_output_" + to_string(buf_idx_);
        const_out.name = const_out_name;
        constant_list_.push_back(const_out);
        flag_list_.push_back(CONSTANT);
        auto const_shape = GetShape(field->checked_type());
        buf_idx_++;

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

        int out_size = 1;
        for (size_t i = 0; i < const_shape.size(); i++) {
          out_size *= const_shape[i];
        }
        Output output;
        output.dtype = "float";
        output.name = const_out_name;
        output.size = out_size;
        output.is_const = true;
        output.shape = const_shape;
        output.call = NULL;
        output_list_.push_back(output);
      } else {
        auto call_node = field.as<tvm::relay::CallNode>();
        CHECK(call_node);
        Output output;
        output.call = call_node;
        output_list_.push_back(output);
      }
    }
    for (auto field : op->fields) {
      auto const_node = field.as<tvm::relay::ConstantNode>();
      if (!const_node) {
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

bool CodegenCSINN::InOpList(const CallNode* call) {
  for (auto op : target_op_list) {
    if (IsOp(call, op)) {
      return true;
    }
  }
  return false;
}

void CodegenCSINN::VisitExpr_(const CallNode* call) {
  /* Get the arguments for various CSINN kernels. */
  /* QNN op */
  if (first_visit_expr) {
    first_visit_expr = false;
    Output output;
    output.call = call;
    output_list_.push_back(output);
  }
  if (IsOp(call, "qnn.csi.abs")) {
    Unary(call, "abs");
  } else if (IsOp(call, "qnn.csi.acos")) {
    Unary(call, "acos");
  } else if (IsOp(call, "qnn.csi.acosh")) {
    Unary(call, "acosh");
  } else if (IsOp(call, "qnn.csi.add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.argmax")) {
    Reduce(call, "argmax", "int32_t");
  } else if (IsOp(call, "qnn.csi.argmin")) {
    Reduce(call, "argmin", "int32_t");
  } else if (IsOp(call, "qnn.csi.asin")) {
    Unary(call, "asin");
  } else if (IsOp(call, "qnn.csi.asinh")) {
    Unary(call, "asinh");
  } else if (IsOp(call, "qnn.csi.atan")) {
    Unary(call, "atan");
  } else if (IsOp(call, "qnn.csi.atanh")) {
    Unary(call, "atanh");
  } else if (IsOp(call, "qnn.csi.avgpool2d")) {
    AvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.avgpool3d")) {
    AvgPool3d(call);
  } else if (IsOp(call, "qnn.csi.batch_to_space_nd")) {
    BatchToSpaceND(call);
  } else if (IsOp(call, "qnn.csi.bias_add")) {
    DisoOp(call, "add");
  } else if (IsOp(call, "qnn.csi.broadcast_to")) {
    BroadCastTo(call);
  } else if (IsOp(call, "qnn.csi.cast")) {
    Unary(call, "cast");
  } else if (IsOp(call, "qnn.csi.ceil")) {
    Unary(call, "ceil");
  } else if (IsOp(call, "qnn.csi.cache_matmul")) {
    CacheMatMul(call);
  } else if (IsOp(call, "qnn.csi.cache_conv1d")) {
    CacheConv1d(call);
  } else if (IsOp(call, "qnn.csi.clip")) {
    Clip(call);
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    Concat(call);
  } else if (IsOp(call, "qnn.csi.conv1d")) {
    Conv1d(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2d(call, "conv2d");
  } else if (IsOp(call, "qnn.csi.conv2d_relu")) {
    Conv2d(call, "conv2d_relu");
  } else if (IsOp(call, "qnn.csi.conv2d_relu6")) {
    Conv2d(call, "conv2d_relu6");
  } else if (IsOp(call, "qnn.csi.conv3d")) {
    Conv3d(call);
  } else if (IsOp(call, "qnn.csi.cos")) {
    Unary(call, "cos");
  } else if (IsOp(call, "qnn.csi.cosh")) {
    Unary(call, "cosh");
  } else if (IsOp(call, "qnn.csi.crop_resize")) {
    CropResize(call);
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2d(call);
  } else if (IsOp(call, "qnn.csi.deconv3d")) {
    DeConv3d(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    Dense(call);
  } else if (IsOp(call, "qnn.csi.depth_to_space")) {
    DepthToSpace(call);
  } else if (IsOp(call, "qnn.csi.dilation2d")) {
    Dilation2d(call);
  } else if (IsOp(call, "qnn.csi.div")) {
    DisoOp(call, "div");
  } else if (IsOp(call, "qnn.csi.equal")) {
    DisoOp(call, "equal", "bool");
  } else if (IsOp(call, "qnn.csi.erf")) {
    Unary(call, "erf");
  } else if (IsOp(call, "qnn.csi.exp")) {
    Unary(call, "exp");
  } else if (IsOp(call, "qnn.csi.expand_dims")) {
    ExpandDims(call);
  } else if (IsOp(call, "qnn.csi.flatten")) {
    Flatten(call);
  } else if (IsOp(call, "qnn.csi.floor")) {
    Unary(call, "floor");
  } else if (IsOp(call, "qnn.csi.floor_div")) {
    DisoOp(call, "floor_divide");
  } else if (IsOp(call, "qnn.csi.floor_mod")) {
    DisoOp(call, "floor_mod");
  } else if (IsOp(call, "qnn.csi.fsmn")) {
    Fsmn(call);
  } else if (IsOp(call, "qnn.csi.full")) {
    Full(call);
  } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
    GlobalAvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.global_maxpool2d")) {
    GlobalMaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    LeakyRelu(call);
  } else if (IsOp(call, "qnn.csi.left_shift")) {
    DisoOp(call, "left_shift");
  } else if (IsOp(call, "qnn.csi.log")) {
    Unary(call, "log");
  } else if (IsOp(call, "qnn.csi.layer_norm")) {
    LayerNorm(call);
  } else if (IsOp(call, "qnn.csi.log_softmax")) {
    LogSoftmax(call);
  } else if (IsOp(call, "qnn.csi.lrn")) {
    LRN(call);
  } else if (IsOp(call, "qnn.csi.max")) {
    Reduce(call, "max", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.maxpool3d")) {
    MaxPool3d(call);
  } else if (IsOp(call, "qnn.csi.maximum")) {
    DisoOp(call, "maximum");
  } else if (IsOp(call, "qnn.csi.matmul")) {
    MatMul(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d")) {
    MaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_locat")) {
    MaxPool2dLocat(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d_with_argmax")) {
    Maxpool2dWithArgmax(call);
  } else if (IsOp(call, "qnn.csi.mean")) {
    Reduce(call, "mean", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.min")) {
    Reduce(call, "min", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.minimum")) {
    DisoOp(call, "minimum");
  } else if (IsOp(call, "qnn.csi.mod")) {
    DisoOp(call, "mod");
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOp(call, "mul");
  } else if (IsOp(call, "qnn.csi.negative")) {
    Unary(call, "negative");
  } else if (IsOp(call, "qnn.csi.pad")) {
    Pad(call);
  } else if (IsOp(call, "qnn.csi.power")) {
    DisoOp(call, "power");
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PRelu(call);
  } else if (IsOp(call, "qnn.csi.prod")) {
    Reduce(call, "prod", cfg->dtype_weight);
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
  } else if (IsOp(call, "qnn.csi.reverse")) {
    Reverse(call);
  } else if (IsOp(call, "qnn.csi.right_shift")) {
    DisoOp(call, "right_shift");
  } else if (IsOp(call, "qnn.csi.roipooling")) {
    ROIPool(call);
  } else if (IsOp(call, "qnn.csi.round")) {
    Unary(call, "round");
  } else if (IsOp(call, "qnn.csi.scatter_nd")) {
    ScatterND(call);
  } else if (IsOp(call, "qnn.csi.segment_max")) {
    Segment(call, "max");
  } else if (IsOp(call, "qnn.csi.segment_mean")) {
    Segment(call, "mean");
  } else if (IsOp(call, "qnn.csi.segment_min")) {
    Segment(call, "min");
  } else if (IsOp(call, "qnn.csi.segment_prod")) {
    Segment(call, "prob");
  } else if (IsOp(call, "qnn.csi.segment_sum")) {
    Segment(call, "sum");
  } else if (IsOp(call, "qnn.csi.sigmoid")) {
    Sigmoid(call);
  } else if (IsOp(call, "qnn.csi.sign")) {
    Unary(call, "sign");
  } else if (IsOp(call, "qnn.csi.sin")) {
    Unary(call, "sin");
  } else if (IsOp(call, "qnn.csi.sinh")) {
    Unary(call, "sinh");
  } else if (IsOp(call, "qnn.csi.softmax")) {
    Softmax(call);
  } else if (IsOp(call, "qnn.csi.space_to_batch_nd")) {
    SpaceToBatchND(call);
  } else if (IsOp(call, "qnn.csi.space_to_depth")) {
    SpaceToDepth(call);
  } else if (IsOp(call, "qnn.csi.split")) {
    Split(call);
  } else if (IsOp(call, "qnn.csi.sqrt")) {
    Unary(call, "sqrt");
  } else if (IsOp(call, "qnn.csi.rsqrt")) {
    Unary(call, "rsqrt");
  } else if (IsOp(call, "qnn.csi.squeeze")) {
    Squeeze(call);
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSlice(call);
  } else if (IsOp(call, "qnn.csi.subtract")) {
    DisoOp(call, "sub");
  } else if (IsOp(call, "qnn.csi.sum")) {
    Reduce(call, "sum", cfg->dtype_weight);
  } else if (IsOp(call, "qnn.csi.take")) {
    Take(call);
  } else if (IsOp(call, "qnn.csi.tan")) {
    Unary(call, "tan");
  } else if (IsOp(call, "qnn.csi.tanh")) {
    Unary(call, "tanh");
  } else if (IsOp(call, "qnn.csi.tile")) {
    Tile(call);
  } else if (IsOp(call, "qnn.csi.transpose")) {
    Transpose(call);
  } else if (IsOp(call, "qnn.csi.unpooling")) {
    UnPool2d(call);
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    UpSampling(call);

  } else {
    LOG(FATAL) << "Unsupported op: " << AsText(call->op, false);
  }
}

string CodegenCSINN::replace(string a) {
  std::string new_name = a;
  int pos;
  int illegal_str_length = 3;
  char illegal_str[illegal_str_length] = {'.', '/', ':'};
  for (int i = 0; i < illegal_str_length; i++) {
    pos = new_name.find(illegal_str[i]);
    while (pos != -1) {
      new_name.replace(pos, 1, "_");
      pos = new_name.find(illegal_str[i]);
    }
  }

  return new_name;
}

QuantParams* CodegenCSINN::GetIntegralQuantParams(QuantParams* q_params, int32_t tensor_type,
                                                  QConfig_* quantize_cfg) {
  if (quantize_cfg == NULL) {
    quantize_cfg = cfg;
  }
  if (q_params->q_size == 1) {
    return q_params;
  }
  Qinfo* qinfo = q_params->qinfo;
  float min_value = qinfo[0].min;
  float max_value = qinfo[0].max;
  for (int i = 1; i < q_params->q_size; i++) {
    min_value = std::min(qinfo[i].min, min_value);
    max_value = std::max(qinfo[i].max, max_value);
  }
  QuantParams* ret = GetQuantParamsBase(min_value, max_value, tensor_type, quantize_cfg);
  ret->q_size = 1;
  return ret;
}

void CodegenCSINN::Axis0Cast(CSIConstant* data, CSIConstant* output, Qinfo* q_infos,
                             string target_dtype, int q_size, int inner_size) {
  float* input_data = reinterpret_cast<float*>(data->data_buf);
  if (target_dtype == "uint8_t") {
    uint8_t* out = reinterpret_cast<uint8_t*>(malloc(output->size));
    output->data_buf = out;
    output->dtype = "uint8_t";
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, 0);
        out_ = std::min(out_, 255);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int8_t") {
    int8_t* out = reinterpret_cast<int8_t*>(malloc(output->size));
    output->data_buf = out;
    output->dtype = "int8_t";
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -127);
        out_ = std::min(out_, 127);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int16_t") {
    int16_t* out = reinterpret_cast<int16_t*>(malloc(output->size));
    output->data_buf = out;
    output->dtype = "int16_t";
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -32768);
        out_ = std::min(out_, 32767);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int4_t") {
    /* round to byte, inner_size align byte */
    output->size = (inner_size + 1) / 2 * q_size;
    int64_t alloc_size = output->size;
    /* init as 0 for all memory, since may access half of byte */
    int8_t* out = reinterpret_cast<int8_t*>(calloc(alloc_size, 1));
    output->data_buf = out;
    output->dtype = "int4_t";
    for (int c = 0; c < q_size; c++) {
      for (int i = 0; i < inner_size; i++) {
        int index = c * inner_size + i;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -8);
        out_ = std::min(out_, 7);
        int out_index = c * ((inner_size + 1) / 2) + i / 2;
        /* int4 little endian */
        if (i % 2) {
          out[out_index] = (out[out_index] & 0xF) | (out_ << 4);
        } else {
          out[out_index] = (out[out_index] & 0xF0) | (out_ & 0xF);
        }
      }
    }
  } else {
    LOG(ERROR) << "get error dtype:" << target_dtype;
  }
}

void CodegenCSINN::Axis3Cast(CSIConstant* data, CSIConstant* output, Qinfo* q_infos,
                             string target_dtype, int q_size, int inner_size) {
  float* input_data = reinterpret_cast<float*>(data->data_buf);
  if (target_dtype == "uint8_t") {
    uint8_t* out = reinterpret_cast<uint8_t*>(malloc(output->size));
    output->data_buf = out;
    output->dtype = "uint8_t";
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, 0);
        out_ = std::min(out_, 255);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int8_t") {
    int8_t* out = reinterpret_cast<int8_t*>(malloc(output->size));
    output->data_buf = out;
    output->dtype = "int8_t";
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -127);
        out_ = std::min(out_, 127);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int16_t") {
    int16_t* out = reinterpret_cast<int16_t*>(malloc(output->size));
    output->data_buf = out;
    output->dtype = "int16_t";
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -32768);
        out_ = std::min(out_, 32768);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "int4_t") {
    /* round to byte, inner_size align byte */
    output->size = (inner_size + 1) / 2 * q_size;
    int64_t alloc_size = output->size;
    int8_t* out = reinterpret_cast<int8_t*>(malloc(alloc_size));
    output->data_buf = out;
    output->dtype = "int4_t";
    for (int i = 0; i < inner_size; i++) {
      for (int c = 0; c < q_size; c++) {
        int index = i * q_size + c;
        int32_t out_ = std::round(input_data[index] / q_infos[c].scale) + q_infos[c].zero_point;
        out_ = std::max(out_, -8);
        out_ = std::min(out_, 7);
        int out_index = index / 2;
        /* int4 little endian */
        if (index % 2) {
          out[out_index] = (out[out_index] & 0xF) | (out_ << 4);
        } else {
          out[out_index] = (out[out_index] & 0xF0) | (out_ & 0xF);
        }
      }
    }
  } else {
    LOG(ERROR) << "get error dtype:" << target_dtype;
  }
}

// for per-axis (per-channel) quantize kernel
CSIConstant* CodegenCSINN::CastParams(CSIConstant* data, string target_dtype,
                                      QuantParams quant_params, bool depthwise_kernel) {
  Qinfo* q_infos = quant_params.qinfo;
  int q_size = quant_params.q_size;

  CSIConstant* output = new CSIConstant();
  if (data->dtype == target_dtype || target_dtype == "float") {
    constant_list_.push_back(*data);
    flag_list_.push_back(CONSTANT);
    return data;
  } else {
    float* input_data = GetFloatData(data);
    output->name = data->name;
    int unit_size = 1;
    if (data->dtype == "uint8_t" || data->dtype == "int8_t") {
      unit_size = 1;
    } else if (data->dtype == "int16_t") {
      unit_size = 2;
    } else if (data->dtype == "float" || data->dtype == "int32_t") {
      unit_size = 4;
    } else if (data->dtype == "float64" || data->dtype == "int64_t") {
      unit_size = 8;
    }
    int size = data->size / unit_size;
    int inner_size = size / q_size;
    if (target_dtype == "int4_t") {
      // int4 only support NHWC
      if (depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "int8_t" || target_dtype == "uint8_t") {
      output->size = size;
      if ((layout_ == "NHWC") && depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "int16_t") {
      output->size = data->size / 2;
      if ((layout_ == "NHWC") && depthwise_kernel) {
        Axis3Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      } else {
        Axis0Cast(data, output, q_infos, target_dtype, q_size, inner_size);
      }
    } else if (target_dtype == "int32_t") {
      int32_t* out = reinterpret_cast<int32_t*>(malloc(data->size));
      output->data_buf = out;
      output->size = data->size;
      output->dtype = "int32_t";
      for (int i = 0; i < size; i++) {
        int32_t out_ = std::round(input_data[i] / q_infos->scale);
        out[i] = out_;
      }
    } else if (target_dtype == "float16") {
      int16_t* out;
      int alloc_size = 0;
      if (data->dtype == "int8_t") {
        alloc_size = data->size * 2;
      } else {
        /* for dtype == float */
        alloc_size = data->size / 2;
      }
      out = reinterpret_cast<int16_t*>(malloc(alloc_size));
      output->data_buf = out;
      output->size = data->size / 2;
      output->dtype = "float16";
      for (int i = 0; i < size; i++) {
        int16_t out_ = float32_to_float16(input_data[i]);
        out[i] = out_;
      }
    } else if (target_dtype == "bfloat16") {
      int16_t* out = reinterpret_cast<int16_t*>(malloc(data->size / 2));
      output->data_buf = out;
      output->size = data->size / 2;
      output->dtype = "bfloat16";
      for (int i = 0; i < size; i++) {
        int16_t out_ = float32_to_bfloat16(input_data[i]);
        out[i] = out_;
      }
    } else {
      LOG(ERROR) << "get error dtype:" << target_dtype;
    }
    free(input_data);
  }
  constant_list_.push_back(*output);
  flag_list_.push_back(CONSTANT);
  return output;
}

CSIConstant* CodegenCSINN::CastParams(CSIConstant* data, string target_dtype,
                                      QuantParams integral_input_quant,
                                      QuantParams kernel_quant_params) {
  if (data->dtype == target_dtype) {
    constant_list_.push_back(*data);
    flag_list_.push_back(CONSTANT);
    return data;
  }
  float* input_data = reinterpret_cast<float*>(data->data_buf);
  int q_size = kernel_quant_params.q_size;
  Qinfo* qinfos = kernel_quant_params.qinfo;
  float iscale = integral_input_quant.qinfo->scale;

  CSIConstant* output = new CSIConstant();
  output->name = data->name;
  output->size = data->size;
  if (target_dtype == "int32_t") {
    output->dtype = "int32_t";
    int32_t* out = reinterpret_cast<int32_t*>(malloc(data->size));
    output->data_buf = out;
    int size = data->size / 4;

    for (int i = 0; i < q_size; i++) {
      for (int j = 0; j < size / q_size; j++) {
        int index = i * (size / q_size) + j;
        float out_ = std::round(input_data[index] / (qinfos[i].scale * iscale));
        int int32_max = std::numeric_limits<int>::max();
        if (std::abs(out_) > int32_max) {
          // LOG(WARNING) << "bias will overflow! Force changed wscale";
          out[index] = int32_max;
        } else {
          out[index] = out_;
        }
      }
    }
  } else if (target_dtype == "float") {
    output->size = data->size;
    float* out = reinterpret_cast<float*>(malloc(output->size));
    memcpy(out, input_data, output->size);
    output->data_buf = out;
  } else if (target_dtype == "int16_t") {
    int16_t* out = reinterpret_cast<int16_t*>(malloc(data->size / 2));
    output->data_buf = out;
    output->size = data->size / 2;
    output->dtype = "int16_t";

    int size = data->size / 4;
    for (int i = 0; i < q_size; i++) {
      for (int j = 0; j < size / q_size; j++) {
        int index = i * (size / q_size) + j;
        int32_t out_ = std::round(input_data[index] / (qinfos[i].scale * iscale));
        out_ = std::max(out_, -32768);
        out_ = std::min(out_, 32767);
        out[index] = out_;
      }
    }
  } else if (target_dtype == "float16") {
    int16_t* out = reinterpret_cast<int16_t*>(malloc(data->size / 2));
    output->data_buf = out;
    output->size = data->size / 2;
    output->dtype = "float16";
    for (uint i = 0; i < data->size / 4; i++) {
      int16_t out_ = float32_to_float16(input_data[i]);
      out[i] = out_;
    }
  } else if (target_dtype == "bfloat16") {
    int16_t* out = reinterpret_cast<int16_t*>(malloc(data->size / 2));
    output->data_buf = out;
    output->size = data->size / 2;
    output->dtype = "bfloat16";
    for (uint i = 0; i < data->size / 4; i++) {
      int16_t out_ = float32_to_bfloat16(input_data[i]);
      out[i] = out_;
    }
  } else {
    LOG(ERROR) << "get error dtype:" << target_dtype;
  }
  constant_list_.push_back(*output);
  flag_list_.push_back(CONSTANT);
  return output;
}

void CodegenCSINN::EmitHeader(void) {
  std::ostringstream t0;
  PrintOneLine(code_stream_, "#include <csi_nn.h>");
  PrintNewLine(code_stream_);
}

void CodegenCSINN::EmitVersion(void) {
  std::ostringstream t0;
  t0 << "/* auto generate by HHB_VERSION " << HHB_VERSION << " */";
  PrintOneLine(code_stream_, t0);
  PrintNewLine(code_stream_);
}

void CodegenCSINN::EmitSessionSetup(void) {
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

  PrintNewLine(code_stream_);
  PrintOneLine(code_stream_, "csi_session_setup(sess);");
  PrintOneLine(code_stream_, "return sess;");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

void CodegenCSINN::EmitSessionRun(void) {
  std::ostringstream t0;
  t0 << "void csinn_run(";
  for (uint32_t i = 0; i < ext_func_args_.size(); i++) {
    t0 << "void* "
       << "data" << to_string(i);
    if (i != ext_func_args_.size() - 1) {
      t0 << ", ";
    }
  }
  t0 << ", void *sess) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  PrintOneLine(code_stream_, "struct csi_tensor input_tensor;");
  for (uint32_t i = 0; i < ext_func_args_.size(); i++) {
    t0 << "input_tensor.data = data" << to_string(i) << ";";
    PrintOneLine(code_stream_, t0);
    t0 << "csi_update_input(" << to_string(i) << ", "
       << "&input_tensor, sess);";
    PrintOneLine(code_stream_, t0);
  }
  PrintOneLine(code_stream_, "csi_session_run(sess);");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

void CodegenCSINN::EmitNBGSetup(void) {
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
  t0 << "sess->base_api = " << target_name_ << ";";
  PrintOneLine(code_stream_, t0);
  t0 << "sess->base_dtype = " << base_dtype_ << ";";
  PrintOneLine(code_stream_, t0);
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

string CodegenCSINN::EmitGraph(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  EmitSessionRun();
  EmitNBGSetup();
  DumpConstant();
  return code_stream_.str();
}

void CodegenCSINN::SetDim(string name, std::vector<int> shape) {
  std::ostringstream t0;
  if (shape.size() == 0) {
    t0 << name << "->dim[" << 0 << "] = 1";
    PushDeclLine(t0);
    t0 << name << "->dim_count = 1";
    PushDeclLine(t0);
    return;
  }
  for (size_t i = 0; i < shape.size(); i++) {
    t0 << name << "->dim[" << i << "] = " << shape[i];
    PushDeclLine(t0);
  }
  t0 << name << "->dim_count = " << shape.size();
  PushDeclLine(t0);
}

void CodegenCSINN::CreateGraphTensor(QuantParams q_params) {
  std::ostringstream t0;
  t0 << "struct csi_tensor *" << q_params.name << " = csi_alloc_tensor(sess);\n";
  for (uint32_t i = 0; i < q_params.shape.size(); i++) {
    t0 << "  " << q_params.name << "->dim[" << to_string(i)
       << "] = " << to_string(q_params.shape[i]) << ";\n";
  }
  t0 << "  " << q_params.name << "->dim_count = " << to_string(q_params.shape.size()) << ";\n";
  t0 << "  " << q_params.name << "->name = "
     << "\"" << q_params.name << "\""
     << ";\n";
  t0 << "  " << q_params.name << "->qinfo->zero_point = " << to_string(q_params.qinfo->zero_point)
     << ";\n";
  t0 << "  " << q_params.name << "->qinfo->scale = " << to_string(q_params.qinfo->scale) << ";\n";
  t0 << "  " << q_params.name << "->qinfo->min = " << to_string(q_params.qinfo->min) << ";\n";
  t0 << "  " << q_params.name << "->qinfo->max = " << to_string(q_params.qinfo->max) << ";\n";
  std::string io_dtype;
  if (cfg->quantization_scheme == "CSINN_QUANT_INT4_ASYM_W_SYM") {
    io_dtype = "CSINN_DTYPE_INT4";
  } else if (cfg->quantization_scheme == "CSINN_QUANT_UINT8_ASYM") {
    io_dtype = "CSINN_DTYPE_UINT8";
  } else if (cfg->quantization_scheme == "CSINN_QUANT_INT8_SYM") {
    io_dtype = "CSINN_DTYPE_INT8";
  } else if (cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM" ||
             cfg->quantization_scheme == "CSINN_QUANT_INT8_ORIGINAL" ||
             cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM") {
    io_dtype = "CSINN_DTYPE_INT8";
  } else if (cfg->quantization_scheme == "CSINN_QUANT_INT16_SYM") {
    io_dtype = "CSINN_DTYPE_INT16";
  } else if (cfg->quantization_scheme == "CSINN_QUANT_FLOAT16") {
    io_dtype = "CSINN_DTYPE_FLOAT16";
  } else if (cfg->quantization_scheme == "CSINN_QUANT_BFLOAT16") {
    io_dtype = "CSINN_DTYPE_BFLOAT16";
  } else if (cfg->quantization_scheme == "unset") {
    io_dtype = GetCSINNDtype(cfg->dtype_weight);
  } else {
    LOG(WARNING) << "Unsupport quantization scheme " << cfg->quantization_scheme;
  }
  t0 << "  " << q_params.name << "->dtype = " << io_dtype << ";\n";
  t0 << "  " << q_params.name << "->layout = " << GetCSINNActLayout(q_params.shape) << ";\n";
  PrintOneLine(code_stream_, t0);
}

void CodegenCSINN::CreateConstantTensorBase(string name, size_t size, std::vector<int> shape,
                                            string target_dtype, string layout) {
  std::ostringstream t0;
  t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
  PushDeclLine(t0);
  t0 << name << "->data = params_base + " << to_string(constant_offset);
  PushDeclLine(t0);
  t0 << name << "->name = "
     << "\"" << name << "\"";
  PushDeclLine(t0);
  t0 << name << "->is_const = 1";
  PushDeclLine(t0);
  t0 << name << "->dtype = " << GetCSINNDtype(target_dtype);
  PushDeclLine(t0);
  // if (shape.size() == 0) shape.push_back(1);
  t0 << name << "->layout = " << layout;
  PushDeclLine(t0);
  constant_offset += size;

  SetDim(name, shape);
}

void CodegenCSINN::CreateConstantTensor(CSIConstant* data, string name, std::vector<int> shape,
                                        string target_dtype, QuantParams quant_params,
                                        bool depthwise_kernel, bool is_bias) {
  float* input_data = reinterpret_cast<float*>(data->data_buf);
  if (is_bias && shape.size() == 0 && std::abs(input_data[0]) < 1e-5) {
    // no bias
    std::ostringstream t0;
    t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
    PushDeclLine(t0);
    t0 << name << "->data = NULL";
    PushDeclLine(t0);
    t0 << name << "->name = "
       << "\"" << name << "\"";
    PushDeclLine(t0);
    t0 << name << "->is_const = 1";
    PushDeclLine(t0);
    t0 << name << "->dim_count = 0";
    PushDeclLine(t0);
  } else {
    CSIConstant* data_cast = CastParams(data, target_dtype, quant_params, depthwise_kernel);
    std::ostringstream t0;
    string constant_layout = "";
    if (depthwise_kernel) {
      if (layout_ == "NCHW") {
        constant_layout = "CSINN_LAYOUT_O1HW";
      } else {
        constant_layout = "CSINN_LAYOUT_1HWO";
      }
    } else {
      constant_layout = GetCSINNWeightLayout(shape);
    }
    CreateConstantTensorBase(name, data_cast->size, shape, target_dtype, constant_layout);
    t0 << name << "->qinfo = (struct csi_quant_info *)(params_base + " << to_string(constant_offset)
       << ")";
    PushDeclLine(t0);
    t0 << name << "->quant_channel = " << double_to_string(quant_params.q_size);
    PushDeclLine(t0);
    constant_offset += quant_params.q_size * sizeof(Qinfo);
    quant_params.name = name;
    qinfo_list_.push_back(quant_params);
    flag_list_.push_back(QINFO);
  }
}

void CodegenCSINN::CreateConstantTensor(CSIConstant* data, string name, std::vector<int> shape,
                                        string target_dtype, QuantParams input_quant_params,
                                        QuantParams kernel_quant_params,
                                        QuantParams bias_quant_params) {
  float* input_data = reinterpret_cast<float*>(data->data_buf);
  if (shape.size() == 0 && std::abs(input_data[0]) < 1e-5) {
    // no bias
    std::ostringstream t0;
    t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
    PushDeclLine(t0);
    t0 << name << "->data = NULL";
    PushDeclLine(t0);
    t0 << name << "->name = "
       << "\"" << name << "\"";
    PushDeclLine(t0);
    t0 << name << "->is_const = 1";
    PushDeclLine(t0);
    t0 << name << "->dim_count = 0";
    PushDeclLine(t0);
  } else {
    QuantParams* integral_input_quant = GetIntegralQuantParams(&input_quant_params, ACTIVATE);
    CSIConstant* data_cast =
        CastParams(data, target_dtype, *integral_input_quant, kernel_quant_params);
    std::ostringstream t0;
    string layout = GetCSINNWeightLayout(shape);
    CreateConstantTensorBase(name, data_cast->size, shape, target_dtype, layout);
    for (int i = 0; i < bias_quant_params.q_size; i++) {
      bias_quant_params.qinfo[i].scale =
          integral_input_quant->qinfo->scale * kernel_quant_params.qinfo[i].scale;
      bias_quant_params.qinfo[i].zero_point = 0;
    }
    t0 << name << "->qinfo = (struct csi_quant_info *)(params_base + " << to_string(constant_offset)
       << ")";
    PushDeclLine(t0);
    t0 << name << "->quant_channel = " << double_to_string(bias_quant_params.q_size);
    PushDeclLine(t0);
    constant_offset += bias_quant_params.q_size * sizeof(Qinfo);
    bias_quant_params.name = name;
    qinfo_list_.push_back(bias_quant_params);
    flag_list_.push_back(QINFO);
  }
}

void CodegenCSINN::CreateTensor(string name, string data, std::vector<int> shape,
                                QuantParams quant_params, string dtype) {
  std::ostringstream t0;
  t0 << "struct csi_tensor *" << name << " = csi_alloc_tensor(sess)";
  PushDeclLine(t0);
  tensor_data[name] = data;
  t0 << name << "->name = "
     << "\"" << name << "\"";
  PushDeclLine(t0);
  t0 << name << "->dtype = " << GetCSINNDtype(dtype);
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

string CodegenCSINN::InputTensorCall(const CallNode* pre_call, int input_index,
                                     QuantParams quant_params, string dtype) {
  auto ishape = GetShape(pre_call->checked_type());
  auto input = out_[0];

  if (input.call != pre_call) {
    input = GetRealInput(pre_call);
    CHECK_NE(input.size, -1);
  }

  if (input.need_copy == true) {
    return input.name;
  } else {
    string input_name = "input" + to_string(input_index) + "_" + to_string(buf_idx_);
    CreateTensor(input_name, input.name, ishape, quant_params, dtype);
    quant_params.name = input_name;
    return input_name;
  }
}

string CodegenCSINN::InputTensorVar(const VarNode* pre_var, int input_index,
                                    QuantParams quant_params, string dtype) {
  auto ishape = GetShape(pre_var->checked_type());
  auto input = out_[0];
  string var_name = replace(pre_var->name_hint());

  if (input.name != var_name) {
    input = GetRealInput(pre_var);
    CHECK_EQ(input.size, -1);
  }

  if (io_nodes.end() != io_nodes.find(var_name)) {
    return io_nodes[var_name].name;
  } else {
    string input_name = "input" + to_string(input_index) + "_" + to_string(buf_idx_);
    quant_params.name = input_name;
    quant_params.offset = constant_offset;
    quant_params.shape = ishape;
    io_nodes[var_name] = quant_params;
    CreateTensor(input_name, "__" + input.name, ishape, quant_params, dtype);
    return input_name;
  }
}

string CodegenCSINN::InputTensorTupleItem(const TupleGetItemNode* pre_call,
                                          QuantParams quant_params, string dtype) {
  auto input = out_[0];
  CHECK(pre_call);
  auto pre_tuple = pre_call->tuple.as<tvm::relay::CallNode>();
  if (input.call != pre_tuple) {
    input = GetRealInput(pre_tuple);
    CHECK_NE(input.size, -1);
  }
  auto input_name = input.names[pre_call->index];
  quant_params.name = input_name;
  return input_name;
}

string CodegenCSINN::InputTensorName(const CallNode* call, int input_index,
                                     QuantParams quant_params, string dtype) {
  string input_name;
  if (auto pre_call = call->args[input_index].as<CallNode>()) {
    input_name = InputTensorCall(pre_call, input_index, quant_params, dtype);
  } else if (auto pre_var = call->args[input_index].as<VarNode>()) {
    input_name = InputTensorVar(pre_var, input_index, quant_params, dtype);
  } else {
    auto pre_call = call->args[input_index].as<TupleGetItemNode>();
    CHECK(pre_call);
    input_name = InputTensorTupleItem(pre_call, quant_params, dtype);
  }
  return input_name;
}

string CodegenCSINN::InputTensor(std::ostringstream& decl, const CallNode* call, int input_index,
                                 QuantParams quant_params, string dtype) {
  string input_name = InputTensorName(call, input_index, quant_params, dtype);
  decl << input_name;
  return input_name;
}

void CodegenCSINN::setup_callback(std::ostringstream& decl, string op_name, string params_name) {
  std::ostringstream t0;
  t0 << "csi_" << op_name << "_init" << decl.str();
  PushDeclLine(t0);
}

void CodegenCSINN::params_common_setup(std::ostringstream& decl, const CallNode* call,
                                       string op_name, string params_name, string layer_name,
                                       string layout = "CSINN_LAYOUT_NCHW") {
  std::ostringstream t0;
  if (!(layout_ == "NCHW" && layout == "CSINN_LAYOUT_NCHW")) {
    t0 << params_name << "->base.layout = CSINN_LAYOUT_" << layout_;
    PushDeclLine(t0);
  }
  t0 << params_name << "->base.name = "
     << "\"" << get_complete_layer_name(op_name, layer_name) << "\"";
  params_idx_++;
  PushDeclLine(t0);
  setup_callback(decl, op_name, params_name);
  CreateTensorSessData();
}

string CodegenCSINN::OutputTensor(std::ostringstream& decl, const CallNode* call,
                                  QuantParams quant_params, string dtype) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  auto out_shape = GetShape(call->checked_type());
  // if output is a single number, out_shape.size() here is zero
  if (out_shape.size() == 0) {
    out_shape.push_back(1);
  }
  string output_name = "output_" + to_string(buf_idx_);
  quant_params.name = output_name;
  quant_params.offset = constant_offset;
  quant_params.shape = out_shape;
  CreateTensor(output_name, "alloc", out_shape, quant_params, dtype);
  decl << ", " << output_name;
  int out_index = CheckOutput(call);
  if (out_index > -1) {
    io_nodes[output_name] = quant_params;
  }
  return output_name;
}

string CodegenCSINN::DataConvertTensor(std::vector<int> shape, QuantParams quant_params,
                                       string dtype) {
  // if output is a single number, out_shape.size() here is zero
  if (shape.size() == 0) {
    shape.push_back(1);
  }
  string output_name = "hybrid_output_" + to_string(buf_idx_);
  quant_params.name = output_name;
  quant_params.offset = constant_offset;
  quant_params.shape = shape;
  // string alloc_buffer_name = "hybrid_alloc" + to_string(buf_idx_);
  CreateTensor(output_name, "hybrid_alloc", shape, quant_params, dtype);
  // decl << ", " << output_name;
  return output_name;
}

void CodegenCSINN::DumpConstant() {
  std::ofstream params;
  params.open(params_path_, std::ios::out | std::ios::binary);
  int q_count = 0;
  int c_count = 0;
  CHECK(flag_list_.size() == qinfo_list_.size() + constant_list_.size());
  for (uint i = 0; i < flag_list_.size(); i++) {
    if (flag_list_[i] == QINFO) {
      params.write(reinterpret_cast<char*>(qinfo_list_[q_count].qinfo),
                   qinfo_list_[q_count].q_size * sizeof(Qinfo));
      q_count++;
    } else if (flag_list_[i] == CONSTANT) {
      params.write(reinterpret_cast<char*>(constant_list_[c_count].data_buf),
                   constant_list_[c_count].size);
      c_count++;
    }
  }
  params.close();
  if (debug_level_ == "INFO") {
    std::ofstream qinfo_file;
    std::ofstream const_file;
    qinfo_file.open("./hhb_debug_qinfo.h", std::ios::out);
    const_file.open("./hhb_debug_const.h", std::ios::out);
    int q_count = 0;
    int c_count = 0;
    CHECK(flag_list_.size() == qinfo_list_.size() + constant_list_.size());
    for (uint i = 0; i < flag_list_.size(); i++) {
      if (flag_list_[i] == QINFO) {
        qinfo_file << "qinfo " << qinfo_list_[q_count].name << " = {\n";
        qinfo_file << "// size: " << qinfo_list_[q_count].q_size << "\n";
        struct Qinfo* qinfo = qinfo_list_[q_count].qinfo;
        for (int i = 0; i < qinfo_list_[q_count].q_size; i++) {
          qinfo_file << "zero_point = " << qinfo[i].zero_point << ", ";
          qinfo_file << "scale = " << qinfo[i].scale << ", ";
          qinfo_file << "multiplier = " << qinfo[i].multiplier << ", ";
          qinfo_file << "shift = " << qinfo[i].shift << ", ";
          qinfo_file << "min = " << qinfo[i].min << ", ";
          qinfo_file << "max = " << qinfo[i].max << "\n";
        }
        qinfo_file << "}\n";
        q_count++;
      } else if (flag_list_[i] == CONSTANT) {
        const_file << "constant data " << constant_list_[c_count].name << " = {\n";

        if (constant_list_[c_count].dtype == "float") {
          float* data_buf = static_cast<float*>(constant_list_[c_count].data_buf);
          for (uint i = 0; i < constant_list_[c_count].size / 4; i++) {
            const_file << data_buf[i] << ", ";
            if (i % 16 == 15) {
              const_file << "\n";
            }
          }
        } else {
          uint8_t* data_buf = static_cast<uint8_t*>(constant_list_[c_count].data_buf);
          for (uint i = 0; i < constant_list_[c_count].size; i++) {
            const_file << "0x" << std::hex << static_cast<int>(data_buf[i]) << ", ";
            if (i % 16 == 15) {
              const_file << "\n";
            }
          }
        }
        const_file << "}\n";
        c_count++;
      }
    }
    qinfo_file.close();
    const_file.close();
  }
}

int CodegenCSINN::CheckOutput(const CallNode* call) {
  for (uint i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].call == call) {
      return i;
    }
  }
  return -1;
}

Output* CodegenCSINN::GetOutput(string name) {
  if (out_[0].name == name) {
    return &(out_[0]);
  }
  for (uint i = 0; i < out_list_.size(); i++) {
    if (out_list_[i].name == name) {
      return &(out_list_[i]);
    }
  }
  return NULL;
}

void CodegenCSINN::PushOutput(string name, const CallNode* call, string dtype) {
  auto type_node = call->checked_type().as<TensorTypeNode>();
  CHECK(type_node);
  if (dtype == "") {
    dtype = cfg->dtype_weight;
  }

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
  int out_index = CheckOutput(call);
  if (out_index > -1) {
    auto& out = output_list_[out_index];
    out = output;
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

QuantParams* CodegenCSINN::GetQuantParams(Array<Array<IndexExpr>> q_params,
                                          QConfig_* quantize_cfg) {
  if (quantize_cfg == NULL) {
    quantize_cfg = cfg;
  }
  int size = q_params.size();
  QuantParams* out_q_params = new QuantParams[size];
  for (int i = 0; i < size; i++) {
    auto q_param = q_params[i];
    int32_t tensor_type = q_param[0].as<IntImmNode>()->value;
    int32_t value_type = q_param[1].as<IntImmNode>()->value;
    int32_t q_type = q_param[2].as<IntImmNode>()->value;
    uint start_idx = 3;
    if (q_type == PER_TENSOR) {
      if (value_type == USE_MINMAX) {
        float min_value = q_param[start_idx].as<FloatImmNode>()->value;
        float max_value = q_param[start_idx + 1].as<FloatImmNode>()->value;
        out_q_params[i] = *GetQuantParamsBase(min_value, max_value, tensor_type, quantize_cfg);
        out_q_params[i].q_size = 1;
      } else if (value_type == USE_SCALE) {
        float scale = q_param[start_idx].as<FloatImmNode>()->value;
        float zp = q_param[start_idx + 1].as<IntImmNode>()->value;
        out_q_params[i] = *GetQuantParamsBase(scale, zp);
        out_q_params[i].q_size = 1;
      }
    } else if (q_type == PER_CHANNEL) {
      // flag + single channel == 3
      uint length = (q_param.size() - 3) / 2;
      out_q_params[i] = *new QuantParams();
      Qinfo* q_infos = new Qinfo[length];
      for (uint j = start_idx; j < q_param.size(); j = j + 2) {
        int index = (j - start_idx) / 2;
        if (value_type == USE_MINMAX) {
          float min_value = q_param[j].as<FloatImmNode>()->value;
          float max_value = q_param[j + 1].as<FloatImmNode>()->value;
          QuantParams* tmp = GetQuantParamsBase(min_value, max_value, tensor_type, quantize_cfg);
          q_infos[index] = *tmp->qinfo;
        } else if (value_type == USE_SCALE) {
          q_infos[index].scale = q_param[j].as<FloatImmNode>()->value;
          q_infos[index].zero_point = q_param[j + 1].as<IntImmNode>()->value;
          q_infos[index].min = 0;
          q_infos[index].max = 0;
          int multiplier, shift;
          GetMultiplierAndShift(q_infos[index].scale, &multiplier, &shift);
        }
      }
      out_q_params[i].qinfo = q_infos;
      out_q_params[i].q_size = length;
      if ((cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM" ||
           cfg->quantization_scheme == "CSINN_QUANT_INT4_ASYM_W_SYM") &&
          tensor_type == ACTIVATE) {
        out_q_params[i] = *GetIntegralQuantParams(&out_q_params[i], tensor_type);
      }
    }
  }
  return out_q_params;
}

QuantParams* CodegenCSINN::GetQuantParamsBase(float scale, int32_t zp) {
  QuantParams* q_params = new QuantParams();
  Qinfo* qinfo = new Qinfo();
  qinfo->scale = scale;
  qinfo->zero_point = zp;
  qinfo->min = 0;
  qinfo->max = 0;
  GetMultiplierAndShift(scale, &qinfo->multiplier, &qinfo->shift);
  q_params->qinfo = qinfo;
  return q_params;
}

QuantParams* CodegenCSINN::GetQuantParamsBase(float scale, int32_t zp, float min_value,
                                              float max_value) {
  QuantParams* q_params = new QuantParams();
  Qinfo* qinfo = new Qinfo();
  qinfo->scale = scale;
  qinfo->zero_point = zp;
  qinfo->min = min_value;
  qinfo->max = max_value;
  GetMultiplierAndShift(scale, &qinfo->multiplier, &qinfo->shift);
  q_params->qinfo = qinfo;
  return q_params;
}

void CodegenCSINN::GetAsymScale(float min_value, float max_value, int bits, Qinfo* qinfo) {
  int valid_range = std::pow(2, bits) - 1;
  max_value = std::max(max_value, 0.0f);
  min_value = std::min(min_value, 0.0f);
  if (cfg->dtype_input == "uint8_t") {
    qinfo->scale = (max_value - min_value) / valid_range;
    if (qinfo->scale == 0) {
      qinfo->scale = std::abs(max_value);
    }
    qinfo->zero_point = std::min(
        valid_range, static_cast<int>(std::max(0.0f, std::round(0 - min_value / qinfo->scale))));
  } else if (cfg->dtype_input == "int8_t") {
    qinfo->scale = (max_value - min_value) / valid_range;
    if (qinfo->scale == 0) {
      qinfo->scale = 1;
    }
    float low_bound = -std::pow(2, bits - 1);
    int high_bound = std::pow(2, bits - 1) - 1;
    qinfo->zero_point = std::min(
        high_bound,
        static_cast<int>(std::max(low_bound, std::round(-128 - min_value / qinfo->scale))));
  } else if (cfg->dtype_input == "int4_t") {
    qinfo->scale = (max_value - min_value) / valid_range;
    if (qinfo->scale == 0) {
      qinfo->scale = 1;
    }
    float low_bound = -127;
    int high_bound = 127;
    qinfo->zero_point =
        std::min(high_bound,
                 static_cast<int>(std::max(low_bound, std::round(-8 - min_value / qinfo->scale))));
  } else {
    LOG(ERROR) << "get error dtype:" << cfg->dtype_input;
  }
}

void CodegenCSINN::GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo) {
  int valid_range = std::pow(2, bits - 1) - 1;
  float abs_max = std::max(std::abs(min_value), std::abs(max_value));
  qinfo->scale = abs_max / valid_range;
  qinfo->zero_point = 0;
}

QuantParams* CodegenCSINN::GetQuantParamsBase(float min_value, float max_value, int32_t tensor_type,
                                              QConfig_* quantize_cfg) {
  if (quantize_cfg == NULL) {
    quantize_cfg = cfg;
  }
  string quant_type;
  if (tensor_type == ACTIVATE) {
    quant_type = quantize_cfg->activate_quantized_type;
  } else if (tensor_type == WEIGHT) {
    quant_type = quantize_cfg->weight_quantized_type;
  }
  int bits = quantize_cfg->nbit_input;
  QuantParams* params = new QuantParams();
  Qinfo* qinfo = new Qinfo();
  qinfo->min = min_value;
  qinfo->max = max_value;
  if (quant_type == "asym") {
    GetAsymScale(min_value, max_value, bits, qinfo);
  } else if (quant_type == "sym") {
    GetSymScale(min_value, max_value, bits, qinfo);
  }

  if (qinfo->scale == 0) {
    qinfo->scale = 1.0;
  }

  GetMultiplierAndShift(qinfo->scale, &qinfo->multiplier, &qinfo->shift);
  params->qinfo = qinfo;

  return params;
}

template <typename T>
void CodegenCSINN::SisoOp(std::ostringstream& decl, const CallNode* call, const T* attr,
                          string op_name) {
  // QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 1) << "op expects 1 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name(op_name, attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item(hybrid_layer_name, complete_name);
  siso_input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 1, is_layer_hybrid);

  output2params[output_name] = complete_name;
  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
}

void CodegenCSINN::malloc_params(string struct_name, string params_name) {
  std::ostringstream t0;
  t0 << "struct " << struct_name << " *" << params_name << " = csi_alloc_params(sizeof(struct "
     << struct_name << "), sess)";
  PushDeclLine(t0);
}

void CodegenCSINN::Unary(const CallNode* call, string op_name) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOp<QnnCSIUnaryAttrs>(decl, call, attr, op_name);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("siso_params", params_name);

  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
}

void CodegenCSINN::DisoOp(const CallNode* call, string op_name, string out_dtype) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnBinaryOpAttrs>();
  CHECK(attr);
  // QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  string lhs_name, rhs_name;
  std::map<int, string> free_tensor;

  string complete_name = get_complete_layer_name(op_name, attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);

  /* Emit input0 tensor */
  VisitExpr(call->args[0]);
  // if call->args[0] is input, this check is invalid.
  // CHECK(out_.size() == 1) << "Every args expects a single out_";
  if (call->args[0].as<tvm::relay::CallNode>() || call->args[0].as<tvm::relay::VarNode>() ||
      call->args[0].as<tvm::relay::TupleGetItemNode>()) {
    lhs_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);
    free_tensor[0] = lhs_name;
    buf_idx_++;
  } else {
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto lhs = constant_[0];
    auto lhs_shape = GetShape(call->args[0]->checked_type());
    lhs_name = "lhs_" + to_string(buf_idx_);
    buf_idx_++;
    CreateHybridConstantTensor(&lhs, lhs_name, lhs_shape, attr->q_params, 0, is_layer_hybrid);
    decl << lhs_name;
  }

  decl << ", ";

  /* Emit input1 tensor */
  if (call->args[1].as<tvm::relay::CallNode>() || call->args[1].as<tvm::relay::VarNode>() ||
      call->args[1].as<tvm::relay::TupleGetItemNode>()) {
    VisitExpr(call->args[1]);
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    rhs_name = HybridInputTensor(decl, call, 1, attr->q_params, 1, is_layer_hybrid);
    free_tensor[1] = rhs_name;
  } else {
    // add constant arg
    VisitExpr(call->args[1]);
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto rhs = constant_[0];
    auto rhs_shape = GetShape(call->args[1]->checked_type());
    rhs_name = "rhs_" + to_string(buf_idx_);
    CreateHybridConstantTensor(&rhs, rhs_name, rhs_shape, attr->q_params, 1, is_layer_hybrid);
    decl << rhs_name;
  }

  /* Emit output tensor */
  // if (out_dtype == "") {
  //   out_dtype = cfg->dtype_weight;
  // }
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 2, is_layer_hybrid);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("diso_params", params_name);

  output2params[output_name] = complete_name;

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  buf_idx_++;
  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
  for (auto iter = free_tensor.begin(); iter != free_tensor.end(); iter++) {
    FreeTensor(call->args[iter->first], iter->second);
  }
}

template <typename T>
void CodegenCSINN::SetupPadding(string name, const T* attr) {
  Array<IndexExpr> pad = attr->padding;
  std::ostringstream t0;
  if (pad.size() == 4) {
    t0 << name << "->pad_top = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_left = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_down = " << to_string(pad[2].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_right = " << to_string(pad[3].as<IntImmNode>()->value);
    PushDeclLine(t0);
  } else if (pad.size() == 6) {
    t0 << name << "->pad_front = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_top = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_left = " << to_string(pad[2].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_back = " << to_string(pad[3].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_down = " << to_string(pad[4].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_right = " << to_string(pad[5].as<IntImmNode>()->value);
    PushDeclLine(t0);
  } else {
    CHECK_EQ(pad.size(), 2);
    t0 << name << "->pad_top = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_left = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_down = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_right = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
  }
}
template <typename T>
void CodegenCSINN::Setup1dPadding(string name, const T* attr) {
  Array<IndexExpr> pad = attr->padding;
  std::ostringstream t0;
  if (pad.size() == 2) {
    t0 << name << "->pad_left = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_right = " << to_string(pad[1].as<IntImmNode>()->value);
    PushDeclLine(t0);
  } else {
    CHECK_EQ(pad.size(), 1);
    t0 << name << "->pad_left = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
    t0 << name << "->pad_right = " << to_string(pad[0].as<IntImmNode>()->value);
    PushDeclLine(t0);
  }
}

template <typename T>
void CodegenCSINN::SetupConv2dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("conv2d_params", name);
  t0 << name << "->group = " << to_string(attr->groups);
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << "->dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->conv_extra.kernel_tm = NULL";
  PushDeclLine(t0);
  t0 << name << "->conv_extra.conv_mode = CSINN_DIRECT";
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupDilation2dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("dilation2d_params", name);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilations;
  t0 << name << "->dilation_height = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->dilation_width = " << to_string(dilation[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupConv3dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("conv3d_params", name);
  t0 << name << "->group = " << to_string(attr->groups);
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_depth = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_height = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_width = " << to_string(strides[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << "->dilation_depth = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->dilation_height = " << to_string(dilation[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->dilation_width = " << to_string(dilation[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupConv1dParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("conv1d_params", name);
  t0 << name << "->group = " << to_string(attr->groups);
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_width = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << name << "->dilation_width = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Setup1dPadding<T>(name, attr);
}

template <typename T>
void CodegenCSINN::SetupPoolParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("pool_params", name);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_height = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_width = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> pool_size = attr->pool_size;
  t0 << name << "->filter_height = " << to_string(pool_size[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->filter_width = " << to_string(pool_size[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  auto ceil_mode = attr->ceil_mode;
  t0 << name << "->ceil_mode = " << to_string(ceil_mode);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

template <typename T>
void CodegenCSINN::SetupPool3DParams(string name, const T* attr) {
  std::ostringstream t0;
  malloc_params("pool_params", name);
  Array<IndexExpr> strides = attr->strides;
  t0 << name << "->stride_depth = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_height = " << to_string(strides[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->stride_width = " << to_string(strides[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> pool_size = attr->pool_size;
  t0 << name << "->filter_depth = " << to_string(pool_size[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->filter_height = " << to_string(pool_size[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << name << "->filter_width = " << to_string(pool_size[2].as<IntImmNode>()->value);
  PushDeclLine(t0);
  SetupPadding(name, attr);
}

void CodegenCSINN::FreeTensor(const Expr& expr, string name) {
  auto iter = layer_count.find(expr.get());
  if (iter == layer_count.end()) {
    CHECK(0);
  }
  auto& count = iter->second;
  if (count > 1) {
    count--;
  } else {
    if (const VarNode* var = expr.as<VarNode>()) {
      name = replace(var->name_hint());
    }
    // Exclude input/output nodes
    if (io_nodes.find(name) == io_nodes.end()) {
      ext_func_body.push_back("csi_mem_free(" + name + "->data);");
      ext_func_body.push_back("csi_mem_free(" + name + ");");
    }
  }
}

std::shared_ptr<std::vector<float>> CodegenCSINN::FuseZpToBias(const CallNode* call,
                                                               QuantParams* q_params,
                                                               bool is_depthwise) {
  if (q_params[0].q_size > 1) {
    LOG(ERROR) << "only support fuse zp to bais in int8_asym_w_sym mode!";
  }
  auto weight_node = call->args[1].as<ConstantNode>();
  int w_size = weight_node->data.Length();
  float* weight_data = reinterpret_cast<float*>(malloc(w_size));
  weight_node->data.CopyToBytes(weight_data, w_size);

  auto bias_node = call->args[2].as<ConstantNode>();
  int b_size = bias_node->data.Length();
  float* bias_data = reinterpret_cast<float*>(malloc(b_size));
  bias_node->data.CopyToBytes(bias_data, b_size);

  auto in_params = q_params[0];

  auto b_shape = GetShape(call->args[2]->checked_type());
  auto w_shape = GetShape(call->args[1]->checked_type());
  int b_length = b_shape.size() ? b_shape[0] : w_shape[0];
  float sp = q_params[0].qinfo->scale * q_params[0].qinfo->zero_point;
  auto out = std::make_shared<std::vector<float>>();

  if (layout_ == "NHWC" && is_depthwise) {
    int outer_size = 1;
    for (uint i = 0; i < w_shape.size() - 1; i++) {
      outer_size *= w_shape[i];
    }
    for (int i = 0; i < b_length; i++) {
      float new_b = b_shape.size() ? bias_data[i] : 0.0;
      for (int j = 0; j < outer_size; j++) {
        int w_index = b_length * j + i;
        new_b -= weight_data[w_index] * sp;
      }
      out->push_back(new_b);
    }
  } else {
    int inner_size = 1;
    for (uint i = 1; i < w_shape.size(); i++) {
      inner_size *= w_shape[i];
    }
    for (int i = 0; i < b_length; i++) {
      float new_b = b_shape.size() ? bias_data[i] : 0.0;
      for (int j = 0; j < inner_size; j++) {
        int w_index = i * inner_size + j;
        new_b -= weight_data[w_index] * sp;
      }
      out->push_back(new_b);
    }
  }

  free(bias_data);
  free(weight_data);
  return out;
}

void CodegenCSINN::Conv1d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  const auto* attr = call->attrs.as<QnnCSIConv1DAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 3) << "Conv1d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());
  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(&kernel, kernel_name, wshape, cfg->dtype_weight, q_params[1]);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(&bias, bias_name, bshape, cfg->dtype_activation, q_params[0], q_params[1],
                       q_params[2]);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  SetupConv1dParams<QnnCSIConv1DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "conv1d", params_name, attr->layer_name.c_str());
  end_stream(decl, "conv1d");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::params_hybrid_setup(std::ostringstream& decl, string op_name,
                                       std::vector<int> shape, string params_name,
                                       string layer_name, string dtype, string layout) {
  std::ostringstream t0;
  if (!(layout_ == "NCHW" && layout == "CSINN_LAYOUT_NCHW")) {
    t0 << params_name << "->base.layout = CSINN_LAYOUT_" << layout_;
    PushDeclLine(t0);
  }
  t0 << params_name << "->base.name = "
     << "\"" << op_name + "_" + layer_name << "\"";
  PushDeclLine(t0);
  setup_callback(decl, op_name, params_name);
  CreateHybridTensorSessData(shape, dtype);
}

string CodegenCSINN::InsertDataConvert(const CallNode* call, int input_index, string input_name,
                                       QuantParams quant_params, string dtype) {
  std::vector<int> ishape;
  if (auto pre_call = call->args[input_index].as<CallNode>()) {
    ishape = GetShape(pre_call->checked_type());
  } else if (auto pre_var = call->args[input_index].as<VarNode>()) {
    ishape = GetShape(pre_var->checked_type());
  } else {
    auto pre_call = call->args[input_index].as<TupleGetItemNode>();
    CHECK(pre_call);
    ishape = GetShape(pre_call->checked_type());
  }

  string hybrid_output_name = DataConvertTensor(ishape, quant_params, dtype);

  std::ostringstream decl;
  decl << "(" << input_name << ", " << hybrid_output_name << ", ";
  string params_name = "hybrid_params_" + to_string(buf_idx_);
  decl << params_name << ")";

  malloc_params("siso_params", params_name);

  params_hybrid_setup(decl, "data_convert", ishape, params_name, params_name, dtype);

  std::ostringstream func;
  func << "csi_data_convert" << decl.str() << ";";
  ext_func_body.push_back(func.str());

  hybrid_buffer_name_.push_back(hybrid_output_name);
  return hybrid_output_name;
}

string CodegenCSINN::HybridInputTensor(std::ostringstream& decl, const CallNode* call,
                                       int input_index, Array<Array<IndexExpr>> q_params,
                                       int params_index, bool is_layer_hybrid) {
  bool input_hybrid_quantize = false;

  // get input tensor name
  string tensor_name = "";
  if (auto pre_call = call->args[input_index].as<CallNode>()) {
    auto input = GetRealInput(pre_call);
    tensor_name = input.name;
  } else if (auto pre_call = call->args[input_index].as<TupleGetItemNode>()) {
    auto pre_tuple = pre_call->tuple.as<tvm::relay::CallNode>();
    auto input = GetRealInput(pre_tuple);
    tensor_name = input.name;
  }

  // if hybrid_layer_name includes the params name of last layer, we can infer that
  // the pre-layer's output tensor of the current layer has been converted into hybrid quantization.
  auto iter = output2params.find(tensor_name);
  if (iter != output2params.end() && is_contain_item<string>(hybrid_layer_name, iter->second)) {
    input_hybrid_quantize = true;
  }

  string input_name = "";
  if (input_hybrid_quantize) {
    QuantParams* hybrid_q_params =
        GetQuantParams(get_quant_params_expr(q_params, params_index), hybrid_cfg);
    input_name = InputTensorName(call, input_index, hybrid_q_params[0], hybrid_cfg->dtype_weight);

    if (!is_contain_item<string>(hybrid_layer_name, input_name) && !is_layer_hybrid) {
      // hybrid input +
      QuantParams* base_q_params = GetQuantParams(get_quant_params_expr(q_params, 0), cfg);
      string base_input_name =
          InsertDataConvert(call, input_index, input_name, base_q_params[0], cfg->dtype_weight);
      if (input_index == 0) {
        decl.str("");
        decl << "(" << base_input_name;
      } else {
        decl << base_input_name;
      }
    } else {
      decl << input_name;
    }
  } else {
    QuantParams* base_q_params = GetQuantParams(get_quant_params_expr(q_params, params_index), cfg);
    input_name = InputTensorName(call, input_index, base_q_params[0], cfg->dtype_weight);

    if (is_contain_item<string>(hybrid_layer_name, input_name) || is_layer_hybrid) {
      QuantParams* hybrid_q_params = GetQuantParams(get_quant_params_expr(q_params, 0), hybrid_cfg);
      string hybrid_input_name = InsertDataConvert(call, input_index, input_name,
                                                   hybrid_q_params[0], hybrid_cfg->dtype_weight);
      if (input_index == 0) {
        decl.str("");
        decl << "(" << hybrid_input_name;
      } else {
        decl << hybrid_input_name;
      }
    } else {
      decl << input_name;
    }
  }

  return input_name;
}

string CodegenCSINN::HybridOutputTensor(std::ostringstream& decl, const CallNode* call,
                                        Array<Array<IndexExpr>> q_params, int params_index,
                                        bool is_layer_hybrid) {
  string output_name = "output_" + to_string(buf_idx_);
  if (is_layer_hybrid) {
    QuantParams* hybrid_q_params =
        GetQuantParams(get_quant_params_expr(q_params, params_index), hybrid_cfg);
    // output_name = OutputTensor(decl, call, hybrid_q_params[0], hybrid_cfg->dtype_weight);
    auto type_node = call->checked_type().as<TensorTypeNode>();
    CHECK(type_node);
    auto out_shape = GetShape(call->checked_type());
    // if output is a single number, out_shape.size() here is zero
    if (out_shape.size() == 0) {
      out_shape.push_back(1);
    }
    hybrid_q_params[0].name = output_name;
    hybrid_q_params[0].offset = constant_offset;
    hybrid_q_params[0].shape = out_shape;
    CreateTensor(output_name, "hybrid_alloc", out_shape, hybrid_q_params[0],
                 hybrid_cfg->dtype_weight);
    decl << ", " << output_name;
    int out_index = CheckOutput(call);
    if (out_index > -1) {
      io_nodes[output_name] = hybrid_q_params[0];
    }

    CreateHybridTensorSessData(out_shape, hybrid_cfg->dtype_weight);
  } else {
    QuantParams* base_q_params = GetQuantParams(get_quant_params_expr(q_params, params_index), cfg);
    output_name = OutputTensor(decl, call, base_q_params[0], cfg->dtype_weight);
  }
  return output_name;
}

void CodegenCSINN::CreateHybridConstantTensor(CSIConstant* data, string name,
                                              std::vector<int> shape,
                                              Array<Array<IndexExpr>> q_params, int params_index,
                                              bool is_layer_hybrid, bool depthwise_kernel) {
  if (is_layer_hybrid || is_contain_item<string>(hybrid_layer_name, name)) {
    QuantParams* hybrid_q_params =
        GetQuantParams(get_quant_params_expr(q_params, params_index), hybrid_cfg);
    CreateConstantTensor(data, name, shape, hybrid_cfg->dtype_weight, hybrid_q_params[0],
                         depthwise_kernel);
  } else {
    QuantParams* base_q_params = GetQuantParams(get_quant_params_expr(q_params, params_index), cfg);
    CreateConstantTensor(data, name, shape, cfg->dtype_weight, base_q_params[0], depthwise_kernel);
  }
}

void CodegenCSINN::CreateBiasTensor(const CallNode* call, CSIConstant* data, string name,
                                    Array<Array<IndexExpr>> q_params, bool* fuse_zp,
                                    bool is_layer_hybrid, bool is_input_hybrid,
                                    bool is_weight_hybrid, bool depthwise_kernel) {
  // bool fuse_zp = false;
  std::shared_ptr<std::vector<float>> new_bias;
  if (is_contain_item<string>(hybrid_layer_name, name) || is_layer_hybrid) {
    if ((hybrid_cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM" ||
         hybrid_cfg->quantization_scheme == "CSINN_QUANT_INT4_ASYM_W_SYM") &&
        cfg->fuse_zp2bias) {
      *fuse_zp = true;
      QuantParams* hybrid_q_params = GetQuantParams(q_params, hybrid_cfg);
      new_bias = FuseZpToBias(call, hybrid_q_params, depthwise_kernel);
    }
  } else {
    if ((cfg->quantization_scheme == "CSINN_QUANT_INT8_ASYM_W_SYM" ||
         cfg->quantization_scheme == "CSINN_QUANT_INT4_ASYM_W_SYM") &&
        cfg->fuse_zp2bias) {
      *fuse_zp = true;
      QuantParams* base_q_params = GetQuantParams(q_params, cfg);
      new_bias = FuseZpToBias(call, base_q_params, depthwise_kernel);
    }
  }

  auto bshape = GetShape(call->args[2]->checked_type());
  if (*fuse_zp) {
    if (bshape.size() == 0) {
      free(data->data_buf);
      data->size = new_bias->size() * 4;
      data->data_buf = reinterpret_cast<float*>(malloc(data->size));
      bshape.push_back(new_bias->size());
    }
    float* data_buf = static_cast<float*>(data->data_buf);
    std::copy(new_bias->begin(), new_bias->end(), data_buf);
  }

  QuantParams in_q_params;
  QuantParams weight_q_params;
  QuantParams bias_q_params;

  auto p = GetQuantParams(q_params, cfg);
  in_q_params = p[0];
  weight_q_params = p[1];
  bias_q_params = p[1];

  string input_dtype = cfg->dtype_weight;
  string weight_dtype = cfg->dtype_weight;
  string bias_dtype = cfg->dtype_activation;
  if (is_input_hybrid) {
    p = GetQuantParams(get_quant_params_expr(q_params, 0), hybrid_cfg);
    in_q_params = p[0];
    input_dtype = hybrid_cfg->dtype_weight;
  }
  if (is_weight_hybrid) {
    p = GetQuantParams(get_quant_params_expr(q_params, 1), hybrid_cfg);
    weight_q_params = p[0];
    weight_dtype = hybrid_cfg->dtype_weight;
  }
  if (is_layer_hybrid) {
    auto hybrid_q_params = GetQuantParams(q_params, hybrid_cfg);
    in_q_params = hybrid_q_params[0];
    weight_q_params = hybrid_q_params[1];
    bias_q_params = hybrid_q_params[2];

    input_dtype = hybrid_cfg->dtype_weight;
    weight_dtype = hybrid_cfg->dtype_weight;
    bias_dtype = hybrid_cfg->dtype_activation;
  }

  if (is_contain_item<string>(hybrid_layer_name, name)) {
    if (input_dtype == "int16_t" && weight_dtype == "int16_t" && bias_dtype == "int32_t") {
      CreateConstantTensor(data, name, bshape, hybrid_cfg->dtype_activation, bias_q_params, false,
                           true);
    } else {
      CreateConstantTensor(data, name, bshape, hybrid_cfg->dtype_activation, in_q_params,
                           weight_q_params, bias_q_params);
    }
  } else {
    if (input_dtype == "int16_t" && weight_dtype == "int16_t" && bias_dtype == "int32_t") {
      CreateConstantTensor(data, name, bshape, cfg->dtype_activation, bias_q_params, false, true);
    } else {
      CreateConstantTensor(data, name, bshape, cfg->dtype_activation, in_q_params, weight_q_params,
                           bias_q_params);
    }
  }
}

void CodegenCSINN::Conv2d(const CallNode* call, string op_name) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIConv2DAttrs>();
  CHECK(attr);
  // QuantParams* q_params = GetQuantParams(attr->q_params);
  CHECK(call->args.size() == 3) << "Conv2d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  // check for depthwise
  auto ishape = GetShape(call->args[0]->checked_type());
  auto wshape = GetShape(call->args[1]->checked_type());
  auto bshape = GetShape(call->args[2]->checked_type());

  bool depthwise_kernel = is_depthwise(ishape, wshape, attr->groups, layout_);

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string params_name = "params_" + to_string(buf_idx_);
  string complete_name = get_complete_layer_name(op_name, attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);

  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 3, is_layer_hybrid);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  string kernel_name = "kernel_" + to_string(buf_idx_);

  CreateHybridConstantTensor(&kernel, kernel_name, wshape, attr->q_params, 1, is_layer_hybrid,
                             depthwise_kernel);

  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;
  bool is_input_hybrid = is_contain_item<string>(hybrid_layer_name, input_name);
  bool is_weight_hybrid = is_contain_item<string>(hybrid_layer_name, kernel_name);
  CreateBiasTensor(call, &bias, bias_name, attr->q_params, &fuse_zp, is_layer_hybrid,
                   is_input_hybrid, is_weight_hybrid, depthwise_kernel);
  decl << ", " << bias_name;

  output2params[output_name] = complete_name;

  decl << ", " << params_name << ")";

  SetupConv2dParams<QnnCSIConv2DAttrs>(params_name, attr);
  if (fuse_zp) {
    std::ostringstream t0;
    t0 << params_name << "->conv_extra.fuse_zp2bias = true";
    PushDeclLine(t0);
  }
  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }

  params_common_setup(decl, call, op_name, params_name, attr->layer_name.c_str());
  end_stream(decl, op_name);
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Conv3d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  const auto* attr = call->attrs.as<QnnCSIConv3DAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 3) << "Conv3d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());
  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(&kernel, kernel_name, wshape, cfg->dtype_weight, q_params[1]);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(&bias, bias_name, bshape, cfg->dtype_activation, q_params[0], q_params[1],
                       q_params[2]);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  SetupConv3dParams<QnnCSIConv3DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "conv3d", params_name, attr->layer_name.c_str());
  end_stream(decl, "conv3d");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Dilation2d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIDilation2DAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 2) << "Dilation2D expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[2], cfg->dtype_weight);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(&kernel, kernel_name, wshape, cfg->dtype_weight, q_params[1]);

  decl << ", " << kernel_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  SetupDilation2dParams<QnnCSIDilation2DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "dilation2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "dilation2d");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::DeConv2d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIDeConv2DAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 3) << "DeConv2d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string complete_name = get_complete_layer_name("deconv2d", attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);

  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 3, is_layer_hybrid);

  output2params[output_name] = complete_name;

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());
  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateHybridConstantTensor(&kernel, kernel_name, wshape, attr->q_params, 1, is_layer_hybrid);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;
  bool is_input_hybrid = is_contain_item<string>(hybrid_layer_name, input_name);
  bool is_weight_hybrid = is_contain_item<string>(hybrid_layer_name, kernel_name);
  CreateBiasTensor(call, &bias, bias_name, attr->q_params, &fuse_zp, is_layer_hybrid,
                   is_input_hybrid, is_weight_hybrid);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  SetupConv2dParams<QnnCSIDeConv2DAttrs>(params_name, attr);

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  params_common_setup(decl, call, "deconv2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "deconv2d");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::DeConv3d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIDeConv3DAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  CHECK(call->args.size() == 3) << "DeConv3d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(&kernel, kernel_name, wshape, cfg->dtype_weight, q_params[1]);

  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(&bias, bias_name, bshape, cfg->dtype_activation, q_params[0], q_params[1],
                       q_params[2]);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  SetupConv3dParams<QnnCSIDeConv3DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "deconv3d", params_name, attr->layer_name.c_str());
  end_stream(decl, "deconv3d");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Dense(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* dense_attr = call->attrs.as<QnnCSIDenseAttrs>();
  CHECK(dense_attr);

  CHECK(call->args.size() == 3) << "Dense expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string complete_name = get_complete_layer_name("fullyconnected", dense_attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);

  string input_name = HybridInputTensor(decl, call, 0, dense_attr->q_params, 0, is_layer_hybrid);

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, dense_attr->q_params, 3, is_layer_hybrid);
  output2params[output_name] = complete_name;

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());

  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateHybridConstantTensor(&kernel, kernel_name, wshape, dense_attr->q_params, 1,
                             is_layer_hybrid);

  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  bool fuse_zp = false;
  bool is_input_hybrid = is_contain_item<string>(hybrid_layer_name, input_name);
  bool is_weight_hybrid = is_contain_item<string>(hybrid_layer_name, kernel_name);
  CreateBiasTensor(call, &bias, bias_name, dense_attr->q_params, &fuse_zp, is_layer_hybrid,
                   is_input_hybrid, is_weight_hybrid);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  malloc_params("fc_params", params_name);
  int units;
  if (dense_attr->units.defined()) {
    units = dense_attr->units.as<IntImmNode>()->value;
  } else {
    units = wshape[0];
  }
  t0 << params_name << "->units = " << to_string(units);
  PushDeclLine(t0);
  if (fuse_zp) {
    t0 << params_name << "->fc_extra.fuse_zp2bias = true";
    PushDeclLine(t0);
  }
  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }

  params_common_setup(decl, call, "fullyconnected", params_name, dense_attr->layer_name.c_str());
  end_stream(decl, "fullyconnected");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Softmax(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOp<QnnCSIAxisAttrs>(decl, call, attr, "softmax");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("softmax_params", params_name);
  int actual_aixs = attr->axis;
  auto ishape = GetShape(call->args[0]->checked_type());
  if (attr->axis < 0) {
    actual_aixs += ishape.size();
  }
  t0 << params_name << "->axis = " << to_string(actual_aixs);
  PushDeclLine(t0);

  params_common_setup(decl, call, "softmax", params_name, attr->layer_name.c_str());
  end_stream(decl, "softmax");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Reverse(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOp<QnnCSIAxisAttrs>(decl, call, attr, "reverse");
  auto ishape = GetShape(call->args[0]->checked_type());
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("reverse_params", params_name);
  int axis = attr->axis < 0 ? attr->axis + ishape.size() : attr->axis;
  t0 << params_name << "->axis = " << axis;
  PushDeclLine(t0);

  params_common_setup(decl, call, "reverse", params_name, attr->layer_name.c_str());
  end_stream(decl, "reverse");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::LogSoftmax(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIAxisAttrs>();
  SisoOp<QnnCSIAxisAttrs>(decl, call, attr, "log_softmax");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("softmax_params", params_name);
  int axis = attr->axis == -1 ? 1 : attr->axis;
  t0 << params_name << "->axis = " << to_string(axis);
  PushDeclLine(t0);

  params_common_setup(decl, call, "log_softmax", params_name, attr->layer_name.c_str());
  end_stream(decl, "log_softmax");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::ExpandDims(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIExpandDimsAttrs>();
  SisoOp<QnnCSIExpandDimsAttrs>(decl, call, attr, "expand_dims");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("expand_dims_params", params_name);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  PushDeclLine(t0);

  params_common_setup(decl, call, "expand_dims", params_name, attr->layer_name.c_str());
  end_stream(decl, "expand_dims");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::MaxPool2d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DAttrs>();
  SisoOp<QnnCSIMaxPool2DAttrs>(decl, call, attr, "maxpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  SetupPoolParams(params_name, attr);

  params_common_setup(decl, call, "maxpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "maxpool2d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::AvgPool2d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIAvgPool2DAttrs>();
  SisoOp<QnnCSIAvgPool2DAttrs>(decl, call, attr, "avgpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  SetupPoolParams(params_name, attr);
  std::ostringstream t0;
  auto count_include_pad = attr->count_include_pad;
  t0 << params_name << "->count_include_pad = " << to_string(count_include_pad);
  PushDeclLine(t0);

  params_common_setup(decl, call, "avgpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "avgpool2d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::AvgPool3d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIAvgPool3DAttrs>();
  SisoOp<QnnCSIAvgPool3DAttrs>(decl, call, attr, "avgpool3d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  SetupPool3DParams(params_name, attr);

  params_common_setup(decl, call, "avgpool3d", params_name, attr->layer_name.c_str(),
                      "CSINN_NCDHW");
  end_stream(decl, "avgpool3d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::MaxPool3d(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool3DAttrs>();
  SisoOp<QnnCSIMaxPool3DAttrs>(decl, call, attr, "maxpool3d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  SetupPool3DParams(params_name, attr);

  params_common_setup(decl, call, "maxpool3d", params_name, attr->layer_name.c_str(),
                      "CSINN_NCDHW");
  end_stream(decl, "maxpool3d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::GlobalAvgPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIGlobalAvgPoolAttrs>();
  SisoOp<QnnCSIGlobalAvgPoolAttrs>(decl, call, attr, "global_avgpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("pool_params", params_name);

  params_common_setup(decl, call, "global_avgpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "global_avgpool2d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::GlobalMaxPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIGlobalMaxPoolAttrs>();
  SisoOp<QnnCSIGlobalMaxPoolAttrs>(decl, call, attr, "global_maxpool2d");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("pool_params", params_name);

  params_common_setup(decl, call, "global_maxpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "global_maxpool2d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Maxpool2dWithArgmax(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DAttrs>();
  SisoOp<QnnCSIMaxPool2DAttrs>(decl, call, attr);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  SetupPoolParams(params_name, attr);

  params_common_setup(decl, call, "maxpool2d", params_name, attr->layer_name.c_str());
  end_stream(decl, "maxpool2d");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::MaxPool2dLocat(const CallNode* call) {
  std::ostringstream decl;
  const auto* attr = call->attrs.as<QnnCSIMaxPool2DLocatAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 1) << "MaxPool2dLocat expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[1], "int32_t");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  SetupPoolParams(params_name, attr);

  PushOutput(output_name, call, "int32_t");
  params_common_setup(decl, call, "maxpool2d_locat", params_name, attr->layer_name.c_str());
  end_stream(decl, "maxpool2d_locat");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::UnPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnPoolingAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 2) << "Unpool2d expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  decl << ", ";

  /* Emit_ mask tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string mask_name = InputTensor(decl, call, 1, q_params[1], "int32_t");

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[2], cfg->dtype_weight);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("unpooling_params", params_name);

  t0 << params_name
     << "->pad_out_height = " << to_string(attr->out_padding[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name
     << "->pad_out_width = " << to_string(attr->out_padding[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << "->scale_height = " << to_string(attr->scales[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << "->scale_width = " << to_string(attr->scales[1].as<IntImmNode>()->value);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "unpooling", params_name, attr->layer_name.c_str());
  end_stream(decl, "unpooling");
  FreeTensor(call->args[0], input_name);
  FreeTensor(call->args[1], mask_name);
}

void CodegenCSINN::PSROIPool(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIPSROIPoolingAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 2) << "PSROIPooling expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  decl << ", ";

  /* Emit_ roi tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string roi_name = InputTensor(decl, call, 1, q_params[1], "int32_t");

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[2], cfg->dtype_weight);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(attr->spatial_scale, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("psroipooling_params", params_name);

  t0 << params_name << "->output_dim = " << to_string(attr->output_dim);
  PushDeclLine(t0);
  t0 << params_name << "->group_size = " << to_string(attr->group_size);
  PushDeclLine(t0);
  t0 << params_name << "->spatial_scale = " << to_string(attr->spatial_scale);
  PushDeclLine(t0);
  t0 << params_name << "->spatial_scale_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->spatial_scale_shift = " << to_string(shift);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "psroipooling", params_name, attr->layer_name.c_str());
  end_stream(decl, "psroipooling");
  FreeTensor(call->args[0], input_name);
  FreeTensor(call->args[1], roi_name);
}

void CodegenCSINN::ROIPool(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIROIPoolingAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  CHECK(call->args.size() == 2) << "ROIPooling expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  decl << ", ";

  /* Emit_ roi tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string roi_name = InputTensor(decl, call, 1, q_params[1], "int32_t");

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[2], cfg->dtype_weight);

  int32_t multiplier;
  int32_t shift;
  GetMultiplierAndShift(attr->spatial_scale, &multiplier, &shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  Array<IndexExpr> pooled_size = attr->pooled_size;

  malloc_params("roi_pool_params", params_name);

  t0 << params_name << "->pooled_size_h = " << to_string(pooled_size[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << "->pooled_size_w = " << to_string(pooled_size[1].as<IntImmNode>()->value);
  PushDeclLine(t0);
  t0 << params_name << "->spatial_scale = " << to_string(attr->spatial_scale);
  PushDeclLine(t0);
  t0 << params_name << "->spatial_scale_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->spatial_scale_shift = " << to_string(shift);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "roipool", params_name, attr->layer_name.c_str());
  end_stream(decl, "roipool");
  FreeTensor(call->args[0], input_name);
  FreeTensor(call->args[1], roi_name);
}

void CodegenCSINN::Proposal(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  std::ostringstream mstream, sstream, fstream;
  const auto* attr = call->attrs.as<QnnCSIProposalAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  CHECK(call->args.size() == 3) << "Proposal expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ cls tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string cls_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  decl << ", ";

  /* Emit_ bbox tensor */
  VisitExpr(call->args[1]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string bbox_name = InputTensor(decl, call, 1, q_params[1], cfg->dtype_weight);

  /* Emit_ im_info tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto im_info = constant_[0];
  auto im_info_shape = GetShape(call->args[2]->checked_type());
  string im_info_name = "im_info_" + to_string(buf_idx_);
  string layout = GetCSINNWeightLayout(im_info_shape);
  CreateConstantTensorBase(im_info_name, im_info.size, im_info_shape, "int32_t", layout);
  t0 << im_info_name << "->dtype = CSINN_DTYPE_FLOAT32";
  PushDeclLine(t0);

  decl << "," << im_info_name;

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

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
  decl << ", " << params_name << ")";

  malloc_params("proposal_params", params_name);
  t0 << params_name << "->scales = scale_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << "->scale_multipliers = scale_multipliers_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << "->scale_shifts = scale_shifts_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << "->scales_num = " << to_string(scales_num);
  PushDeclLine(t0);
  t0 << params_name << "->ratios = ratios_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << "->ratio_multipliers = ratio_multipliers_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << "->ratio_shifts = ratio_shifts_" << to_string(buf_idx_);
  PushDeclLine(t0);
  t0 << params_name << "->ratios_num = " << to_string(ratios_num);
  PushDeclLine(t0);
  t0 << params_name << "->feature_stride = " << to_string(attr->feature_stride);
  PushDeclLine(t0);
  t0 << params_name << "->threshold = " << to_string(attr->threshold);
  PushDeclLine(t0);
  t0 << params_name << "->threshold_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->threshold_shift = " << to_string(shift);
  PushDeclLine(t0);
  t0 << params_name << "->rpn_pre_nms_top_n = " << to_string(attr->rpn_pre_nms_top_n);
  PushDeclLine(t0);
  t0 << params_name << "->rpn_post_nms_top_n = " << to_string(attr->rpn_post_nms_top_n);
  PushDeclLine(t0);
  t0 << params_name << "->rpn_min_size = " << to_string(attr->rpn_min_size);
  PushDeclLine(t0);
  t0 << params_name << "->iou_loss = " << to_string(attr->iou_loss);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "proposal", params_name, attr->layer_name.c_str());
  end_stream(decl, "proposal");
  FreeTensor(call->args[0], cls_name);
  FreeTensor(call->args[1], bbox_name);
}

void CodegenCSINN::UpSampling(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUpSamplingAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 1) << "UpSampling expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("resize", attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);
  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 1, is_layer_hybrid);
  output2params[output_name] = complete_name;

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("resize_params", params_name);
  t0 << params_name << "->resize_mode = ";
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
  t0 << params_name << "->align_corners = " << to_string(attr->align_corners);
  PushDeclLine(t0);

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  params_common_setup(decl, call, "resize", params_name, attr->layer_name.c_str());
  end_stream(decl, "resize");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Relu(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream buf;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOp<QnnCSIUnaryAttrs>(decl, call, attr, "relu");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("relu_params", params_name);
  params_common_setup(decl, call, "relu", params_name, attr->layer_name.c_str());
  end_stream(decl, "relu");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Fsmn(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIFsmnAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  CHECK(call->args.size() == 5) << "fsmn expects 5 args";

  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit l_filter tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto l_filter = constant_[0];
  auto lshape = GetShape(call->args[1]->checked_type());
  string l_filter_name = "l_filter_" + to_string(buf_idx_);
  CreateConstantTensor(&l_filter, l_filter_name, lshape, cfg->dtype_weight, q_params[1]);
  decl << ", " << l_filter_name;

  /* Emit r_filter tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto r_filter = constant_[0];
  auto rshape = GetShape(call->args[2]->checked_type());
  string r_filter_name = "r_filter_" + to_string(buf_idx_);
  CreateConstantTensor(&r_filter, r_filter_name, rshape, cfg->dtype_weight, q_params[2]);
  decl << ", " << r_filter_name;

  /* Emit frame sequence tensor */
  VisitExpr(call->args[3]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto sequence = constant_[0];
  auto seq_shape = GetShape(call->args[3]->checked_type());
  string sequence_name = "sequence_" + to_string(buf_idx_);
  CreateConstantTensor(&sequence, sequence_name, seq_shape, cfg->dtype_weight, q_params[3]);
  decl << ", " << sequence_name;

  /* Emit frame counter tensor */
  VisitExpr(call->args[4]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto frame_counter = constant_[0];
  auto counter_shape = GetShape(call->args[4]->checked_type());
  string counter_name = "frame_counter_" + to_string(buf_idx_);
  CreateConstantTensor(&frame_counter, counter_name, counter_shape, "int32_t", q_params[4]);
  decl << ", " << counter_name;

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[4], cfg->dtype_weight);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("fsmn_params", params_name);

  t0 << params_name << "->l_order = " << to_string(attr->l_order);
  PushDeclLine(t0);
  t0 << params_name << "->r_order = " << to_string(attr->r_order);
  PushDeclLine(t0);
  t0 << params_name << "->l_stride = " << to_string(attr->l_stride);
  PushDeclLine(t0);
  t0 << params_name << "->r_stride = " << to_string(attr->r_stride);
  PushDeclLine(t0);
  t0 << params_name << "->unavailable_frames = " << to_string(attr->unavailable_frames);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "fsmn", params_name, attr->layer_name.c_str());
  end_stream(decl, "fsmn");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Full(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIFullAttrs>();
  auto shape = attr->shape;
  SisoOp<QnnCSIFullAttrs>(decl, call, attr, "full");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  t0 << "int32_t *shape_" << buf_idx_ << " = malloc(" << shape.size() << " * 4)";
  PushDeclLine(t0);
  for (uint k = 0; k < shape.size(); k++) {
    t0 << "shape_" << buf_idx_ << "[" << k << "] = " << Downcast<IndexExpr>(shape[k]);
    PushDeclLine(t0);
  }

  malloc_params("full_params", params_name);
  t0 << params_name << "->shape = shape_" << buf_idx_;
  PushDeclLine(t0);

  params_common_setup(decl, call, "full", params_name, attr->layer_name.c_str());
  end_stream(decl, "full");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Take(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSITakeAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  String mode = attr->mode;
  // CHECK(mode == "fast") << "only mode is fast, input indices are in bound.";
  CHECK(call->args.size() == 2) << "take expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  auto in_shape = GetShape(call->args[0]->checked_type());
  int* axis = NULL;
  if (attr->axis.defined()) {
    axis = static_cast<int*>(malloc(4));
    axis[0] = static_cast<int>(attr->axis->value);
    axis[0] = axis[0] < 0 ? axis[0] + in_shape.size() : axis[0];
  }

  /* Emit indices tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto indices = constant_[0];
  auto indices_shape = GetShape(call->args[1]->checked_type());
  string indices_name = "indices_" + to_string(buf_idx_);
  CreateConstantTensor(&indices, indices_name, indices_shape, "int32_t", q_params[1]);
  decl << ", " << indices_name;

  string params_name = "params_" + to_string(buf_idx_);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[2], cfg->dtype_weight);

  decl << ", " << params_name << ")";

  /* Use gather op */
  malloc_params("gather_params", params_name);
  if (axis == NULL) {
    t0 << params_name << "->axis = NULL";
  } else {
    t0 << params_name << "->axis = " << axis[0];
  }
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "gather", params_name, attr->layer_name.c_str());
  end_stream(decl, "gather");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Clip(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIClipAttrs>();
  double min = attr->a_min;
  double max = attr->a_max;

  SisoOp<QnnCSIClipAttrs>(decl, call, attr, "clip");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("clip_params", params_name);
  t0 << params_name << "->min_value = " << to_string(min);
  PushDeclLine(t0);
  t0 << params_name << "->max_value = " << to_string(max);
  PushDeclLine(t0);

  params_common_setup(decl, call, "clip", params_name, attr->layer_name.c_str());
  end_stream(decl, "clip");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Pad(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIPadAttrs>();
  auto pad_width = attr->pad_width;
  string pad_mode = attr->pad_mode;
  QuantParams* q_params = GetQuantParams(attr->q_params);
  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  siso_input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[2], cfg->dtype_weight);

  PushOutput(output_name, call);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  t0 << "int32_t *pad_before_" << buf_idx_ << " = malloc(" << pad_width.size() << " * 4)";
  PushDeclLine(t0);
  for (uint k = 0; k < pad_width.size(); k++) {
    t0 << "pad_before_" << buf_idx_ << "[" << k << "] = " << Downcast<IndexExpr>(pad_width[k][0]);
    PushDeclLine(t0);
  }

  t0 << "int32_t *pad_after_" << buf_idx_ << " = malloc(" << pad_width.size() << " * 4)";
  PushDeclLine(t0);
  for (uint k = 0; k < pad_width.size(); k++) {
    t0 << "pad_after_" << buf_idx_ << "[" << k << "] = " << Downcast<IndexExpr>(pad_width[k][1]);
    PushDeclLine(t0);
  }

  malloc_params("pad_params", params_name);
  t0 << params_name << "->pad_before = pad_before_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->pad_after = pad_after_" << buf_idx_;
  PushDeclLine(t0);
  if (pad_mode == "constant") {
    t0 << params_name << "->pad_mode = CSINN_PAD_CONSTANT";
  } else {
    t0 << params_name << "->pad_mode = CSINN_PAD_EDGE";
  }

  PushDeclLine(t0);
  VisitExpr(call->args[1]);
  auto pad_value = constant_[0];
  float* pad_value_ = reinterpret_cast<float*>(pad_value.data_buf);
  /* FIXME: real pad_value in arg[1] */
  t0 << params_name << "->pad_value = " << *pad_value_;
  PushDeclLine(t0);
  t0 << params_name << "->pad_num = " << to_string(pad_width.size());
  PushDeclLine(t0);

  params_common_setup(decl, call, "pad", params_name, attr->layer_name.c_str());
  end_stream(decl, "pad");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Tile(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSITileAttrs>();
  auto reps = attr->reps;
  SisoOp<QnnCSITileAttrs>(decl, call, attr, "tile");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  t0 << "int32_t *reps_" << buf_idx_ << " = malloc(" << reps.size() << " * 4)";
  PushDeclLine(t0);
  for (uint k = 0; k < reps.size(); k++) {
    t0 << "reps_" << buf_idx_ << "[" << k << "] = " << Downcast<IndexExpr>(reps[k]);
    PushDeclLine(t0);
  }

  malloc_params("reps_params", params_name);
  t0 << params_name << "->reps = reps_" << buf_idx_;
  PushDeclLine(t0);

  params_common_setup(decl, call, "tile", params_name + ".tile", attr->layer_name.c_str());
  end_stream(decl, "tile");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::DepthToSpace(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISubPixelAttrs>();
  int block_size = attr->block_size;
  string mode = attr->mode;
  SisoOp<QnnCSISubPixelAttrs>(decl, call, attr, "depth_to_space");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("depth_to_space_params", params_name);
  t0 << params_name << "->block_size = " << to_string(block_size);
  PushDeclLine(t0);
  if (mode == "DCR") {
    t0 << params_name << "->mode = CSINN_DEPTHTOSPACE_DCR";
  } else if (mode == "CDR") {
    t0 << params_name << "->mode = CSINN_DEPTHTOSPACE_CRD";
  }
  PushDeclLine(t0);

  params_common_setup(decl, call, "depth_to_space", params_name, attr->layer_name.c_str());
  end_stream(decl, "depth_to_space");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::SpaceToDepth(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISubPixelAttrs>();
  int block_size = attr->block_size;
  SisoOp<QnnCSISubPixelAttrs>(decl, call, attr, "space_to_depth");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("space_to_depth_params", params_name);
  t0 << params_name << "->block_size = " << to_string(block_size);
  PushDeclLine(t0);

  params_common_setup(decl, call, "space_to_depth", params_name, attr->layer_name.c_str());
  end_stream(decl, "space_to_depth");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Relu6(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  SisoOp<QnnCSIUnaryAttrs>(decl, call, attr, "relu6");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("relu_params", params_name);
  t0 << params_name << "->n = 6";
  PushDeclLine(t0);

  params_common_setup(decl, call, "relu6", params_name, attr->layer_name.c_str());
  end_stream(decl, "relu6");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::PRelu(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIPReluAttrs>();
  CHECK(attr);
  CHECK(call->args.size() == 2) << "PRelu expects 2 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("prelu", attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);
  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto alpha = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());
  string alpha_name = "alpha_" + to_string(buf_idx_);
  CreateHybridConstantTensor(&alpha, alpha_name, wshape, attr->q_params, 1, is_layer_hybrid);
  decl << ", " << alpha_name;

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 2, is_layer_hybrid);
  output2params[output_name] = complete_name;

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("prelu_params", params_name);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  PushDeclLine(t0);

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  params_common_setup(decl, call, "prelu", params_name, attr->layer_name.c_str());
  end_stream(decl, "prelu");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::LeakyRelu(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSILeakyReluAttrs>();
  CHECK(attr);
  double alpha = attr->alpha;
  CHECK(call->args.size() == 1) << "LeakyRelu expects 1 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("leaky_relu", attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);
  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 1, is_layer_hybrid);
  output2params[output_name] = complete_name;

  buf_idx_++;

  int32_t alpha_multiplier;
  int32_t alpha_shift;
  GetMultiplierAndShift(alpha, &alpha_multiplier, &alpha_shift);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  malloc_params("relu_params", params_name);

  t0 << params_name << "->n = " << to_string(attr->alpha);
  PushDeclLine(t0);
  t0 << params_name << "->n_multiplier = " << to_string(alpha_multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->n_shift = " << to_string(alpha_shift);
  PushDeclLine(t0);

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  params_common_setup(decl, call, "leaky_relu", params_name, attr->layer_name.c_str());
  end_stream(decl, "leaky_relu");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Concat(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnConcatenateAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
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
  std::map<int, string> free_tensor;

  for (int i = 0; i < input_num; i++) {
    std::ostringstream mem_stream;
    if (auto sub_input_node = tuple->fields[i].as<tvm::relay::CallNode>()) {
      auto sub_input = GetRealInput(sub_input_node);
      CHECK(sub_input.need_copy == true);
      mem_stream << input_name << "[" << i << "] = " << sub_input.name << ";";
      free_tensor[i] = sub_input.name;
    } else if (auto sub_input_var_node = tuple->fields[i].as<tvm::relay::VarNode>()) {
      string var_name = InputTensorVar(sub_input_var_node, i, q_params[i], cfg->dtype_weight);
      mem_stream << input_name << "[" << i << "] = " << var_name << ";";
    } else if (auto sub_input_item_node = tuple->fields[i].as<tvm::relay::TupleGetItemNode>()) {
      string item_name = InputTensorTupleItem(sub_input_item_node, q_params[i], cfg->dtype_weight);
      mem_stream << input_name << "[" << i << "] = " << item_name << ";";
      free_tensor[i] = item_name;
    } else {
      auto sub_input_const_node = tuple->fields[i].as<tvm::relay::ConstantNode>();
      CHECK(sub_input_const_node);
      CSIConstant* const_out = new CSIConstant();
      const_out->name = "constant_" + to_string(const_idx_++);
      const_out->dtype = GetDtypeString(sub_input_const_node->data.DataType());
      const_out->size = sub_input_const_node->data.Length();
      const_out->data_buf = reinterpret_cast<float*>(malloc(const_out->size));
      sub_input_const_node->data.CopyToBytes(const_out->data_buf, const_out->size);
      auto const_name = const_out->name + "_" + to_string(i);
      auto const_shape = GetShape(sub_input_const_node->checked_type());
      CreateConstantTensor(const_out, const_name, const_shape, cfg->dtype_weight, q_params[i]);
      mem_stream << input_name << "[" << i << "] = " << const_name << ";";
    }
    ext_func_body.push_back(mem_stream.str());
  }
  decl << input_name;

  /* Emit output tensor */
  string output_name =
      OutputTensor(decl, call, q_params[attr->q_params.size() - 1], cfg->dtype_weight);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("concat_params", params_name);

  t0 << params_name << "->inputs_count = " << to_string(input_num);
  PushDeclLine(t0);
  t0 << params_name << "->axis = " << to_string(attr->axis);
  PushDeclLine(t0);
  PushOutput(output_name, call);

  params_common_setup(decl, call, "concat", params_name, attr->layer_name.c_str());
  end_stream(decl, "concat");
  for (auto iter = free_tensor.begin(); iter != free_tensor.end(); iter++) {
    FreeTensor(tuple->fields[iter->first], iter->second);
  }
}

void CodegenCSINN::LRN(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSILRNAttrs>();
  SisoOp<QnnCSILRNAttrs>(decl, call, attr, "lrn");
  int32_t multiplier, shift;

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  malloc_params("lrn_params", params_name);

  /* range */
  t0 << params_name << "->range = " << to_string(attr->size);
  PushDeclLine(t0);
  t0 << params_name << "->norm_region = CSINN_LRN_" << attr->norm_region;
  PushDeclLine(t0);
  /* bias */
  GetMultiplierAndShift(attr->bias, &multiplier, &shift);
  t0 << params_name << "->bias = " << to_string(attr->bias);
  PushDeclLine(t0);
  t0 << params_name << "->bias_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->bias_shift = " << to_string(shift);
  PushDeclLine(t0);

  /* alpha */
  GetMultiplierAndShift(attr->alpha, &multiplier, &shift);
  t0 << params_name << "->alpha = " << to_string(attr->alpha);
  PushDeclLine(t0);
  t0 << params_name << "->alpha_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->alpha_shift = " << to_string(shift);
  PushDeclLine(t0);

  /* beta */
  GetMultiplierAndShift(attr->beta, &multiplier, &shift);
  t0 << params_name << "->beta = " << to_string(attr->beta);
  PushDeclLine(t0);
  t0 << params_name << "->beta_multiplier = " << to_string(multiplier);
  PushDeclLine(t0);
  t0 << params_name << "->beta_shift = " << to_string(shift);
  PushDeclLine(t0);

  params_common_setup(decl, call, "lrn", params_name, attr->layer_name.c_str());
  end_stream(decl, "lrn");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Flatten(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  SisoOp<QnnCSIUnaryAttrs>(decl, call, attr, "flatten");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("flatten_params", params_name);

  params_common_setup(decl, call, "flatten", params_name, attr->layer_name.c_str());
  end_stream(decl, "flatten");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Sigmoid(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  SisoOp<QnnCSIUnaryAttrs>(decl, call, attr, "sigmoid");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("sigmoid_params", params_name);

  params_common_setup(decl, call, "sigmoid", params_name, attr->layer_name.c_str());
  end_stream(decl, "sigmoid");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Transpose(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSITransposeAttrs>();
  CHECK(attrs);

  QuantParams* q_params = GetQuantParams(attrs->q_params);
  CHECK(call->args.size() == 1) << "Transpose expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  string perm_name = "permute_" + to_string(buf_idx_);
  int32_t perm_size = attrs->axes.size();

  t0 << "int32_t *" << perm_name << " = malloc(" << perm_size << " * 4)";
  PushDeclLine(t0);
  for (int i = 0; i < perm_size; i++) {
    t0 << perm_name << "[" << i << "] = " << to_string(attrs->axes[i].as<IntImmNode>()->value);
    PushDeclLine(t0);
  }

  string output_name = OutputTensor(decl, call, q_params[1], cfg->dtype_weight);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  malloc_params("transpose_params", params_name);
  t0 << params_name << "->permute = " << perm_name;
  PushDeclLine(t0);
  t0 << params_name << "->permute_num = " << to_string(perm_size);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "transpose", params_name, attrs->layer_name.c_str());
  end_stream(decl, "transpose");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Reshape(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIReshapeAttrs>();
  SisoOp<QnnCSIReshapeAttrs>(decl, call, attr, "reshape");

  auto out_shape = GetShape(call->checked_type());
  string new_shape_name = "shape_" + to_string(buf_idx_);
  int32_t new_shape_dim_num = out_shape.size();
  t0 << "int32_t *" << new_shape_name << " = malloc(" << new_shape_dim_num << " * 4)";
  PushDeclLine(t0);
  for (int i = 0; i < new_shape_dim_num; i++) {
    t0 << new_shape_name << "[" << i << "] = " << to_string(out_shape[i]);
    PushDeclLine(t0);
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  malloc_params("reshape_params", params_name);

  t0 << params_name << "->shape = " << new_shape_name;
  PushDeclLine(t0);
  t0 << params_name << "->shape_num = " << new_shape_dim_num;
  PushDeclLine(t0);
  params_common_setup(decl, call, "reshape", params_name, attr->layer_name.c_str());

  end_stream(decl, "reshape");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::BroadCastTo(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIBroadCastToAttrs>();
  SisoOp<QnnCSIBroadCastToAttrs>(decl, call, attr, "broadcast_to");

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("broadcast_to_params", params_name);
  t0 << params_name << "->shape = malloc(" << attr->shape.size() << " * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < attr->shape.size(); i++) {
    t0 << params_name << "->shape[" << i
       << "] = " << to_string(attr->shape[i].as<IntImmNode>()->value);
    PushDeclLine(t0);
  }
  t0 << params_name << "->shape_count = " << attr->shape.size();
  PushDeclLine(t0);

  params_common_setup(decl, call, "broadcast_to", params_name, attr->layer_name.c_str());
  end_stream(decl, "broadcast_to");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Squeeze(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSISqueezeAttrs>();
  SisoOp<QnnCSISqueezeAttrs>(decl, call, attr, "squeeze");

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
  t0 << params_name << "->axis = " << squeeze_axis_name;
  PushDeclLine(t0);
  t0 << params_name << "->axis_num = " << squeeze_axis_dim_num;
  PushDeclLine(t0);
  params_common_setup(decl, call, "squeeze", params_name, attr->layer_name.c_str());
  end_stream(decl, "squeeze");
  FreeTensor(call->args[0], siso_input_name);
}

void CodegenCSINN::Segment(const CallNode* call, string name) {
  const auto* attr = call->attrs.as<QnnCSISegmentAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  std::ostringstream decl;
  std::ostringstream t0;

  CHECK(call->args.size() == 2) << "op expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("segment_" + name, attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);
  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  /* Emit idx tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto idx = constant_[0];
  auto ishape = GetShape(call->args[1]->checked_type());
  string idx_name = "idx_" + to_string(buf_idx_);
  CreateConstantTensor(&idx, idx_name, ishape, "int32_t", q_params[1]);
  decl << ", " << idx_name;

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attr->q_params, 2, is_layer_hybrid);
  output2params[output_name] = complete_name;

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("segment_params", params_name);

  params_common_setup(decl, call, "segment_" + name, params_name, attr->layer_name.c_str());
  end_stream(decl, "segment_" + name);
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::ScatterND(const CallNode* call) {
  const auto* attr = call->attrs.as<QnnCSIUnaryAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  std::ostringstream decl;
  std::ostringstream t0;

  CHECK(call->args.size() == 3) << "op expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit idx tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto idx = constant_[0];
  auto ishape = GetShape(call->args[1]->checked_type());
  string idx_name = "idx_" + to_string(buf_idx_);
  CreateConstantTensor(&idx, idx_name, ishape, "int32_t", q_params[1]);
  decl << ", " << idx_name << ", ";

  VisitExpr(call->args[2]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string updates_name = InputTensor(decl, call, 2, q_params[2], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  PushOutput(output_name, call);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("scatter_nd_params", params_name);

  params_common_setup(decl, call, "scatter_nd", params_name, attr->layer_name.c_str());
  end_stream(decl, "scatter_nd");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Reduce(const CallNode* call, string name, string out_dtype) {
  std::ostringstream decl;
  std::ostringstream t0;

  const auto* attr = call->attrs.as<QnnCSIReduceAttrs>();
  CHECK(attr);
  // x86 reference
  auto axis = attr->axis;

  auto input_shape = GetShape(call->args[0]->checked_type());

  CHECK(call->args.size() == 1) << name << " expects 1 args";
  auto out_shape = GetShape(call->checked_type());
  VisitExpr(call->args[0]);

  string complete_name = get_complete_layer_name(name, attr->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);

  CHECK(out_.size() == 1) << "Every args expects a single out_";
  decl << "(";
  // string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  string input_name = HybridInputTensor(decl, call, 0, attr->q_params, 0, is_layer_hybrid);

  string output_name = HybridOutputTensor(decl, call, attr->q_params, 1, is_layer_hybrid);

  output2params[output_name] = complete_name;

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

  t0 << "int32_t *out_strides_" << buf_idx_ << " = malloc(" << out_strides.size() << " * 4)";
  PushDeclLine(t0);
  t0 << "int32_t *out_extents_" << buf_idx_ << " = malloc(" << out_extents.size() << " * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < out_strides.size(); i++) {
    t0 << "out_strides_" << buf_idx_ << "[" << i << "] = " << to_string(out_strides[i]);
    PushDeclLine(t0);
  }
  for (uint i = 0; i < out_extents.size(); i++) {
    t0 << "out_extents_" << buf_idx_ << "[" << i << "] = " << to_string(out_extents[i]);
    PushDeclLine(t0);
  }

  t0 << "int32_t *inner_strides_" << buf_idx_ << " = malloc(" << inner_strides.size() << " * 4)";
  PushDeclLine(t0);
  t0 << "int32_t *inner_extents_" << buf_idx_ << " = malloc(" << inner_extents.size() << " * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < inner_strides.size(); i++) {
    t0 << "inner_strides_" << buf_idx_ << "[" << i << "] = " << to_string(inner_strides[i]);
    PushDeclLine(t0);
  }
  for (uint i = 0; i < inner_extents.size(); i++) {
    t0 << "inner_extents_" << buf_idx_ << "[" << i << "] = " << to_string(inner_extents[i]);
    PushDeclLine(t0);
  }

  t0 << "int32_t *aixs_" << buf_idx_ << " = malloc(" << axis.size() << " * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < axis.size(); i++) {
    t0 << "aixs_" << buf_idx_ << "[" << i << "] = " << to_string(axis[i].as<IntImmNode>()->value);
    PushDeclLine(t0);
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("reduce_params", params_name);
  t0 << params_name << "->out_strides = out_strides_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->out_extents = out_extents_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->n = " << to_string(out_extents.size());
  PushDeclLine(t0);
  t0 << params_name << "->inner_strides = inner_strides_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->inner_extents = inner_extents_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->m = " << to_string(inner_extents.size());
  PushDeclLine(t0);
  t0 << params_name << "->axis = aixs_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->axis_count = " << axis.size();
  PushDeclLine(t0);
  if (attr->keepdims) {
    t0 << params_name << "->keepdims = true";
  }
  PushDeclLine(t0);
  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  params_common_setup(decl, call, name, params_name, attr->layer_name.c_str());
  end_stream(decl, name);
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::CropResize(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSICropResizeAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  auto crop_size = attr->crop_size;
  auto method = attr->method;
  auto extrapolation_value = attr->extrapolation_value;
  CHECK(call->args.size() == 3) << "CropResize expects 3 args";
  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit boxes tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto boxes = constant_[0];
  auto bshape = GetShape(call->args[1]->checked_type());
  string boxes_name = "boxes_" + to_string(buf_idx_);
  CreateConstantTensor(&boxes, boxes_name, bshape, "int32_t", q_params[1]);
  decl << ", " << boxes_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto box_indices = constant_[0];
  auto index_shape = GetShape(call->args[2]->checked_type());
  string index_name = "index_" + to_string(buf_idx_);
  CreateConstantTensor(&box_indices, index_name, index_shape, "int32_t", q_params[2]);
  decl << ", " << index_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  t0 << "  int32_t crop_size_" << buf_idx_ << "[" << crop_size.size() << "] = {";
  for (uint i = 0; i < crop_size.size(); i++) {
    t0 << Downcast<IndexExpr>(crop_size[i]) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);
  malloc_params("crop_resize_params", params_name);

  t0 << params_name << "->method = " << method;
  PushDeclLine(t0);
  t0 << params_name << "->extrapolation_value = " << extrapolation_value;
  PushDeclLine(t0);
  t0 << params_name << "->crop_size = crop_size_" << buf_idx_;
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "crop_resize", params_name, attr->layer_name.c_str());
  end_stream(decl, "crop_resize");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::StridedSlice(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0, t1;
  const auto* attr = call->attrs.as<QnnCSIStridedSliceAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  // x86 reference
  auto begin = attr->begin;
  auto end = attr->end;
  auto strides = attr->strides;

  CHECK(call->args.size() == 1) << "strided slic expects 1 args";
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  decl << "(";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  string output_name = OutputTensor(decl, call, q_params[1], cfg->dtype_weight);

  t0 << "int32_t *begin_" << buf_idx_ << " = malloc(" << begin.size() << " * 4)";
  PushDeclLine(t0);
  t0 << "int32_t *end_" << buf_idx_ << " = malloc(" << end.size() << " * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < begin.size(); i++) {
    t0 << "begin_" << buf_idx_ << "[" << i << "] = " << to_string(begin[i]);
    PushDeclLine(t0);
  }
  auto ishape = GetShape(call->args[0]->checked_type());
  for (uint i = 0; i < end.size(); i++) {
    int end_ =
        end[i].as<IntImmNode>()->value > ishape[i] ? ishape[i] : end[i].as<IntImmNode>()->value;
    t0 << "end_" << buf_idx_ << "[" << i << "] = " << to_string(end_);
    PushDeclLine(t0);
  }

  uint stride_size = strides.size();
  if (stride_size == 1) {
    stride_size = begin.size();
  }

  t0 << "int32_t *strides_" << buf_idx_ << " = malloc(" << stride_size << " * 4)";
  PushDeclLine(t0);

  for (uint i = 0; i < stride_size; i++) {
    if (i < strides.size()) {
      t0 << "strides_" << buf_idx_ << "[" << i << "] = " << to_string(strides[i]);
      PushDeclLine(t0);
    } else {
      t0 << "strides_" << buf_idx_ << "[" << i << "] = " << to_string(strides[0]);
      PushDeclLine(t0);
    }
  }

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  malloc_params("strided_slice_params", params_name);
  t0 << params_name << "->begin = begin_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->end = end_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->stride = strides_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->slice_count = " << begin.size();
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "strided_slice", params_name, attr->layer_name.c_str());
  end_stream(decl, "strided_slice");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::Split(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;

  const auto* attr = call->attrs.as<QnnCSISplitAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);
  // x86 reference
  auto axis = attr->axis;

  CHECK(call->args.size() == 1) << "strided slic expects 1 args";
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";

  string out_name = "output_" + to_string(buf_idx_);
  t0 << "struct csi_tensor *" << out_name << "[" << attr->q_params.size() - 1 << "]";
  PushDeclLine(t0);
  decl << "(";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);
  auto in_shape = GetShape(call->args[0]->checked_type());

  decl << ", " << out_name << ", ";
  string params_name = "params_" + to_string(buf_idx_);
  decl << params_name << ")";

  std::vector<string> out_names;
  for (uint i = 0; i < attr->q_params.size() - 1; i++) {
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
    if (cfg->dtype_weight == "float16" || cfg->dtype_weight == "bfloat16" ||
        cfg->dtype_weight == "int16_t") {
      out_size = out_size * 2;
    }
    malloc_buf(out, out_size);
    alloc_idx_++;
    string output_name = "output_" + to_string(buf_idx_) + "_" + to_string(i);
    CreateTensor(output_name, out, out_shape, q_params[i + 1], cfg->dtype_weight);

    t0 << out_name << "[" << to_string(i) << "] = " << output_name;
    PushDeclLine(t0);

    out_names.push_back(output_name);
  }

  Array<Integer> indices_or_sections;
  if (const IntImmNode* sections = attr->indices_or_sections.as<IntImmNode>()) {
    axis = axis == -1 ? in_shape.size() - 1 : axis;
    t0 << "int32_t axis_len" << buf_idx_ << " = ";
    t0 << input_name << "->dim[" << axis << "]";
    PushDeclLine(t0);

    t0 << "int32_t index_" << buf_idx_ << " = ";
    t0 << "axis_len" << buf_idx_ << " / " << sections->value;
    PushDeclLine(t0);

    t0 << "int32_t *indices_or_sections_" << buf_idx_ << " = malloc(sizeof(int32_t) * "
       << sections->value - 1 << ")";
    PushDeclLine(t0);
    for (int x = 1; x < sections->value; x++) {
      t0 << "indices_or_sections_" << buf_idx_ << "[" << x - 1 << "] = index_" << buf_idx_ << " * "
         << x;
      PushDeclLine(t0);
    }
  } else {
    auto indices_ = Downcast<Array<ObjectRef>>(attr->indices_or_sections);
    t0 << "int32_t *indices_or_sections_" << buf_idx_ << " = malloc(sizeof(int32_t) * "
       << indices_.size() << ")";
    PushDeclLine(t0);
    for (uint32_t k = 0; k < indices_.size(); k++) {
      auto idx = Downcast<IndexExpr>(indices_[k]);
      t0 << "indices_or_sections_" << buf_idx_ << "[" << k
         << "] = " << to_string(*tir::as_const_int(idx)) << ";";
      PushDeclLine(t0);
    }
  }

  malloc_params("split_params", params_name);
  t0 << params_name << "->split_index = indices_or_sections_" << buf_idx_;
  PushDeclLine(t0);
  t0 << params_name << "->output_num = " << attr->q_params.size() - 1;
  PushDeclLine(t0);
  t0 << params_name << "->axis = " << to_string(axis);
  PushDeclLine(t0);

  PushOutput(out_names, call);
  params_common_setup(decl, call, "split", params_name, attr->layer_name.c_str());
  end_stream(decl, "split");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::BatchToSpaceND(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSIBatchToSpaceNDAttrs>();
  CHECK(attrs);

  CHECK(call->args.size() == 1) << "BatchToSpaceND expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);

  string complete_name = get_complete_layer_name("batch_to_space_nd", attrs->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);

  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = HybridInputTensor(decl, call, 0, attrs->q_params, 0, is_layer_hybrid);

  string block_shape_name = "block_shape_" + to_string(buf_idx_);
  string crops_name = "crops_" + to_string(buf_idx_);
  int32_t spatial_dim_cnt = attrs->block_shape.size();

  // Emit block shape
  t0 << "int32_t " << block_shape_name << "[" << spatial_dim_cnt << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->block_shape[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  // Emit crops
  t0 << "int32_t " << crops_name << "[" << spatial_dim_cnt * 2 << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->crops[i][0].as<IntImmNode>()->value) << ", ";
    t0 << to_string(attrs->crops[i][1].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string output_name = HybridOutputTensor(decl, call, attrs->q_params, 1, is_layer_hybrid);
  string params_name = "params_" + to_string(buf_idx_);
  output2params[output_name] = complete_name;
  decl << ", " << params_name << ")";
  malloc_params("batch_to_space_nd_params", params_name);

  t0 << params_name << "->crops = " << crops_name;
  PushDeclLine(t0);
  t0 << params_name << "->block_shape = " << block_shape_name;
  PushDeclLine(t0);
  t0 << params_name << "->spatial_dim_cnt = " << to_string(spatial_dim_cnt);
  PushDeclLine(t0);

  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }
  params_common_setup(decl, call, "batch_to_space_nd", params_name, attrs->layer_name.c_str());
  end_stream(decl, "batch_to_space_nd");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::SpaceToBatchND(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream pstream;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSISpaceToBatchNDAttrs>();
  CHECK(attrs);

  QuantParams* q_params = GetQuantParams(attrs->q_params);
  CHECK(call->args.size() == 1) << "SpaceToBatchND expects 1 args";

  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  string block_shape_name = "block_shape_" + to_string(buf_idx_);
  string paddings_name = "paddings_" + to_string(buf_idx_);
  int32_t spatial_dim_cnt = attrs->block_shape.size();

  // Emit block shape
  t0 << "int32_t " << block_shape_name << "[" << spatial_dim_cnt << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->block_shape[i].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  // Emit paddings
  t0 << "int32_t " << paddings_name << "[" << spatial_dim_cnt * 2 << "] = {";
  for (int i = 0; i < spatial_dim_cnt; i++) {
    t0 << to_string(attrs->paddings[i][0].as<IntImmNode>()->value) << ", ";
    t0 << to_string(attrs->paddings[i][1].as<IntImmNode>()->value) << ", ";
  }
  t0 << "}";
  PushDeclLine(t0);

  string output_name = OutputTensor(decl, call, q_params[1], cfg->dtype_weight);
  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";
  malloc_params("space_to_batch_nd_params", params_name);

  t0 << params_name << "->paddings = " << paddings_name;
  PushDeclLine(t0);
  t0 << params_name << "->block_shape = " << block_shape_name;
  PushDeclLine(t0);
  t0 << params_name << "->spatial_dim_cnt = " << to_string(spatial_dim_cnt);
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "space_to_batch_nd", params_name, attrs->layer_name.c_str());
  end_stream(decl, "space_to_batch_nd");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::MatMul(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSIMatMulAttrs>();
  CHECK(attrs);
  std::map<int, string> free_tensor;

  CHECK(call->args.size() == 3) << "Dense expects 2 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string complete_name = get_complete_layer_name("matmul", attrs->layer_name.c_str());
  bool is_layer_hybrid = is_contain_item<string>(hybrid_layer_name, complete_name);
  string input1_name = HybridInputTensor(decl, call, 0, attrs->q_params, 0, is_layer_hybrid);
  free_tensor[0] = input1_name;
  buf_idx_++;
  decl << ", ";
  string input2_name;

  /* Emit input tensor */
  VisitExpr(call->args[1]);
  if (call->args[1].as<tvm::relay::CallNode>() || call->args[1].as<tvm::relay::VarNode>() ||
      call->args[0].as<tvm::relay::TupleGetItemNode>()) {
    CHECK(out_.size() == 1) << "Every args expects a single out_";
    input2_name = HybridInputTensor(decl, call, 1, attrs->q_params, 1, is_layer_hybrid);
    free_tensor[1] = input2_name;
  } else {
    // add constant arg
    CHECK(constant_.size() == 1) << "Every args expects a single constant_";
    auto data_b = constant_[0];
    auto b_shape = GetShape(call->args[1]->checked_type());
    input2_name = "data_b_" + to_string(buf_idx_);
    CreateHybridConstantTensor(&data_b, input2_name, b_shape, attrs->q_params, 1, is_layer_hybrid);
    decl << input2_name;
  }

  // /* Emit bias tensor */
  // VisitExpr(call->args[2]);
  // CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  // auto bias = constant_[0];
  // auto bshape = GetShape(call->args[2]->checked_type());
  // string bias_name = "bias_" + to_string(buf_idx_);
  // CreateConstantTensor(&bias, bias_name, bshape, cfg->dtype_activation, q_params[0], q_params[1],
  //                      q_params[2]);
  // decl << ", " << bias_name;

  /* Emit output tensor */
  string output_name = HybridOutputTensor(decl, call, attrs->q_params, 2, is_layer_hybrid);
  output2params[output_name] = complete_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  malloc_params("matmul_params", params_name);
  string transpose_a = attrs->transpose_a ? "true" : "false";
  string transpose_b = attrs->transpose_b ? "true" : "false";
  t0 << params_name << "->trans_a = " << transpose_a;
  PushDeclLine(t0);
  t0 << params_name << "->trans_b = " << transpose_b;
  PushDeclLine(t0);
  if (is_layer_hybrid) {
    PushOutput(output_name, call, hybrid_cfg->dtype_weight);
  } else {
    PushOutput(output_name, call, cfg->dtype_weight);
  }

  params_common_setup(decl, call, "matmul", params_name, attrs->layer_name.c_str());
  end_stream(decl, "matmul");
  for (auto iter = free_tensor.begin(); iter != free_tensor.end(); iter++) {
    FreeTensor(call->args[iter->first], iter->second);
  }
}

void CodegenCSINN::CacheMatMul(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSICacheMatMulAttrs>();
  CHECK(attrs);
  QuantParams* q_params = GetQuantParams(attrs->q_params);

  CHECK(call->args.size() == 3) << "CacheMatMul expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit weight tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto weight = constant_[0];
  auto weight_shape = GetShape(call->args[1]->checked_type());

  string weight_name = "weight_" + to_string(buf_idx_);
  CreateConstantTensor(&weight, weight_name, weight_shape, cfg->dtype_weight, q_params[1]);

  decl << ", " << weight_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(&bias, bias_name, bshape, cfg->dtype_activation, q_params[0], q_params[1],
                       q_params[2]);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  malloc_params("cache_matmul_params", params_name);

  string cache_shape_name = "cache_shape_" + to_string(buf_idx_);
  t0 << "int32_t *" << cache_shape_name << " = malloc(" << attrs->cache_shape.size() << " * 4)";

  PushDeclLine(t0);
  for (uint i = 0; i < attrs->cache_shape.size(); i++) {
    t0 << cache_shape_name << "[" << i << "] = " << to_string(attrs->cache_shape[i]);
    PushDeclLine(t0);
  }

  string shape_name = "shape_" + to_string(buf_idx_);
  t0 << "int32_t *" << shape_name << " = malloc(" << attrs->shape.size() << " * 4)";

  PushDeclLine(t0);
  for (uint i = 0; i < attrs->shape.size(); i++) {
    t0 << shape_name << "[" << i << "] = " << to_string(attrs->shape[i]);
    PushDeclLine(t0);
  }

  string axes_name = "axes_" + to_string(buf_idx_);
  t0 << "int32_t *" << axes_name << " = malloc(" << attrs->axes.size() << " * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < attrs->axes.size(); i++) {
    t0 << axes_name << "[" << i << "] = " << to_string(attrs->axes[i]);
    PushDeclLine(t0);
  }

  t0 << params_name << "->cache_shape = " << cache_shape_name;
  PushDeclLine(t0);
  t0 << params_name << "->shape = " << shape_name;
  PushDeclLine(t0);
  t0 << params_name << "->axes = " << axes_name;
  PushDeclLine(t0);
  PushOutput(output_name, call);

  params_common_setup(decl, call, "cache_matmul", params_name, attrs->layer_name.c_str());
  end_stream(decl, "cache_matmul");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::CacheConv1d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSICacheConv1DAttrs>();
  CHECK(attr);
  QuantParams* q_params = GetQuantParams(attr->q_params);

  CHECK(call->args.size() == 3) << "Conv1d expects 3 args";

  /* Make function call with arguments start */
  decl << "(";

  /* Emit_ input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit kernel tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto kernel = constant_[0];
  auto wshape = GetShape(call->args[1]->checked_type());
  string kernel_name = "kernel_" + to_string(buf_idx_);
  CreateConstantTensor(&kernel, kernel_name, wshape, cfg->dtype_weight, q_params[1]);
  decl << ", " << kernel_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto bias = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string bias_name = "bias_" + to_string(buf_idx_);
  CreateConstantTensor(&bias, bias_name, bshape, cfg->dtype_activation, q_params[0], q_params[1],
                       q_params[2]);

  decl << ", " << bias_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";

  malloc_params("cache_conv1d_params", params_name);
  string shape_name = "cache_shape_" + to_string(buf_idx_);
  t0 << "int32_t *" << shape_name << " = malloc(" << attr->cache_shape.size() << " * 4)";

  PushDeclLine(t0);
  for (uint i = 0; i < attr->cache_shape.size(); i++) {
    t0 << shape_name << "[" << i << "] = " << to_string(attr->cache_shape[i]);
    PushDeclLine(t0);
  }

  t0 << params_name << "->cache_shape = " << shape_name;
  PushDeclLine(t0);

  t0 << params_name << "->group = " << to_string(attr->groups);
  PushDeclLine(t0);
  Array<IndexExpr> strides = attr->strides;
  t0 << params_name << "->stride_width = " << to_string(strides[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Array<IndexExpr> dilation = attr->dilation;
  t0 << params_name << "->dilation_width = " << to_string(dilation[0].as<IntImmNode>()->value);
  PushDeclLine(t0);
  Setup1dPadding<QnnCSICacheConv1DAttrs>(params_name, attr);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "cache_conv1d", params_name, attr->layer_name.c_str());
  end_stream(decl, "cache_conv1d");
  FreeTensor(call->args[0], input_name);
}

void CodegenCSINN::LayerNorm(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attrs = call->attrs.as<QnnCSILayerNormAttrs>();
  CHECK(attrs);
  QuantParams* q_params = GetQuantParams(attrs->q_params);

  CHECK(call->args.size() == 3) << "LayerNorm expects 3 args";

  // Make function call with input buffers when visiting arguments
  decl << "(";

  /* Emit input tensor */
  VisitExpr(call->args[0]);
  CHECK(out_.size() == 1) << "Every args expects a single out_";
  string input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  /* Emit output tensor */
  string output_name = OutputTensor(decl, call, q_params[3], cfg->dtype_weight);

  /* Emit gamma tensor */
  VisitExpr(call->args[1]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto gamma = constant_[0];
  auto gamma_shape = GetShape(call->args[1]->checked_type());

  string gamma_name = "gamma_" + to_string(buf_idx_);
  CreateConstantTensor(&gamma, gamma_name, gamma_shape, cfg->dtype_weight, q_params[1]);

  decl << ", " << gamma_name;

  /* Emit bias tensor */
  VisitExpr(call->args[2]);
  CHECK(constant_.size() == 1) << "Every args expects a single constant_";
  auto beta = constant_[0];
  auto bshape = GetShape(call->args[2]->checked_type());
  string beta_name = "beta_" + to_string(buf_idx_);
  CreateConstantTensor(&beta, beta_name, bshape, cfg->dtype_weight, q_params[2]);

  decl << ", " << beta_name;

  string params_name = "params_" + to_string(buf_idx_);

  decl << ", " << params_name << ")";
  malloc_params("layer_norm_params", params_name);

  t0 << params_name << "->epsilon = " << attrs->epsilon;
  PushDeclLine(t0);
  t0 << params_name << "->axis = " << attrs->axis;
  PushDeclLine(t0);
  string center = attrs->center ? "true" : "false";
  t0 << params_name << "->center = " << center;
  PushDeclLine(t0);
  string scale = attrs->scale ? "true" : "false";
  t0 << params_name << "->scale = " << scale;
  PushDeclLine(t0);
  PushOutput(output_name, call);

  params_common_setup(decl, call, "layer_norm", params_name, attrs->layer_name.c_str());
  end_stream(decl, "layer_norm");
  FreeTensor(call->args[0], input_name);
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
