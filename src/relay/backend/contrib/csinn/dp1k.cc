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
 * \file src/relay/backend/contrib/csinn/DP1K.cc
 * \brief Implementation of CSINN DP1K codegen APIs.
 */

#include "dp1k.h"

#include <float.h>

#include <string>
#include <vector>

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

string CodegenDP1K::EmitGraph(void) {
  EmitVersion();
  EmitHeader();
  EmitSessionSetup();
  EmitSessionRun();
  DumpConstant();
  return code_stream_.str();
}

void CodegenDP1K::VisitExpr_(const CallNode* call) {
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
  } else if (IsOp(call, "qnn.csi.concatenate")) {
    Concat(call);
  } else if (IsOp(call, "qnn.csi.conv2d")) {
    Conv2d(call, "conv2d");
  } else if (IsOp(call, "qnn.csi.deconv2d")) {
    DeConv2d(call);
  } else if (IsOp(call, "qnn.csi.dense")) {
    Dense(call);
  } else if (IsOp(call, "qnn.csi.global_avgpool2d")) {
    GlobalAvgPool2d(call);
  } else if (IsOp(call, "qnn.csi.leaky_relu")) {
    LeakyRelu(call);
  } else if (IsOp(call, "qnn.csi.maxpool2d")) {
    MaxPool2d(call);
  } else if (IsOp(call, "qnn.csi.mul")) {
    DisoOp(call, "mul");
  } else if (IsOp(call, "qnn.csi.prelu")) {
    PRelu(call);
  } else if (IsOp(call, "qnn.csi.relu")) {
    Relu(call);
  } else if (IsOp(call, "qnn.csi.reshape")) {
    Reshape(call);
  } else if (IsOp(call, "qnn.csi.sigmoid")) {
    Sigmoid(call);
  } else if (IsOp(call, "qnn.csi.softmax")) {
    Softmax(call);
  } else if (IsOp(call, "qnn.csi.strided_slice")) {
    StridedSlice(call);
  } else if (IsOp(call, "qnn.csi.transpose")) {
    Transpose(call);
  } else if (IsOp(call, "qnn.csi.upsampling")) {
    UpSampling(call);
  } else {
    std::cerr << "DP1K unsupported op: " << AsText(call->op, false) << "\n";
    exit(-1);
  }
}

QuantParams* CodegenDP1K::GetQuantParams(Array<Array<IndexExpr>> q_params) {
  int size = q_params.size();
  QuantParams* out_q_params = new QuantParams[size];
  for (int i = 0; i < size; i++) {
    auto q_param = q_params[i];
    if (q_param.size() == 3 || q_param.size() > 4) {
      // flag + single channel == 3
      uint length = q_param.size() / 2;
      out_q_params[i] = *new QuantParams();
      Qinfo* q_infos = new Qinfo[length + 1];
      int32_t flag = q_param[0].as<IntImmNode>()->value;
      float g_max_value = -FLT_MAX;
      float g_min_value = FLT_MAX;
      for (uint j = 1; j < q_param.size(); j = j + 2) {
        int index = (j - 1) / 2 + 1;
        if (flag == USE_MINMAX) {
          float min_value = q_param[j].as<FloatImmNode>()->value;
          float max_value = q_param[j + 1].as<FloatImmNode>()->value;
          if (g_max_value < max_value) {
            g_max_value = max_value;
          }
          if (g_min_value > min_value) {
            g_min_value = min_value;
          }
          QuantParams* tmp = GetQuantParamsBase(min_value, max_value);
          q_infos[index] = *tmp->qinfo;
        } else if (flag == USE_SCALE) {
          std::cerr << "DP1K unsupported flag \n";
          exit(-1);
        }
      }
      QuantParams* g_quant = GetQuantParamsBase(g_min_value, g_max_value);
      q_infos[0] = *g_quant->qinfo;
      out_q_params[i].qinfo = q_infos;
      out_q_params[i].q_size = length + 1;
    }
  }
  return out_q_params;
}

void CodegenDP1K::GlobalAvgPool2d(const CallNode* call) {
  std::ostringstream decl;
  std::ostringstream t0;
  const auto* attr = call->attrs.as<QnnCSIGlobalAvgPoolAttrs>();
  SisoOp<QnnCSIGlobalAvgPoolAttrs>(decl, call, attr);
  auto in_shape = GetShape(call->args[0]->checked_type());
  CHECK_EQ(in_shape.size(), 4);

  string params_name = "params_" + to_string(buf_idx_);
  decl << ", " << params_name << ")";

  QnnCSIAvgPool2DAttrs* new_attr = new QnnCSIAvgPool2DAttrs;
  new_attr->ceil_mode = false;
  new_attr->strides.push_back(1);
  new_attr->strides.push_back(1);
  new_attr->pool_size.push_back(in_shape[2]);
  new_attr->pool_size.push_back(in_shape[3]);
  new_attr->padding.push_back(0);
  new_attr->padding.push_back(0);

  SetupPoolParams<QnnCSIAvgPool2DAttrs>(params_name, new_attr);
  params_common_setup(decl, call, "averagepool", params_name, attr->layer_name.c_str(),
                      "CSINN_LAYOUT_NCHW");
  end_stream(decl, "averagepool");
}

void CodegenDP1K::StridedSlice(const CallNode* call) {
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

  string input_name;
  decl << "(";
  input_name = InputTensor(decl, call, 0, q_params[0], cfg->dtype_weight);

  string output_name = OutputTensor(decl, call, q_params[1], cfg->dtype_weight);
  auto in_shape = GetShape(call->args[0]->checked_type());
  CHECK_EQ(in_shape.size(), 4);
  t0 << "int32_t *begin_" << buf_idx_ << " = malloc(4 * 4)";
  PushDeclLine(t0);
  t0 << "int32_t *end_" << buf_idx_ << " = malloc(4 * 4)";
  PushDeclLine(t0);
  for (uint i = 0; i < begin.size(); i++) {
    t0 << "begin_" << buf_idx_ << "[" << i << "] = " << to_string(begin[i]);
    PushDeclLine(t0);
  }
  for (uint i = begin.size(); i < in_shape.size(); i++) {
    t0 << "begin_" << buf_idx_ << "[" << i << "] = 0";
    PushDeclLine(t0);
  }

  for (uint i = 0; i < end.size(); i++) {
    t0 << "end_" << buf_idx_ << "[" << i << "] = " << to_string(end[i]);
    PushDeclLine(t0);
  }

  for (uint i = end.size(); i < in_shape.size(); i++) {
    t0 << "end_" << buf_idx_ << "[" << i << "] = " << to_string(in_shape[i]);
    PushDeclLine(t0);
  }

  uint stride_size = strides.size();
  if (stride_size == 1) {
    stride_size = begin.size();
  }

  t0 << "int32_t *strides_" << buf_idx_ << " = malloc(4 * 4)";
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

  for (uint i = stride_size; i < in_shape.size(); i++) {
    t0 << "strides_" << buf_idx_ << "[" << i << "] = 1";
    PushDeclLine(t0);
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
  t0 << params_name << "->slice_count = " << in_shape.size();
  PushDeclLine(t0);

  PushOutput(output_name, call);
  params_common_setup(decl, call, "strided_slice", params_name, attr->layer_name.c_str(),
                      "CSINN_LAYOUT_NCHW");
  end_stream(decl, "strided_slice");
}

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
