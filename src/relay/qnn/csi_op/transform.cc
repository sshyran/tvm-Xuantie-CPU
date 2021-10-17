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
 * \file src/relay/qnn/op/transform.cc
 * \brief QNN Squeeze operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {
TVM_REGISTER_NODE_TYPE(QnnCSISqueezeAttrs);

bool QnnCSISqueezeRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                      const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) {
    return false;
  }
  const auto* param = attrs.as<QnnCSISqueezeAttrs>();
  CHECK(param != nullptr);
  std::vector<IndexExpr> result_shape;
  // if axes is None, squeeze all axes of dimension 1
  if (!param->axis.defined()) {
    for (const auto& e : data->shape) {
      const int64_t* axis_ptr = tir::as_const_int(e);
      CHECK(axis_ptr != nullptr) << "the axes attribute must be concrete";
      if (*axis_ptr != 1) {
        result_shape.push_back(e);
      }
    }
  } else {
    // pair up original shape with a boolean which control whether it will be in the final shape.
    std::vector<std::pair<IndexExpr, bool> > original_shape;
    for (const auto& e : data->shape) {
      original_shape.push_back(std::pair<IndexExpr, bool>(e, true));
    }
    for (const auto& e : param->axis) {
      int64_t axis_val = e->value;
      if (axis_val < 0) {
        axis_val += static_cast<int64_t>(original_shape.size());
      }
      CHECK_GE(axis_val, 0);
      CHECK_LT(axis_val, original_shape.size());
      original_shape.at(axis_val).second = false;
    }
    for (const auto p : original_shape) {
      if (p.second) {
        result_shape.push_back(p.first);
      } else {
        const int64_t* axis_ptr = tir::as_const_int(p.first);
        CHECK(axis_ptr != nullptr) << "cannot get concrete shape of input tensor";
        CHECK_EQ(*axis_ptr, 1) << "cannot squeeze axis with dimension not equal to 1";
      }
    }
  }
  reporter->Assign(types[1], TensorType(result_shape, data->dtype));
  return true;
}

// QNN Squeeze operator.
Expr MakeQnnCSISqueeze(Expr data, Array<Integer> axis, double input_scale, int32_t input_zero_point,
                       double output_scale, int32_t output_zero_point, DataType out_dtype,
                       Array<IndexExpr> max_values, Array<IndexExpr> min_values,
                       String layer_name) {
  auto attrs = make_object<QnnCSISqueezeAttrs>();
  attrs->axis = std::move(axis);

  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.squeeze");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.squeeze")
    .set_attrs_type<QnnCSISqueezeAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSISqueezeRel", QnnCSISqueezeRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSISqueeze").set_body_typed(MakeQnnCSISqueeze);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
