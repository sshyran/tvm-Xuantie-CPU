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
 * \file src/relay/qnn/op/mul.cc
 * \brief QNN mul operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {
TVM_REGISTER_NODE_TYPE(QnnCSIUnaryAttrs);

bool QnnCSIReluRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], types[0]);
  }
  return true;
}

// QNN Relu operator.
Expr MakeQnnCSIRelu(Expr data, double input_scale, int32_t input_zero_point, double output_scale,
                    int32_t output_zero_point, DataType out_dtype, Array<IndexExpr> max_values,
                    Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.relu");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.relu")
    .describe(R"code(Returns the relu input array, computed element-wise.

.. math::
   max(x, 0)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIReluRel", QnnCSIReluRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIRelu").set_body_typed(MakeQnnCSIRelu);

// QNN Relu operator.
Expr MakeQnnCSIRelu6(Expr data, double input_scale, int32_t input_zero_point, double output_scale,
                     int32_t output_zero_point, DataType out_dtype, Array<IndexExpr> max_values,
                     Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSIUnaryAttrs>();
  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.relu6");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.relu6")
    .describe(R"code(Returns the relu input array, computed element-wise.

.. math::
   min(max(x, 0),6)

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIUnaryAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIReluRel", QnnCSIReluRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIRelu6").set_body_typed(MakeQnnCSIRelu6);

TVM_REGISTER_NODE_TYPE(QnnCSIPReluAttrs);
bool QnnCSIPReluRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;

  const QnnCSIPReluAttrs* param = attrs.as<QnnCSIPReluAttrs>();
  CHECK(param != nullptr);

  CHECK(param->axis < static_cast<int>(data->shape.size()))
      << "Wrong axis (" << param->axis << ")value.";

  // assign alpha type
  Array<IndexExpr> alpha_shape({data->shape[param->axis]});
  reporter->Assign(types[1], TensorType(alpha_shape, data->dtype));

  // assign output type
  reporter->Assign(types[2], TensorType(data->shape, data->dtype));
  return true;
}

// QNN PRelu operator.
Expr MakeQnnCSIPRelu(Expr data, Expr alpha, int axis, double input_scale, int32_t input_zero_point,
                     double alpha_scale, int32_t alpha_zero_point, double output_scale,
                     int32_t output_zero_point, DataType out_dtype, Array<IndexExpr> max_values,
                     Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSIPReluAttrs>();
  attrs->axis = axis;
  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  attrs->alpha_scale = alpha_scale;
  attrs->alpha_zero_point = alpha_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.prelu");
  return Call(op, {data, alpha}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.prelu")
    .describe(R"code(Returns the relu input array, computed element-wise.

.. math::
    x >= 0? x: alpha * x

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIPReluAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .add_argument("alpha", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIPReluRel", QnnCSIPReluRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIPRelu").set_body_typed(MakeQnnCSIPRelu);

TVM_REGISTER_NODE_TYPE(QnnCSILeakyReluAttrs);
// QNN Relu operator.
Expr MakeQnnCSILeakyRelu(Expr data, double alpha, double input_scale, int32_t input_zero_point,
                         double output_scale, int32_t output_zero_point, DataType out_dtype,
                         Array<IndexExpr> max_values, Array<IndexExpr> min_values,
                         String layer_name) {
  auto attrs = make_object<QnnCSILeakyReluAttrs>();

  attrs->alpha = alpha;
  attrs->input_scale = input_scale;
  attrs->input_zero_point = input_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.leaky_relu");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.leaky_relu")
    .describe(R"code(Returns the relu input array, computed element-wise.

.. math::
    x >= 0? x: alpha * x

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSILeakyReluAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIReluRel", QnnCSIReluRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSILeakyRelu").set_body_typed(MakeQnnCSILeakyRelu);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
