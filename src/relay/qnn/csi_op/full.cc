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
TVM_REGISTER_NODE_TYPE(QnnCSIFullAttrs);

bool QnnCSIFullRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                   const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const QnnCSIFullAttrs* param = attrs.as<QnnCSIFullAttrs>();
  const auto* fill_value = types[0].as<TensorTypeNode>();
  const auto* fill_shape = types[1].as<TensorTypeNode>();
  if (fill_value == nullptr) {
    return false;
  }

  CHECK_EQ(fill_value->shape.size(), 0)
      << "Fill value should be a scalar but has dimension " << fill_value->shape.size() << ".";

  const IntImmNode* shape_shape = fill_shape->shape[0].as<IntImmNode>();
  CHECK(shape_shape) << "Parameter shape must have static shape";

  std::vector<IndexExpr> oshape;
  if (param->shape.size()) {
    const Array<Integer>& cshape_array = param->shape;
    for (size_t i = 0; i < cshape_array.size(); ++i) {
      oshape.push_back(cshape_array[i]);
    }
  } else {
    for (int i = 0; i < shape_shape->value; ++i) {
      oshape.push_back(Any());
    }
  }
  reporter->Assign(types[2], TensorType(oshape, DataType::Int(32)));
  return true;
}
// QNN Multiplication operator.
Expr MakeQnnCSIFull(Expr fill_value, Expr shape, double input_scale, int32_t input_zero_point,
                    double output_scale, int32_t output_zero_point, DataType out_dtype,
                    Array<IndexExpr> max_values, Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSIFullAttrs>();
  if (const auto* cshape = shape.as<ConstantNode>()) {
    attrs->shape = ToVector(cshape->data);
  }
  attrs->input_scale = std::move(input_scale);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_scale = std::move(output_scale);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);

  static const Op& op = Op::Get("qnn.csi.full");
  return Call(op, {fill_value, shape}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.full")
    .describe(R"code(Full the data with target shape.

)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIFullAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIFullRel", QnnCSIFullRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIFull").set_body_typed(MakeQnnCSIFull);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
