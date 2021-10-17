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
 * \file src/relay/qnn/op/dense.cc
 * \brief Property def of qnn dense operator.
 */

#include <tvm/relay/base.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../op/nn/nn.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

// relay.op.qnn.dense
TVM_REGISTER_NODE_TYPE(QnnCSIDenseAttrs);

bool QnnCSIDenseRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                    const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* weight = types[1].as<TensorTypeNode>();

  if (data == nullptr) return false;

  const QnnCSIDenseAttrs* param = attrs.as<QnnCSIDenseAttrs>();
  CHECK(param != nullptr);

  CHECK(static_cast<int>(data->shape.size()) != 0);

  Array<tvm::PrimExpr> oshape = data->shape;
  if (param->units.defined()) {
    Array<tvm::PrimExpr> dshape = data->shape;
    // validate the weight shape is proper if defined
    // Assign weight type
    Array<IndexExpr> wshape({param->units, weight->shape[1]});
    auto i_size = dshape[0];
    if (dshape.size() == 4) {
      i_size = dshape[0] * dshape[1] * dshape[2] * dshape[3];
    }
    // It is possible for weight to be nullptr in which case we will use
    // data dtype as the weight dtype. However if weight dtype is explicitly
    // present we will use that.
    auto weight_dtype = (weight == nullptr ? data->dtype : weight->dtype);
    reporter->Assign(types[1], TensorType(wshape, weight_dtype));
    oshape.Set((oshape.size() - 1), param->units);
    if (oshape.size() == 4) {
      Array<tvm::PrimExpr> tmp;
      tmp.push_back(i_size / weight->shape[1]);
      tmp.push_back(param->units);
      oshape = tmp;
    }

    CHECK_EQ(oshape.size(), 2);

  } else {
    if (weight == nullptr) return false;
    Array<tvm::PrimExpr> wshape = weight->shape;
    tvm::PrimExpr n = oshape[0];
    oshape = wshape;
    oshape.Set(0, n);
    oshape.Set((oshape.size() - 1), wshape[0]);
  }

  DataType out_dtype = param->out_dtype;
  if (out_dtype.bits() == 0) {
    out_dtype = data->dtype;
  }
  // assign output type
  reporter->Assign(types[3], TensorType(oshape, out_dtype));
  return true;
}
// Positional relay function to create quantized dense operator used by frontend FFI.
Expr MakeQnnCSIDense(Expr data, Expr weight, Expr bias, double input_scale,
                     int32_t input_zero_point, double kernel_scale, int32_t kernel_zero_point,
                     double output_scale, int32_t output_zero_point, IndexExpr units,
                     DataType out_dtype, Array<IndexExpr> max_values, Array<IndexExpr> min_values,
                     String layer_name) {
  auto attrs = make_object<QnnCSIDenseAttrs>();
  attrs->units = std::move(units);
  attrs->out_dtype = std::move(out_dtype);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->kernel_zero_point = std::move(kernel_zero_point);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->input_scale = std::move(input_scale);
  attrs->kernel_scale = std::move(kernel_scale);
  attrs->output_scale = std::move(output_scale);
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.dense");
  return Call(op, {data, weight, bias}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.dense")
    .describe(R"code(Applies a linear transformation: :math:`Y = XW^T`.
- **data**: quantized(int8, unit8) `(x1, x2, ..., xn, input_dim)`
- **weight**: quantized(int8, unit8) `(units, input_dim)`
- **bias**: quantized(int32) `(units, input_dim)`
- **out**: quantized(int32) `(x1, x2, ..., xn, units)`.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIDenseAttrs>()
    .set_num_inputs(3)
    .add_argument("data", "quantized nD Tensor", "Input data.")
    .add_argument("weight", "quantized 2D Tensor", "Weight matrix.")
    .add_argument("bias", "quantized 2D Tensor", "bias matrix.")
    .set_support_level(11)
    .add_type_rel("QnnCSIDenseRel", QnnCSIDenseRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIDense").set_body_typed(MakeQnnCSIDense);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
