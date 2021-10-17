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
TVM_REGISTER_NODE_TYPE(QnnBinaryOpAttrs);

// QNN Poweride operator.
Expr MakeQnnCSIPower(Expr lhs, Expr rhs, double lhs_scale, int32_t lhs_zero_point, double rhs_scale,
                     int32_t rhs_zero_point, double output_scale, int32_t output_zero_point,
                     Array<IndexExpr> max_values, Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnBinaryOpAttrs>();
  attrs->lhs_scale = lhs_scale;
  attrs->lhs_zero_point = lhs_zero_point;
  attrs->rhs_scale = rhs_scale;
  attrs->rhs_zero_point = rhs_zero_point;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.power");
  return Call(op, {lhs, rhs}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.power")
    .describe("Elementwise power with with broadcasting for quantized tensors.")
    .set_attrs_type<QnnBinaryOpAttrs>()
    .set_num_inputs(2)
    .add_argument("lhs", "Tensor", "The left hand side quantized tensor.")
    .add_argument("rhs", "Tensor", "The right hand side quantized tensor.")
    .set_support_level(11)
    .add_type_rel("Broadcast", BroadcastRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);
TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIPower").set_body_typed(MakeQnnCSIPower);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
