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
 * \file src/relay/qnn/op/lrn.cc
 * \brief QNN lrn operator.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../../op/tensor/transform.h"
#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

bool QnnCSILRNRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                  const TypeReporter& reporter) {
  const auto* data = types[0].as<TensorTypeNode>();
  if (data == nullptr) return false;
  const auto* param = attrs.as<QnnCSILRNAttrs>();
  CHECK(param != nullptr) << "QnnCSILRNAttrs cannot be nullptr.";
  for (size_t i = 1; i < types.size(); ++i) {
    reporter->Assign(types[i], types[0]);
  }
  return true;
}

// Positional relay function to create LRN operator used by frontend FFI.
TVM_REGISTER_NODE_TYPE(QnnCSILRNAttrs);

Expr MakeQnnCSILRN(Expr data, int size, int axis, double alpha, double beta, double bias,
                   double input_scale, int32_t input_zero_point, double output_scale,
                   int32_t output_zero_point, DataType out_dtype, Array<IndexExpr> max_values,
                   Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSILRNAttrs>();
  attrs->size = std::move(size);
  attrs->axis = std::move(axis);
  attrs->alpha = std::move(alpha);
  attrs->beta = std::move(beta);
  attrs->bias = std::move(bias);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->input_scale = std::move(input_scale);
  attrs->output_scale = std::move(output_scale);
  attrs->out_dtype = std::move(out_dtype);
  attrs->layer_name = std::move(layer_name);
  static const Op& op = Op::Get("qnn.csi.lrn");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.lrn")
    .describe("lrn")
    .set_attrs_type<QnnCSILRNAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "quantized nD Tensor", "Input data.")
    .set_support_level(11)
    .add_type_rel("QnnCSILRNRel", QnnCSILRNRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSILRN").set_body_typed(MakeQnnCSILRN);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
