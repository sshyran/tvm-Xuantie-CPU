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

/* relay.broadcast_to */
TVM_REGISTER_NODE_TYPE(QnnCSIBroadCastToAttrs);

bool QnnCSIBroadCastToRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                          const TypeReporter& reporter) {
  // types = [data_type, ret_type], broadcast_to_type is in attrs bc static
  CHECK_EQ(types.size(), 2);

  const QnnCSIBroadCastToAttrs* param = attrs.as<QnnCSIBroadCastToAttrs>();
  CHECK(param);

  DataType out_dtype = types[0].as<TensorTypeNode>()->dtype;
  std::vector<IndexExpr> oshape;

  const Array<Integer>& cshape_array = param->shape;
  for (size_t i = 0; i < cshape_array.size(); ++i) {
    oshape.push_back(cshape_array[i]);
  }
  reporter->Assign(types[1], TensorType(oshape, out_dtype));
  return BroadcastRel({types[0], types[1], types[1]}, 2, Attrs(), reporter);
}

Expr MakeQnnCSIBroadCastTo(Expr data, Array<Integer> shape, double input_scale,
                           int32_t input_zero_point, double output_scale, int32_t output_zero_point,
                           DataType out_dtype, Array<IndexExpr> max_values,
                           Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnCSIBroadCastToAttrs>();
  attrs->shape = std::move(shape);
  attrs->input_scale = std::move(input_scale);
  attrs->input_zero_point = std::move(input_zero_point);
  attrs->output_scale = std::move(output_scale);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);

  static const Op& op = Op::Get("qnn.csi.broadcast_to");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.broadcast_to")
    .describe(
        R"code(Given tensor, this operation returns a new tensor that has the same values as tensor in
          the same order, except with a new shape given by newshape.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIBroadCastToAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The quantized data tensor.")
    .set_support_level(11)
    .add_type_rel("QnnCSIBroadCastToRel", QnnCSIBroadCastToRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIBroadCastTo").set_body_typed(MakeQnnCSIBroadCastTo);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
