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
 * \file src/relay/qnn/op/concatenate.cc
 * \brief QNN concatenate operator. It concatenates quantized input tensors along a given axis.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/tir/expr.h>

#include "../../op/tensor/transform.h"
#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {

TVM_REGISTER_NODE_TYPE(QnnConcatenateAttrs);

Expr MakeQnnCSIConcatenate(Expr data, Array<tvm::PrimExpr> input_scales,
                           Array<tvm::PrimExpr> input_zero_points, double output_scale,
                           int32_t output_zero_point, int axis, Array<IndexExpr> max_values,
                           Array<IndexExpr> min_values, String layer_name) {
  auto attrs = make_object<QnnConcatenateAttrs>();
  attrs->input_scales = std::move(input_scales);
  attrs->input_zero_points = std::move(input_zero_points);
  attrs->output_scale = std::move(output_scale);
  attrs->output_zero_point = std::move(output_zero_point);
  attrs->axis = std::move(axis);
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.concatenate");
  return Call(op, {data}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.concatenate")
    .describe(R"code(Concatenate the quantized input tensors along the given axis.
)code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnConcatenateAttrs>()
    .set_num_inputs(1)
    .add_argument("data", "Tensor", "The tensor to concatenate.")
    .set_support_level(11)
    .add_type_rel("QnnCSIConcatenate", ConcatenateRel<QnnConcatenateAttrs>)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIConcatenate").set_body_typed(MakeQnnCSIConcatenate);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
