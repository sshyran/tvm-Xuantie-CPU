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
 * \file src/relay/qnn/csi_op/PSROIPooling.cc
 * \brief QNN PSROIPooling operator.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/qnn/attrs.h>

#include "../op/op_common.h"
#include "../util.h"

namespace tvm {
namespace relay {
namespace qnn {
TVM_REGISTER_NODE_TYPE(QnnCSIROIPoolingAttrs);

bool QnnCSIROIPoolingRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                         const TypeReporter& reporter) {
  auto roipooling_attrs = attrs.as<QnnCSIROIPoolingAttrs>();
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* roi_pred = types[1].as<TensorTypeNode>();

  if (!data || !roi_pred) {
    return false;
  }

  //   CHECK_EQ(cls_prob->shape.size(), 4U)
  //       << "The dimension of class probability should be 4, but received " <<
  //       cls_prob->shape.size();
  //   CHECK_EQ(roi_pred->shape.size(), 2U)
  //       << "The dimension of roi prediction should be 2, but received " <<
  //       roi_pred->shape.size();

  auto num_rois = roi_pred->shape[0];
  auto channel = data->shape[1];

  std::vector<IndexExpr> oshape(
      {num_rois, channel, roipooling_attrs->pooled_size[0], roipooling_attrs->pooled_size[1]});
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}
// QNN PSROIPooling operator.
Expr MakeQnnCSIROIPooling(Expr data, Expr roi, Array<IndexExpr> pooled_size, double spatial_scale,
                          Array<tvm::PrimExpr> input_scales, Array<tvm::PrimExpr> input_zero_points,
                          double output_scale, int32_t output_zero_point, DataType out_dtype,
                          Array<IndexExpr> max_values, Array<IndexExpr> min_values,
                          String layer_name) {
  auto attrs = make_object<QnnCSIROIPoolingAttrs>();
  attrs->pooled_size = std::move(pooled_size);
  attrs->spatial_scale = spatial_scale;
  attrs->input_scales = input_scales;
  attrs->input_zero_points = input_zero_points;
  attrs->output_scale = output_scale;
  attrs->output_zero_point = output_zero_point;
  attrs->out_dtype = out_dtype;
  attrs->layer_name = layer_name;
  attrs->max_values = std::move(max_values);
  attrs->min_values = std::move(min_values);
  static const Op& op = Op::Get("qnn.csi.roipooling");
  return Call(op, {data, roi}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("qnn.csi.roipooling")
    .describe(R"code(roipooling
 )code" TVM_ADD_FILELINE)
    .set_attrs_type<QnnCSIROIPoolingAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "feature map for roipooing")
    .add_argument("roi", "Tensor", "roi for proposals")
    .set_support_level(11)
    .add_type_rel("QnnCSIROIPoolingRel", QnnCSIROIPoolingRel)
    .set_attr<TOpPattern>("TOpPattern", kOpaque);

TVM_REGISTER_GLOBAL("relay.qnn.op._make.CSIROIPooling").set_body_typed(MakeQnnCSIROIPooling);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
