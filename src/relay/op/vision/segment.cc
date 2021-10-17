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
 * \file segment_max.cc
 * \brief segment_max operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(SegmentAttrs);

bool SegmentRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* data = types[0].as<TensorTypeNode>();
  const auto* segment_ids = types[1].as<TensorTypeNode>();

  if (data == nullptr) return false;
  if (segment_ids == nullptr) return false;
  const auto dshape = data->shape;
  const auto segment_shape = segment_ids->shape;

  CHECK(num_inputs == 2) << "num_inputs must be 2.";
  CHECK(segment_shape.size() == 1) << "segment_ids only support 1-D";

  const auto param = attrs.as<SegmentAttrs>();
  CHECK(param != nullptr);

  std::vector<IndexExpr> oshape;
  if (dshape.size() == 1) {
    oshape.push_back(param->length);
  } else if (dshape.size() == 2) {
    oshape.push_back(dshape[0]);
    oshape.push_back(param->length);
  }

  // assign output type
  reporter->Assign(types[2], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeSegmentMax(Expr data, Expr segment_ids, int32_t length) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->length = std::move(length);
  static const Op& op = Op::Get("vision.segment_max");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.segment_max").set_body_typed(MakeSegmentMax);

RELAY_REGISTER_OP("vision.segment_max")
    .describe(R"doc(segment_max operator.

 - **data**: Input is 1D or 2D array.
 )doc" TVM_ADD_FILELINE)
    .set_attrs_type<SegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .add_argument("length", "Int", "output length.")
    .set_support_level(5)
    .add_type_rel("SegmentMaxRel", SegmentRel);

Expr MakeSegmentSum(Expr data, Expr segment_ids, int32_t length) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->length = std::move(length);
  static const Op& op = Op::Get("vision.segment_sum");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.segment_sum").set_body_typed(MakeSegmentSum);

RELAY_REGISTER_OP("vision.segment_sum")
    .describe(R"doc(segment_max operator.

 - **data**: Input is 1D or 2D array.
 )doc" TVM_ADD_FILELINE)
    .set_attrs_type<SegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .add_argument("length", "Int", "output length.")
    .set_support_level(5)
    .add_type_rel("SegmentSumRel", SegmentRel);

Expr MakeSegmentMean(Expr data, Expr segment_ids, int32_t length) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->length = std::move(length);
  static const Op& op = Op::Get("vision.segment_mean");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.segment_mean").set_body_typed(MakeSegmentMean);

RELAY_REGISTER_OP("vision.segment_mean")
    .describe(R"doc(segment_max operator.

 - **data**: Input is 1D or 2D array.
 )doc" TVM_ADD_FILELINE)
    .set_attrs_type<SegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .add_argument("length", "Int", "output length.")
    .set_support_level(5)
    .add_type_rel("SegmentMeanRel", SegmentRel);

Expr MakeSegmentProd(Expr data, Expr segment_ids, int32_t length) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->length = std::move(length);
  static const Op& op = Op::Get("vision.segment_prod");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.segment_prod").set_body_typed(MakeSegmentProd);

RELAY_REGISTER_OP("vision.segment_prod")
    .describe(R"doc(segment_max operator.

 - **data**: Input is 1D or 2D array.
 )doc" TVM_ADD_FILELINE)
    .set_attrs_type<SegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .add_argument("length", "Int", "output length.")
    .set_support_level(5)
    .add_type_rel("SegmentProdRel", SegmentRel);

Expr MakeSegmentMin(Expr data, Expr segment_ids, int32_t length) {
  auto attrs = make_object<SegmentAttrs>();
  attrs->length = std::move(length);
  static const Op& op = Op::Get("vision.segment_min");
  return Call(op, {data, segment_ids}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.segment_min").set_body_typed(MakeSegmentMin);

RELAY_REGISTER_OP("vision.segment_min")
    .describe(R"doc(segment_max operator.

 - **data**: Input is 1D or 2D array.
 )doc" TVM_ADD_FILELINE)
    .set_attrs_type<SegmentAttrs>()
    .set_num_inputs(2)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("segment_ids", "Tensor",
                  "1-D tensor whose size is equal to the size"
                  "of data's first dimension. Values should be sorted and can be repeated.")
    .add_argument("length", "Int", "output length.")
    .set_support_level(5)
    .add_type_rel("SegmentMinRel", SegmentRel);

}  // namespace relay
}  // namespace tvm
