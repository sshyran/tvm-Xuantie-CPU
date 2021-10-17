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
 * \file invert_permutation.cc
 * \brief invert_permutation operators
 */
#include <tvm/relay/attrs/vision.h>
#include <tvm/relay/op.h>
#include <tvm/relay/op_attr_types.h>

namespace tvm {
namespace relay {

bool InvertPermutationRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                          const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  const auto* data = types[0].as<TensorTypeNode>();

  if (data == nullptr) return false;

  const auto dshape = data->shape;
  CHECK(dshape.size() == 1) << "InvertPermutation only support input == 1-D";

  std::vector<IndexExpr> oshape(dshape.begin(), dshape.end());
  // assign output type
  reporter->Assign(types[1], TensorType(oshape, data->dtype));
  return true;
}

Expr MakeInvertPermutation(Expr data) {
  static const Op& op = Op::Get("vision.invert_permutation");
  return Call(op, {data}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relay.op.vision._make.invert_permutation")
    .set_body_typed(MakeInvertPermutation);

RELAY_REGISTER_OP("vision.invert_permutation")
    .describe(R"doc(invert_permutation operator.

 - **data**: Input is 1D array
 )doc" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .set_support_level(5)
    .add_type_rel("InvertPermutationRel", InvertPermutationRel);

}  // namespace relay
}  // namespace tvm
