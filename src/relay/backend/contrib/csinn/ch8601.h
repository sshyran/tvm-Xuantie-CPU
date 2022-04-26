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
 * \file src/relay/backend/contrib/csinn/ch8601.h
 * \brief The base class for ch8601
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_CH8601_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_CH8601_H_

#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {

class CodegenCH8601 : public CodegenCSINN {
 public:
  CodegenCH8601() : CodegenCSINN() {}
  virtual ~CodegenCH8601() {}
  virtual string EmitGraph(void);
  virtual void VisitExpr_(const CallNode* call);

  void malloc_buf(string out, int out_size) {}
  void CreateMallocBuf(string name, std::vector<int> shape, string dtype) {}
  void CreateTensorSessData() {}
  void CreateHybridTensorSessData(std::vector<int> shape, string dtype) {}
  void FreeTensor(const Expr& expr, string name) {}
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_CH8601_H_
