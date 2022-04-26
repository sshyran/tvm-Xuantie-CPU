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
 * \file src/relay/backend/contrib/csinn/light.h
 * \brief The base class for light.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_LIGHT_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_LIGHT_H_

#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {

class CodegenLight : public CodegenCSINN {
 public:
  CodegenLight() : CodegenCSINN() {}
  virtual ~CodegenLight() {}

  virtual void VisitExpr_(const CallNode* call);
  void malloc_buf(string out, int out_size) {}
  void CreateMallocBuf(string name, std::vector<int> shape, string dtype) {}
  void CreateTensorSessData() {}
  void CreateHybridTensorSessData(std::vector<int> shape, string dtype) {}
  void FreeTensor(const Expr& expr, string name) {}
  void ModelBinarySave();
  void EmitHeader(void);
  void EmitSessionSetup(void);
  void EmitJitWrapper(void);
  void EmitNBGSetup(void);
  string EmitGraph(void);
  virtual void GetSymScale(float min_value, float max_value, int bits, Qinfo* qinfo);
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_LIGHT_H_
