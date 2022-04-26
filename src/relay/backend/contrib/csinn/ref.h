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
 * \file src/relay/backend/contrib/csinn/ref.h
 * \brief The base class for external codegen tools.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_REF_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_REF_H_

#include <string>
#include <vector>

#include "csinn.h"

namespace tvm {
namespace relay {
namespace contrib {

class CodegenRef : public CodegenCSINN {
 public:
  CodegenRef() : CodegenCSINN() {}
  virtual ~CodegenRef() {}

  virtual void GenerateBackendCFunc(const string& func_name, const Array<Var>& args,
                                    const Output& out);
  virtual string JitImpl(const string& ext_func_id, const Array<Var>& args,
                         const std::vector<string>& buf_decl, const std::vector<string>& body,
                         const std::vector<Output>& out);
  virtual string JIT(const std::vector<Output>& out);
  virtual string JIT(void);

  void malloc_buf(string out, int out_size);
  void CreateMallocBuf(string name, std::vector<int> shape, string dtype);
  void CreateTensorSessData();
  void CreateHybridTensorSessData(std::vector<int> shape, string dtype);
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_REF_H_
