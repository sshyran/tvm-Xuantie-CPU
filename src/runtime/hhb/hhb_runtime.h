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
 * \brief Tiny graph runtime that can run graph
 *        containing only tvm PackedFunc.
 * \file hhb_runtime.h
 */
#ifndef TVM_RUNTIME_HHB_HHB_RUNTIME_H_
#define TVM_RUNTIME_HHB_HHB_RUNTIME_H_

#include <dlpack/dlpack.h>
#include <dmlc/json.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Tiny graph runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
class TVM_DLL HHBRuntime : public ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final { return "HHBRuntime"; }

  void Run();

  /*!
   * \brief Initialize the graph executor with graph and context.
   * \param graph_json The execution graph.
   * \param module The module containing the compiled functions for the host
   *  processor.
   * \param ctxs The context of the host and devices where graph nodes will be
   *  executed on.
   */

  void Init(tvm::runtime::Module module, const std::vector<Device>& ctxs);

  /*!
   * \brief set index-th input to the graph.
   * \param index The input index.
   * \param data_in The input data.
   */
  void SetInput(NDArray data);
  void SetParams(const std::string& params_path);
  void SetOutput(NDArray data);
  /*!
   * \brief Return NDArray for given input index.
   * \param index The input index.
   *
   * \return NDArray corresponding to given input node index.
   */
  NDArray GetInput(int index) const;
  /*!
   * \brief Return NDArray for given output index.
   * \param index The output index.
   *
   * \return NDArray corresponding to given output node index.
   */
  NDArray GetOutput(int index);

 protected:
  /*! \brief The code module that contains both host and device code. */
  tvm::runtime::Module module_;
  /*! \brief Execution context of all devices including the host. */
  std::vector<Device> ctxs_;

  std::vector<NDArray> input_;
  std::vector<NDArray> output_;
  const char* out_buf_;
  size_t output_size_;
  char* params_;
};
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HHB_HHB_RUNTIME_H_
