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
 * \file graph_runtime.cc
 */
#include "hhb_runtime.h"

#include <math.h>
#include <stdio.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Run all the operations one by one.
 */

void HHBRuntime::Run() {
  auto func = module_.GetFunction("csinn_runtime_wrapper_", false);
  int i_size = input_.size();
  void** inputs = reinterpret_cast<void**>(malloc(i_size * sizeof(void*)));
  for (int i = 0; i < i_size; i++) {
    inputs[i] = input_[i]->data;
  }

  int o_size = output_.size();
  void** outputs = reinterpret_cast<void**>(malloc(o_size * sizeof(void*)));
  for (int i = 0; i < o_size; i++) {
    outputs[i] = output_[i]->data;
  }
  func(inputs, outputs, params_);
}

/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param ctxs The context of the host and devices where graph nodes will be
 * executed on.
 */
void HHBRuntime::Init(tvm::runtime::Module module, const std::vector<TVMContext>& ctxs) {
  module_ = module;
  ctxs_ = ctxs;
}

/*!
 * \brief set input to the graph.
 * \param data The input data.
 */
void HHBRuntime::SetOutput(NDArray data) { output_.push_back(data); }

/*!
 * \brief set output to the graph.
 * \param data The output data.
 */
void HHBRuntime::SetInput(NDArray data_in) { input_.push_back(data_in); }

/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray HHBRuntime::GetInput(int index) const { return input_[index]; }

/*!
 * \brief Return NDArray for given output index.
 * \param index The output index.
 *
 * \return NDArray corresponding to given output node index.
 */
NDArray HHBRuntime::GetOutput(int index) {
  input_.clear();
  return output_[index];
}

char* read_file_content(const std::string& path) {
  FILE* fp = fopen(path.c_str(), "rb");
  if (fp == NULL) {
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int file_size = ftell(fp);
  rewind(fp);

  char* buffer = reinterpret_cast<char*>(malloc(file_size));
  if (buffer == NULL) {
    return NULL;
  }

  int ret = fread(buffer, 1, file_size, fp);
  if (ret != file_size) {
    return NULL;
  }

  fclose(fp);
  return buffer;
}

void HHBRuntime::SetParams(const std::string& params_path) {
  params_ = read_file_content(params_path);
}

PackedFunc HHBRuntime::GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->SetInput(args[0]); });
  } else if (name == "set_params") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->SetParams(args[0]); });
  } else if (name == "set_output") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->SetOutput(args[0]); });
  } else if (name == "get_output") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetOutput(args[0]); });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = 0;
      in_idx = args[0];
      CHECK_GE(in_idx, 0);
      *rv = this->GetInput(in_idx);
    });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(); });
  } else {
    return PackedFunc();
  }
}

std::vector<TVMContext> HHBGetAllContext(const TVMArgs& args) {
  // Reserve the first item as the fallback device.
  std::vector<TVMContext> ret;
  TVMContext ctx;
  for (int i = 1; i < args.num_args; i += 2) {
    int dev_type = args[i];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[i + 1];
    ret.push_back(ctx);
  }
  return ret;
}

Module HHBRuntimeCreate(const tvm::runtime::Module& m, const std::vector<TVMContext>& ctxs) {
  auto exec = make_object<HHBRuntime>();
  exec->Init(m, ctxs);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.hhb_runtime.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  const auto& contexts = HHBGetAllContext(args);
  *rv = HHBRuntimeCreate(args[0], contexts);
});

}  // namespace runtime
}  // namespace tvm
