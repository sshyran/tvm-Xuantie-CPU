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
 * \file src/relay/backend/contrib/csinn/codegen.cc
 * \brief Implementation of CSINN codegen APIs.
 */

#include "anole.h"
#include "asp.h"
#include "ch8601.h"
#include "csinn.h"
#include "dp1k.h"
#include "gref.h"
#include "hlight.h"
#include "i805.h"
#include "light.h"
#include "ref.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;
using namespace quantize;

/*!
 * \brief The CSINN codegen helper to generate wrapepr function calls of CSINN
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class CSINNModuleCodegen : public CSourceModuleCodegenBase {
 public:
  explicit CSINNModuleCodegen(const tvm::Target& target, const string& path) {}

  // Create a corresponding CSINN function for the given relay Function.
  void GenCSINNFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<CSINNConfig>();
    }
    String device = cfg.value()->target;
    LayerCounter layer_counter;
    layer_counter.VisitExpr(func->body);

    if (device == "anole") {
      CodegenAnole builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "light") {
      CodegenLight builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
#ifdef BUILD_PNNA
    } else if (device == "light_new") {
      CodegenLightNew builder;
      builder.init_model();
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
#endif
    } else if (device == "hlight") {
      CodegenHLight builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "ch8601") {
      CodegenCH8601 builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "dp1k") {
      CodegenDP1K builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "asp") {
      CodegenASP builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "c906") {
      CodegenGref builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "c908") {
      CodegenGref builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
    } else if (device == "i805") {
      CodegenI805 builder;
      builder.layer_count = layer_counter.GetLayerCounter();
      builder.VisitExpr(func->body);
      code_stream_ << builder.JIT();
    } else {
      CodegenRef builder;
      builder.layer_count = layer_counter.GetLayerCounter();
      builder.VisitExpr(func->body);
      code_stream_ << builder.JIT();
    }
  }

  /*!
   * \brief The overridden function that will create a CSourceModule. In order
   * to compile the generated C source code, users need to specify the paths to
   * some libraries, including some TVM required and csinn specific ones. To make
   * linking simpiler, the CSINN kernels are wrapped in a TVM compatible manner
   * and live under tvm/src/runtime/contrib/csinn folder.
   *
   * \param ref An object ref that could be either a Relay function or module.
   *
   * \return The runtime module that contains C source code.
   */
  runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
    if (ref->IsInstance<FunctionNode>()) {
      GenCSINNFunc(Downcast<Function>(ref));
    } else if (ref->IsInstance<IRModuleNode>()) {
      IRModule mod = Downcast<IRModule>(ref);
      for (const auto& it : mod->functions) {
        GenCSINNFunc(Downcast<Function>(it.second));
      }
    } else {
      LOG(FATAL) << "The input ref is expected to be a Relay function or module"
                 << "\n";
    }

    String sym = "";
    Array<String> variables = {};
    // Create a CSourceModule
    const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
    CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
    return (*pf)(code_stream_.str(), "cc", Array<String>{sym}, variables);
  }

 private:
  /*!
   * \brief The code stream that prints the code that will be compiled using
   * external codegen tools.
   */
  std::ostringstream code_stream_;
  tvm::Target target_;
  string params_path_;
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module CSINNCompiler(const ObjectRef& ref, const tvm::Target& target,
                              const string& params_path) {
  CSINNModuleCodegen csinn(target, params_path);
  return csinn.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.csinn").set_body_typed(CSINNCompiler);

TVM_REGISTER_NODE_TYPE(CSINNConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.csinn.options", CSINNConfig);
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
