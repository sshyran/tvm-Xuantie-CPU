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
#include "csinn.h"
#include "gref.h"
#include "hlight.h"
#include "i805.h"
#include "light.h"
#include "ref.h"
#ifdef BUILD_PNNA
#include "light_new.h"
#endif

#ifdef USE_JSON_RUNTIME
#include <tvm/tir/analysis.h>

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"
#endif

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;
using namespace quantize;

static Map<String, Array<Array<Array<IndexExpr>>>> quant_info;

/*!
 * \brief The CSINN codegen helper to generate wrapepr function calls of CSINN
 * libraries. The code is a CSourceModule that can be compiled separately and
 * linked together with a DSOModule.
 */
class CSINNModuleCodegen : public CSourceModuleCodegenBase {
 public:
  CSINNModuleCodegen() {}

  // Create a corresponding CSINN function for the given relay Function.
  void GenCSINNFunc(const Function& func) {
    CHECK(func.defined()) << "Input error: expect a Relay function.";
    auto ctx = transform::PassContext::Current();
    auto cfg = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!cfg.defined()) {
      cfg = AttrsWithDefaultValues<CSINNConfig>();
    }
    String device = cfg.value()->target;
    bool auto_quant = cfg.value()->auto_hybrid_quantization;
    LayerCounter layer_counter;
    layer_counter.VisitExpr(func->body);

    if (device == "anole") {
      CodegenAnole builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
    } else if (device == "light" && !auto_quant) {
      CodegenLight builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
#ifdef BUILD_PNNA
    } else if (device == "light_new") {
      CodegenLightNew builder(func->body);
      builder.optimization();
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
#endif
    } else if (device == "hlight" || (device == "light" && auto_quant)) {
      CodegenHLight builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
    } else if (device == "asp") {
      CodegenASP builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
    } else if (device == "c906") {
      CodegenGref builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
    } else if (device == "c908") {
      CodegenGref builder;
      builder.VisitExpr(func->body);
      code_stream_ << builder.EmitGraph();
      quant_info = builder.ret_quant_info();
    } else if (device == "i805") {
      CodegenI805 builder;
      builder.layer_count = layer_counter.GetLayerCounter();
      builder.VisitExpr(func->body);
      code_stream_ << builder.JIT();
      quant_info = builder.ret_quant_info();
    } else {
      CodegenRef builder;
      builder.SetExtFuncId(GetExtSymbol(func));
      builder.layer_count = layer_counter.GetLayerCounter();
      builder.VisitExpr(func->body);
      code_stream_ << builder.JIT();
      quant_info = builder.ret_quant_info();
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

    std::string code = code_stream_.str();
    String sym = GetExtSymbol(Downcast<Function>(ref));
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

#ifdef USE_JSON_RUNTIME
class SHLJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  SHLJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    if (cn->op.as<OpNode>()) {
      return JSONSerializer::VisitExpr_(cn);
    }
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "SHL JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    ICHECK(comp.defined()) << "SHL JSON runtime only supports composite functions.";
    name = comp.value();

    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "shl.conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else if (name == "shl.dense") {
      json_node = CreateCompositeDenseJSONNode(cn);
    } else {
      LOG(FATAL) << "Unrecognized SHL pattern: " << name;
    }
    return AddNode(json_node, GetRef<Expr>(cn));
  }

 private:
  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeConvNode {
    const CallNode* conv = nullptr;
    const CallNode* bias = nullptr;
  };

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeConvNode UnpackCompositeConvolution(const CallNode* cn) {
    CompositeConvNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);

    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.bias_add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    nodes.conv = current_call;

    return nodes;
  }

  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeDense {
    const CallNode* dense = nullptr;
    const CallNode* bias = nullptr;
  };

  /*!
   * \brief Extract dense nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite dense nodes.
   */
  static CompositeDense UnpackCompositeDense(const CallNode* cn) {
    CompositeDense nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    ICHECK(fn);

    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "nn.bias_add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }

    nodes.dense = current_call;

    return nodes;
  }

  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallNode* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);

    const auto* conv_attr = nodes.conv->attrs.as<Conv2DAttrs>();
    ICHECK(conv_attr);

    std::string name;
    std::string name_prefix = "shl";

    // Distinguish between normal and depth-wise convolution
    if (conv_attr->channels.defined() &&
        tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
        conv_attr->groups != 1) {
      name = "depthwise_conv2d";
    } else {
      name = "conv2d";
    }

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name_prefix + "." + name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);

    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite dense.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeDenseJSONNode(const CallNode* cn) {
    CompositeDense nodes = UnpackCompositeDense(cn);

    std::string name = "shl.dense";
    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.dense->args[1])[0]);
    inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.dense);

    return json_node;
  }
};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module SHLCompiler(const ObjectRef& ref) {
  ICHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  SHLJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* pf = runtime::Registry::Get("runtime.SHLJSONRuntimeCreate");
  ICHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}
TVM_REGISTER_GLOBAL("relay.ext.shl").set_body_typed(SHLCompiler);

inline constexpr bool IsSHLRuntimeEnabled() { return true; }

TVM_REGISTER_GLOBAL("relay.op.is_shl_runtime_enabled").set_body_typed(IsSHLRuntimeEnabled);
#endif
runtime::Module CSINNCompiler(const ObjectRef& ref) {
  CSINNModuleCodegen csinn;
  return csinn.CreateCSourceModule(ref);
}

Map<String, Array<Array<Array<IndexExpr>>>> CollectQuantInfo() { return quant_info; }

TVM_REGISTER_GLOBAL("relay.ext.csinn.collect_quant_info").set_body_typed(CollectQuantInfo);
TVM_REGISTER_GLOBAL("relay.ext.csinn").set_body_typed(CSINNCompiler);
TVM_REGISTER_NODE_TYPE(CSINNConfigNode);
TVM_REGISTER_PASS_CONFIG_OPTION("relay.ext.csinn.options", CSINNConfig);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
