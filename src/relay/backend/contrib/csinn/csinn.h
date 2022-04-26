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
 * \file src/relay/backend/contrib/csinn/csinn.h
 * \brief The base class for external codegen tools.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_CSINN_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_CSINN_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/relay/qnn/attrs.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../../quantize/quantize.h"
#include "../../utils.h"
#include "../codegen_c/codegen_c.h"
using std::string;
using std::to_string;

namespace tvm {
namespace relay {
namespace contrib {

enum {
  USE_MINMAX = 0,
  USE_SCALE = 1,
};

enum {
  WEIGHT = 0,
  ACTIVATE = 1,
};

enum {
  PER_TENSOR = 0,
  PER_CHANNEL = 1,
};

enum {
  QINFO = 0,
  CONSTANT = 1,
};

struct CSIConstant {
  string name;
  string dtype;
  size_t size;
  void* data_buf;
};

struct Qinfo {
  int32_t zero_point;
  float scale;
  int32_t multiplier;
  int32_t shift;
  float min;
  float max;
};

struct QuantParams {
  struct Qinfo* qinfo;
  string name;
  std::vector<int> shape;
  int32_t q_size;
  int32_t offset;
};

struct QConfig_ {
  string quantization_scheme;
  string dtype_input;
  string dtype_weight;
  string dtype_activation;
  int nbit_input;
  int nbit_weight;
  int nbit_activation;
  string activate_quantized_type;
  string weight_quantized_type;
  bool fuse_zp2bias;
};

#define HHB_VERSION "1.12.0"

/*! \brief Attributes to store the options for CSI-NN2 */
struct CSINNConfigNode : public tvm::AttrsNode<CSINNConfigNode> {
  std::string sid;
  std::string target;
  std::string params_path;
  std::string debug_level;
  std::string run_mode;
  std::string model_save;
  bool multi_thread;
  std::string trace_strategy;
  Array<Integer> input_memory_type;
  Array<Integer> output_memory_type;

  std::string quantization_scheme;
  std::string hybrid_quantization_scheme;
  Array<String> hybrid_layer_name;
  int nbit_input;
  int nbit_weight;
  int nbit_activation;
  std::string dtype_input;
  std::string dtype_weight;
  std::string dtype_activation;
  std::string calibrate_mode;
  double global_scale;
  std::string weight_scale;
  std::string activate_quantized_type;
  std::string weight_quantized_type;
  std::string layout;
  double channel_quantization_ratio_threshold;

  bool fuse_clip;
  bool fuse_relu;
  bool fuse_conv_relu;
  bool fuse_reshape_dense;
  bool fuse_mul_add_to_conv;
  bool channel_quantization;
  bool broadcast_quantization;
  bool fuse_mul_before_conv;
  bool fuse_mul_after_conv;
  bool fuse_add_before_conv;
  bool fuse_add_after_conv;
  bool fuse_zp2bias;
  bool use_custom_fusion;
  bool convert_to_relay;

  /* target specific */
  /* light */
  double light_input_fix_height;
  double light_input_fix_width;

  int h_sram_size;
  int h_max_groups;
  int h_max_out_channel;
  int h_max_kernel_size;
  bool h_contain_weight;

  TVM_DECLARE_ATTRS(CSINNConfigNode, "ext.attrs.CSINNConfigNode") {
    TVM_ATTR_FIELD(sid).set_default("csinn");
    TVM_ATTR_FIELD(target).set_default("ref");
    TVM_ATTR_FIELD(params_path).set_default("./");
    TVM_ATTR_FIELD(debug_level).set_default("WARNING");
    TVM_ATTR_FIELD(run_mode).set_default("graph");
    TVM_ATTR_FIELD(model_save).set_default("run_only");
    TVM_ATTR_FIELD(multi_thread).set_default(false);
    TVM_ATTR_FIELD(trace_strategy).set_default("normal");
    TVM_ATTR_FIELD(quantization_scheme).set_default("unset");
    TVM_ATTR_FIELD(hybrid_quantization_scheme).set_default("unset");
    TVM_ATTR_FIELD(hybrid_layer_name).set_default(Array<String>({""}));
    TVM_ATTR_FIELD(nbit_input).set_default(8);
    TVM_ATTR_FIELD(nbit_weight).set_default(8);
    TVM_ATTR_FIELD(nbit_activation).set_default(32);
    TVM_ATTR_FIELD(dtype_input).set_default("int8");
    TVM_ATTR_FIELD(dtype_weight).set_default("int8");
    TVM_ATTR_FIELD(dtype_activation).set_default("int32");
    TVM_ATTR_FIELD(calibrate_mode).set_default("global_scale");
    TVM_ATTR_FIELD(global_scale).set_default(8.0);
    TVM_ATTR_FIELD(weight_scale).set_default("power2");
    TVM_ATTR_FIELD(activate_quantized_type).set_default("asym");
    TVM_ATTR_FIELD(weight_quantized_type).set_default("asym");
    TVM_ATTR_FIELD(layout).set_default("NCHW");
    TVM_ATTR_FIELD(channel_quantization_ratio_threshold).set_default(0.0);
    TVM_ATTR_FIELD(fuse_clip).set_default(false);
    TVM_ATTR_FIELD(fuse_relu).set_default(false);
    TVM_ATTR_FIELD(fuse_conv_relu).set_default(false);
    TVM_ATTR_FIELD(fuse_reshape_dense).set_default(false);
    TVM_ATTR_FIELD(fuse_mul_add_to_conv).set_default(true);
    TVM_ATTR_FIELD(channel_quantization).set_default(false);
    TVM_ATTR_FIELD(broadcast_quantization).set_default(false);
    TVM_ATTR_FIELD(fuse_mul_before_conv).set_default(true);
    TVM_ATTR_FIELD(fuse_mul_after_conv).set_default(true);
    TVM_ATTR_FIELD(fuse_add_before_conv).set_default(true);
    TVM_ATTR_FIELD(fuse_add_after_conv).set_default(true);
    TVM_ATTR_FIELD(fuse_zp2bias).set_default(false);
    TVM_ATTR_FIELD(use_custom_fusion).set_default(false);
    TVM_ATTR_FIELD(convert_to_relay).set_default(false);
    TVM_ATTR_FIELD(light_input_fix_height).set_default(0.0);
    TVM_ATTR_FIELD(light_input_fix_width).set_default(0.0);
    TVM_ATTR_FIELD(input_memory_type).set_default(Array<Integer>({0}));
    TVM_ATTR_FIELD(output_memory_type).set_default(Array<Integer>({0}));
    TVM_ATTR_FIELD(h_sram_size).set_default(0);
    TVM_ATTR_FIELD(h_max_groups).set_default(0);
    TVM_ATTR_FIELD(h_max_out_channel).set_default(0);
    TVM_ATTR_FIELD(h_max_kernel_size).set_default(0);
    TVM_ATTR_FIELD(h_contain_weight).set_default(false);
  }
};

class LayerCounter : public ExprVisitor {
 public:
  virtual ~LayerCounter() {}
  std::unordered_map<const Object*, size_t> GetLayerCounter() {
    std::unordered_map<const Object*, size_t> layer_count;
    layer_count = this->visit_counter_;
    return layer_count;
  }
};

class CSINNConfig : public Attrs {
 public:
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(CSINNConfig, Attrs, CSINNConfigNode);
};

class CodegenCSINN : public ExprVisitor, public CodegenCBase {
 public:
  CodegenCSINN() {
    auto ctx = transform::PassContext::Current();
    auto opt = ctx->GetConfig<CSINNConfig>("relay.ext.csinn.options");
    if (!opt.defined()) {
      opt = AttrsWithDefaultValues<CSINNConfig>();
    }
    auto opt_cfg = opt.value();
    this->ext_func_id_ = opt_cfg->sid;
    this->layout_ = opt_cfg->layout;
    this->target_ = opt_cfg->target;
    this->params_path_ = opt_cfg->params_path;

    this->output_dir_ = dirnameOf(this->params_path_);

    this->debug_level_ = opt_cfg->debug_level;
    this->multithread = opt_cfg->multi_thread;
    this->model_save = opt_cfg->model_save;
    this->trace_strategy_ = opt_cfg->trace_strategy;
    this->input_memory_type = __convert_list(opt_cfg->input_memory_type);
    this->output_memory_type = __convert_list(opt_cfg->output_memory_type);

    this->hybrid_quantization_scheme = opt_cfg->hybrid_quantization_scheme;
    this->hybrid_layer_name = __convert_string_list(opt_cfg->hybrid_layer_name);

    if (this->target_ == "light") {
      target_name_ = "CSINN_LIGHT";
    } else if (this->target_ == "light_new") {
      target_name_ = "CSINN_LIGHT";
    } else if (this->target_ == "anole") {
      target_name_ = "CSINN_ANOLE";
    } else if (this->target_ == "ch8601") {
      target_name_ = "CSINN_CH8601";
    } else if (this->target_ == "dp1k") {
      target_name_ = "CSINN_DP1K";
    } else if (this->target_ == "c906") {
      target_name_ = "CSINN_C906";
    } else if (this->target_ == "c908") {
      target_name_ = "CSINN_C908";
    } else if (this->target_ == "i805") {
      target_name_ = "CSINN_I805";
    } else if (target_ == "hlight") {
      target_name_ = "CSINN_REF";
    } else if (target_ == "asp") {
      target_name_ = "CSINN_REF";
    } else if (target_ == "x86_ref") {
      target_name_ = "CSINN_REF";
    } else {
      LOG(WARNING) << "Unsupport target " << target_ << ".";
    }

    cfg = new QConfig_();
    if (opt_cfg->dtype_input == "int4") {
      cfg->dtype_input = "int4_t";
    } else if (opt_cfg->dtype_input == "uint8") {
      cfg->dtype_input = "uint8_t";
    } else if (opt_cfg->dtype_input == "int8") {
      cfg->dtype_input = "int8_t";
    } else if (opt_cfg->dtype_input == "float16") {
      cfg->dtype_input = "int16_t";
    } else if (opt_cfg->dtype_input == "bfloat16") {
      cfg->dtype_input = "int16_t";
    } else if (opt_cfg->dtype_input == "float32") {
      cfg->dtype_input = "float";
    } else if (opt_cfg->dtype_input == "int16") {
      cfg->dtype_input = "int16_t";
    } else {
      LOG(WARNING) << "Unsupport dtype input.";
    }

    if (opt_cfg->dtype_weight == "int4") {
      cfg->dtype_weight = "int4_t";
    } else if (opt_cfg->dtype_weight == "uint8") {
      cfg->dtype_weight = "uint8_t";
    } else if (opt_cfg->dtype_weight == "int8") {
      cfg->dtype_weight = "int8_t";
    } else if (opt_cfg->dtype_weight == "float16") {
      cfg->dtype_weight = "float16";
    } else if (opt_cfg->dtype_weight == "bfloat16") {
      cfg->dtype_weight = "bfloat16";
    } else if (opt_cfg->dtype_weight == "float32") {
      cfg->dtype_weight = "float";
    } else if (opt_cfg->dtype_weight == "int16") {
      cfg->dtype_weight = "int16_t";
    } else {
      LOG(WARNING) << "Unsupport dtype weight.";
    }

    if (opt_cfg->dtype_activation == "int32") {
      cfg->dtype_activation = "int32_t";
    } else if (opt_cfg->dtype_activation == "float16") {
      cfg->dtype_activation = "float16";
    } else if (opt_cfg->dtype_activation == "bfloat16") {
      cfg->dtype_activation = "bfloat16";
    } else if (opt_cfg->dtype_activation == "float32") {
      cfg->dtype_activation = "float";
    } else if (opt_cfg->dtype_activation == "int16") {
      cfg->dtype_activation = "int16_t";
    } else {
      LOG(WARNING) << "Unsupport dtype activation.";
    }

    if (opt_cfg->quantization_scheme == "int4_asym_w_sym") {
      cfg->quantization_scheme = "CSINN_QUANT_INT4_ASYM_W_SYM";
    } else if (opt_cfg->quantization_scheme == "uint8_asym") {
      cfg->quantization_scheme = "CSINN_QUANT_UINT8_ASYM";
    } else if (opt_cfg->quantization_scheme == "int8_sym") {
      cfg->quantization_scheme = "CSINN_QUANT_INT8_SYM";
    } else if (opt_cfg->quantization_scheme == "int8_original") {
      cfg->quantization_scheme = "CSINN_QUANT_INT8_ORIGINAL";
    } else if (opt_cfg->quantization_scheme == "int8_asym") {
      cfg->quantization_scheme = "CSINN_QUANT_INT8_ASYM";
    } else if (opt_cfg->quantization_scheme == "int8_asym_w_sym") {
      cfg->quantization_scheme = "CSINN_QUANT_INT8_ASYM_W_SYM";
    } else if (opt_cfg->quantization_scheme == "int16_sym") {
      cfg->quantization_scheme = "CSINN_QUANT_INT16_SYM";
    } else if (opt_cfg->quantization_scheme == "float16") {
      cfg->quantization_scheme = "CSINN_QUANT_FLOAT16";
    } else if (opt_cfg->quantization_scheme == "bfloat16") {
      cfg->quantization_scheme = "CSINN_QUANT_BFLOAT16";
    } else if (opt_cfg->quantization_scheme == "unset") {
      cfg->quantization_scheme = "unset";
    } else {
      LOG(WARNING) << "Unsupport quantization scheme.";
    }

    auto base_dtype = cfg->dtype_weight;

    if (base_dtype == "int4_t") {
      base_dtype_ = "CSINN_DTYPE_INT4";
    } else if (base_dtype == "uint8_t") {
      base_dtype_ = "CSINN_DTYPE_UINT8";
    } else if (base_dtype == "int8_t") {
      base_dtype_ = "CSINN_DTYPE_INT8";
    } else if (base_dtype == "float16") {
      base_dtype_ = "CSINN_DTYPE_FLOAT16";
    } else if (base_dtype == "bfloat16") {
      base_dtype_ = "CSINN_DTYPE_BFLOAT16";
    } else if (base_dtype == "float") {
      base_dtype_ = "CSINN_DTYPE_FLOAT32";
    } else if (base_dtype == "int16_t") {
      base_dtype_ = "CSINN_DTYPE_INT16";
    } else {
      base_dtype_ = "CSINN_DTYPE_FLOAT32";
    }

    cfg->fuse_zp2bias = opt_cfg->fuse_zp2bias;
    cfg->nbit_input = opt_cfg->nbit_input;
    cfg->nbit_weight = opt_cfg->nbit_weight;
    cfg->nbit_activation = opt_cfg->nbit_activation;
    cfg->weight_quantized_type = opt_cfg->weight_quantized_type;
    cfg->activate_quantized_type = opt_cfg->activate_quantized_type;

    hybrid_cfg = new QConfig_();
    __update_hybrid_quantization(hybrid_cfg, hybrid_quantization_scheme);
  }

  virtual ~CodegenCSINN() {}

  virtual string JIT(const std::vector<Output>& out) { return "Unsupport\n"; }
  virtual void VisitExpr(const Expr& expr);
  virtual void VisitExpr_(const VarNode* node);
  virtual void VisitExpr_(const ConstantNode* node);
  virtual void VisitExpr_(const TupleNode* op);
  virtual void VisitExpr_(const CallNode* call);

  virtual string EmitGraph(void);

  virtual void SetDim(string name, std::vector<int> shape);
  virtual void CreateConstantTensorBase(string name, size_t size, std::vector<int> shape,
                                        string target_dtype, string layout);
  // for common constant
  virtual void CreateConstantTensor(CSIConstant* data, string name, std::vector<int> shape,
                                    string target_dtype, QuantParams quant_params,
                                    bool depthwise_kernel = false, bool is_bias = false);
  virtual void CreateHybridConstantTensor(CSIConstant* data, string name, std::vector<int> shape,
                                          Array<Array<IndexExpr>> q_params, int params_index,
                                          bool is_layer_hybrid, bool depthwise_kernel = false);
  // for bias
  virtual void CreateConstantTensor(CSIConstant* data, string name, std::vector<int> shape,
                                    string target_dtype, QuantParams input_quant_params,
                                    QuantParams kernel_quant_params, QuantParams bias_quant_params);
  // virtual void CreateHybridConstantTensor(CSIConstant* data, string name, std::vector<int> shape,
  //                                   Array<Array<IndexExpr>> q_params);

  virtual void CreateBiasTensor(const CallNode* call, CSIConstant* data, string name,
                                Array<Array<IndexExpr>> q_params, bool* fuse_zp,
                                bool is_layer_hybrid, bool is_input_hybrid, bool is_wei_hybrid,
                                bool depthwise_kernel = false);

  virtual void CreateTensor(string name, string data, std::vector<int> shape,
                            QuantParams quant_params, string dtype);
  virtual void CreateMallocBuf(string name, std::vector<int> shape, string dtype) = 0;
  virtual void CreateTensorSessData() = 0;
  virtual void CreateHybridTensorSessData(std::vector<int> shape, string dtype) = 0;
  virtual void CreateGraphTensor(QuantParams q_params);

  virtual Output GetRealInput(const CallNode* call);
  virtual Output GetRealInput(const VarNode* var);
  virtual void PushInput(string name, const CallNode* call);

  virtual string InputTensor(std::ostringstream& decl, const CallNode* call, int input_index,
                             QuantParams quant_params, string dtype);
  virtual string HybridInputTensor(std::ostringstream& decl, const CallNode* call, int input_index,
                                   Array<Array<IndexExpr>> q_params, int params_index,
                                   bool is_layer_hybrid);
  virtual string InputTensorName(const CallNode* call, int input_index, QuantParams quant_params,
                                 string dtype);
  virtual string InputTensorVar(const VarNode* call, int input_index, QuantParams quant_params,
                                string dtype);
  virtual string InputTensorCall(const CallNode* call, int input_index, QuantParams quant_params,
                                 string dtype);
  virtual string InputTensorTupleItem(const TupleGetItemNode* call, QuantParams quant_params,
                                      string dtype);

  virtual string InsertDataConvert(const CallNode* call, int input_index, string input_name,
                                   QuantParams quant_params, string dtype);

  virtual void DumpConstant();
  virtual void SessionRunMode() {}
  virtual void ModelBinarySave() {}
  virtual void malloc_buf(string out, int out_size) = 0;
  virtual bool InOpList(const CallNode* call);
  virtual string OutputTensor(std::ostringstream& decl, const CallNode* call,
                              QuantParams quant_params, string dtype);
  virtual string HybridOutputTensor(std::ostringstream& decl, const CallNode* call,
                                    Array<Array<IndexExpr>> q_params, int params_index,
                                    bool is_layer_hybrid);
  virtual string DataConvertTensor(std::vector<int> shape, QuantParams quant_params, string dtype);

  virtual void PushOutput(string name, const CallNode* call, string dtype = "");
  virtual void PushOutput(std::vector<string> names, const CallNode* call);
  virtual int CheckOutput(const CallNode* call);
  template <typename T>
  void SisoOp(std::ostringstream& decl_stream, const CallNode* call, const T* attr,
              string op_name = "");
  virtual QuantParams* GetQuantParamsBase(float min_value, float max_value, int32_t tensor_type,
                                          QConfig_* quantize_cfg = NULL);
  virtual QuantParams* GetQuantParamsBase(float scale, int32_t zp);
  virtual QuantParams* GetQuantParamsBase(float scale, int32_t zp, float min_value,
                                          float max_value);
  virtual QuantParams* GetQuantParams(Array<Array<IndexExpr>> q_params,
                                      QConfig_* quantize_cfg = NULL);

  virtual void GetAsymScale(float min_value, float max_value, int bits, Qinfo* qinfo);
  virtual void GetSymScale(float min_value, float max_value, int valid_range, Qinfo* qinfo);

  virtual QuantParams* GetIntegralQuantParams(QuantParams* q_params, int32_t tensor_type,
                                              QConfig_* quantize_cfg = NULL);

  virtual string replace(string a);

  virtual void AvgPool2d(const CallNode* call);
  virtual void AvgPool3d(const CallNode* call);
  virtual void BatchToSpaceND(const CallNode* call);
  virtual void BroadCastTo(const CallNode* call);
  virtual void Clip(const CallNode* call);
  virtual void CacheMatMul(const CallNode* call);
  virtual void CacheConv1d(const CallNode* call);
  virtual void Concat(const CallNode* call);
  virtual void Conv1d(const CallNode* call);
  virtual void Conv2d(const CallNode* call, string op_name);
  virtual void Conv3d(const CallNode* call);
  virtual void CropResize(const CallNode* call);
  virtual void DeConv2d(const CallNode* call);
  virtual void DeConv3d(const CallNode* call);
  virtual void Dense(const CallNode* call);
  virtual void DepthToSpace(const CallNode* call);
  virtual void Dilation2d(const CallNode* call);
  virtual void DisoOp(const CallNode* call, string op_name, string out_dtype = "");
  virtual void ExpandDims(const CallNode* call);
  virtual void Flatten(const CallNode* call);
  virtual void Fsmn(const CallNode* call);
  virtual void Full(const CallNode* call);
  virtual void GlobalAvgPool2d(const CallNode* call);
  virtual void GlobalMaxPool2d(const CallNode* call);
  virtual void LayerNorm(const CallNode* call);
  virtual void LRN(const CallNode* call);
  virtual void LeakyRelu(const CallNode* call);
  virtual void LogSoftmax(const CallNode* call);
  virtual void MatMul(const CallNode* call);
  virtual void MaxPool2d(const CallNode* call);
  virtual void MaxPool2dLocat(const CallNode* call);
  virtual void MaxPool3d(const CallNode* call);
  virtual void Maxpool2dWithArgmax(const CallNode* call);
  virtual void PRelu(const CallNode* call);
  virtual void PSROIPool(const CallNode* call);
  virtual void Pad(const CallNode* call);
  virtual void Proposal(const CallNode* call);
  virtual void ROIPool(const CallNode* call);
  virtual void Reduce(const CallNode* call, string name, string out_dtype);
  virtual void Relu(const CallNode* call);
  virtual void Relu6(const CallNode* call);
  virtual void Reshape(const CallNode* call);
  virtual void Reverse(const CallNode* call);
  virtual void ScatterND(const CallNode* call);
  virtual void Segment(const CallNode* call, string name);
  virtual void Sigmoid(const CallNode* call);
  virtual void Softmax(const CallNode* call);
  virtual void SpaceToBatchND(const CallNode* call);
  virtual void SpaceToDepth(const CallNode* call);
  virtual void Split(const CallNode* call);
  virtual void Squeeze(const CallNode* call);
  virtual void StridedSlice(const CallNode* call);
  virtual void Take(const CallNode* call);
  virtual void Tile(const CallNode* call);
  virtual void Transpose(const CallNode* call);
  virtual void UnPool2d(const CallNode* call);
  virtual void Unary(const CallNode* call, string name);
  virtual void UpSampling(const CallNode* call);
  virtual void setup_callback(std::ostringstream& decl, string op_name, string prams_name);
  virtual void params_common_setup(std::ostringstream& decl, const CallNode* call, string op_name,
                                   string params_name, string layer_name, string layout);
  virtual void params_hybrid_setup(std::ostringstream& decl, string op_name, std::vector<int> shape,
                                   string params_name, string layer_name, string dtype,
                                   string layout = "CSINN_LAYOUT_NCHW");

  virtual std::shared_ptr<std::vector<float>> FuseZpToBias(const CallNode* call,
                                                           QuantParams* q_params,
                                                           bool is_depthwise);

  virtual CSIConstant* CastParams(CSIConstant* data, string target_dtype, QuantParams quant_params,
                                  bool depthwise_kernel);
  virtual CSIConstant* CastParams(CSIConstant* data, string target_dtype,
                                  QuantParams total_input_quant, QuantParams kernel_quant_params);
  virtual void Axis0Cast(CSIConstant* data, CSIConstant* output, Qinfo* q_infos,
                         string target_dtype, int q_size, int inner_size);
  virtual void Axis3Cast(CSIConstant* data, CSIConstant* output, Qinfo* q_infos,
                         string target_dtype, int q_size, int inner_size);

  template <typename T>
  void SetupConv1dParams(string name, const T* attr);

  template <typename T>
  void SetupConv2dParams(string name, const T* attr);

  template <typename T>
  void SetupConv3dParams(string name, const T* attr);

  template <typename T>
  void SetupDilation2dParams(string name, const T* attr);

  template <typename T>
  void SetupPadding(string name, const T* attr);

  template <typename T>
  void Setup1dPadding(string name, const T* attr);

  template <typename T>
  void SetupPoolParams(string name, const T* attr);

  template <typename T>
  void SetupPool3DParams(string name, const T* attr);

  virtual void malloc_params(string struct_name, string params_name);

  virtual void EmitVersion(void);
  virtual void EmitHeader(void);
  virtual void EmitSessionSetup(void);
  virtual void EmitSessionRun(void);
  virtual void EmitNBGSetup(void);
  virtual void FreeTensor(const Expr& expr, string name);

  virtual Output* GetOutput(string name);

  std::unordered_map<const Object*, size_t> layer_count;
  std::map<string, string> tensor_data;

 protected:
  string siso_input_name;
  std::vector<Output> output_list_;
  bool first_visit_expr{true};
  std::vector<const tvm::relay::CallNode*> real_out;
  /*! \brief statement of the function that will be compiled using CSINN kernels. */
  std::vector<string> ext_func_body;
  /*! \brief The arguments used by a wrapped function that calls CSINN kernels. */
  Array<Var> ext_func_args_;

  string layout_{""};
  string base_dtype_{""};
  string target_name_{""};
  std::vector<string> target_op_list;

  /*! \brief The declaration of intermeidate buffers. */
  std::vector<string> buf_decl_;
  std::map<string, QuantParams> io_nodes;
  QConfig_* cfg;
  QConfig_* hybrid_cfg;

  string debug_level_{""};
  int buf_idx_{0};
  int alloc_idx_{0};
  int layer_index_{0};
  int params_idx_{0};

  string output_dir_{"."};

  bool multithread{false};
  string model_save{""};
  string trace_strategy_{"normal"};

  std::vector<int> input_memory_type;
  std::vector<int> output_memory_type;

  /*! \brief Hybird quantization buffers. */
  string hybrid_quantization_scheme{"unset"};
  std::vector<string> hybrid_layer_name;
  std::map<string, string> output2params;

  std::vector<string> hybrid_buffer_name_;

  /*! \brief The name of the the outputs. */
  std::vector<Output> out_;

  /*! \brief The name of the the constant. */
  std::vector<CSIConstant> constant_;

  std::vector<CSIConstant> constant_list_;

  std::vector<QuantParams> qinfo_list_;

  std::vector<int> flag_list_;

  /*! \brief The id of the external csinn ext_func. */
  string ext_func_id_{""};

  void EnterScope() { indent_ += 2; }

  void ExitScope() {
    CHECK_GE(indent_, 2U) << "Wrong ident found.";
    indent_ -= 2;
  }

  void __update_hybrid_quantization(QConfig_* config, const string& hybrid_quantization_scheme) {
    if (hybrid_quantization_scheme == "int4_asym_w_sym") {
      config->dtype_input = "int4_t";
      config->dtype_weight = "int4_t";
      config->dtype_activation = "int32_t";
      config->quantization_scheme = "CSINN_QUANT_INT4_ASYM_W_SYM";

      config->nbit_input = 4;
      config->nbit_weight = 4;
      config->nbit_activation = 32;
      config->weight_quantized_type = "sym";
      config->activate_quantized_type = "asym";
    } else if (hybrid_quantization_scheme == "uint8_asym") {
      config->dtype_input = "uint8_t";
      config->dtype_weight = "uint8_t";
      config->dtype_activation = "int32_t";
      config->quantization_scheme = "CSINN_QUANT_UINT8_ASYM";

      config->nbit_input = 8;
      config->nbit_weight = 8;
      config->nbit_activation = 32;
      config->weight_quantized_type = "asym";
      config->activate_quantized_type = "asym";
    } else if (hybrid_quantization_scheme == "int8_sym" ||
               hybrid_quantization_scheme == "int8_original") {
      config->dtype_input = "int8_t";
      config->dtype_weight = "int8_t";
      config->dtype_activation = "int32_t";
      config->quantization_scheme = "CSINN_QUANT_INT8_SYM";
      if (hybrid_quantization_scheme == "int8_original") {
        config->quantization_scheme = "CSINN_QUANT_INT8_ORIGINAL";
      }

      config->nbit_input = 8;
      config->nbit_weight = 8;
      config->nbit_activation = 32;
      config->weight_quantized_type = "sym";
      config->activate_quantized_type = "sym";
    } else if (hybrid_quantization_scheme == "int8_asym_w_sym") {
      config->dtype_input = "int8_t";
      config->dtype_weight = "int8_t";
      config->dtype_activation = "int32_t";
      config->quantization_scheme = "CSINN_QUANT_INT8_ASYM_W_SYM";

      config->nbit_input = 8;
      config->nbit_weight = 8;
      config->nbit_activation = 32;
      config->weight_quantized_type = "sym";
      config->activate_quantized_type = "asym";
    } else if (hybrid_quantization_scheme == "int8_asym") {
      config->dtype_input = "int8_t";
      config->dtype_weight = "int8_t";
      config->dtype_activation = "int32_t";
      config->quantization_scheme = "CSINN_QUANT_INT8_ASYM";

      config->nbit_input = 8;
      config->nbit_weight = 8;
      config->nbit_activation = 32;
      config->weight_quantized_type = "asym";
      config->activate_quantized_type = "asym";
    } else if (hybrid_quantization_scheme == "int16_sym") {
      config->dtype_input = "int16_t";
      config->dtype_weight = "int16_t";
      config->dtype_activation = "int32_t";
      config->quantization_scheme = "CSINN_QUANT_INT16_SYM";

      config->nbit_input = 16;
      config->nbit_weight = 16;
      config->nbit_activation = 32;
      config->weight_quantized_type = "sym";
      config->activate_quantized_type = "sym";
    } else if (hybrid_quantization_scheme == "float16") {
      config->dtype_input = "int16_t";
      config->dtype_weight = "float16";
      config->dtype_activation = "float16";
      config->quantization_scheme = "CSINN_QUANT_FLOAT16";

      config->nbit_input = 16;
      config->nbit_weight = 16;
      config->nbit_activation = 16;
      config->weight_quantized_type = "sym";
      config->activate_quantized_type = "sym";
    } else if (hybrid_quantization_scheme == "bfloat16") {
      config->dtype_input = "int16_t";
      config->dtype_weight = "bfloat16";
      config->dtype_activation = "bfloat16";
      config->quantization_scheme = "CSINN_QUANT_BFLOAT16";

      config->nbit_input = 16;
      config->nbit_weight = 16;
      config->nbit_activation = 16;
      config->weight_quantized_type = "sym";
      config->activate_quantized_type = "sym";
    } else if (hybrid_quantization_scheme == "unset") {
      config->quantization_scheme = "unset";
    } else {
      LOG(WARNING) << "Unsupport quantization scheme.";
    }
  }

  string GetCSINNDtype(string dtype) {
    string csi_dtype;
    if (dtype == "int4_t") {
      csi_dtype = "CSINN_DTYPE_INT4";
    } else if (dtype == "uint8_t" || dtype == "uint8") {
      csi_dtype = "CSINN_DTYPE_UINT8";
    } else if (dtype == "int8_t" || dtype == "int8") {
      csi_dtype = "CSINN_DTYPE_INT8";
    } else if (dtype == "float" || dtype == "float32") {
      csi_dtype = "CSINN_DTYPE_FLOAT32";
    } else if (dtype == "float16") {
      csi_dtype = "CSINN_DTYPE_FLOAT16";
    } else if (dtype == "bfloat16") {
      csi_dtype = "CSINN_DTYPE_BFLOAT16";
    } else if (dtype == "int32_t" || dtype == "int32") {
      csi_dtype = "CSINN_DTYPE_INT32";
    } else if (dtype == "bool") {
      csi_dtype = "CSINN_DTYPE_BOOL";
    } else if (dtype == "int16_t" || dtype == "int16") {
      csi_dtype = "CSINN_DTYPE_INT16";
    } else {
      LOG(FATAL) << "Unsupported dtype " << dtype;
    }
    return csi_dtype;
  }

  string GetCSINNActLayout(std::vector<int> shape) {
    string csi_layout;
    if (shape.size() == 1 || shape.size() == 0) {
      csi_layout = "CSINN_LAYOUT_N";
    } else if (shape.size() == 2) {
      csi_layout = "CSINN_LAYOUT_NC";
    } else if (shape.size() == 3) {
      if (layout_ == "NCHW") csi_layout = "CSINN_LAYOUT_NCW";
      if (layout_ == "NHWC") csi_layout = "CSINN_LAYOUT_NWC";
    } else if (shape.size() == 4) {
      if (layout_ == "NCHW") csi_layout = "CSINN_LAYOUT_NCHW";
      if (layout_ == "NHWC") csi_layout = "CSINN_LAYOUT_NHWC";
    } else if (shape.size() == 5) {
      if (layout_ == "NCHW") csi_layout = "CSINN_LAYOUT_NCDHW";
      if (layout_ == "NHWC") csi_layout = "CSINN_LAYOUT_NDHWC";
    } else {
      LOG(FATAL) << "Unsupported shape size " << shape.size();
    }
    return csi_layout;
  }

  string GetCSINNWeightLayout(std::vector<int> shape) {
    string csi_layout;
    if (shape.size() == 0) {
      csi_layout = "CSINN_LAYOUT_NULL";
    } else if (shape.size() == 1) {
      csi_layout = "CSINN_LAYOUT_O";
    } else if (shape.size() == 2) {
      csi_layout = "CSINN_LAYOUT_OI";
    } else if (shape.size() == 3) {
      if (layout_ == "NCHW") csi_layout = "CSINN_LAYOUT_OIW";
      if (layout_ == "NHWC") csi_layout = "CSINN_LAYOUT_OWI";
    } else if (shape.size() == 4) {
      if (layout_ == "NCHW") csi_layout = "CSINN_LAYOUT_OIHW";
      if (layout_ == "NHWC") csi_layout = "CSINN_LAYOUT_OHWI";
    } else if (shape.size() == 5) {
      if (layout_ == "NCHW") csi_layout = "CSINN_LAYOUT_OIDHW";
      if (layout_ == "NHWC") csi_layout = "CSINN_LAYOUT_ODHWI";
    } else {
      LOG(FATAL) << "Unsupported shape size " << shape.size();
    }
    return csi_layout;
  }

  string GetCSINNMemoryType(int type) {
    string mtype;
    if (type == 0) {
      mtype = "CSINN_MEM_TYPE_CPU_NOT_ALIGNED";
    } else if (type == 1) {
      mtype = "CSINN_MEM_TYPE_CPU_ALIGNED";
    } else if (type == 2) {
      mtype = "CSINN_MEM_TYPE_DMABUF";
    } else {
      LOG(FATAL) << "Unsupported memory type " << mtype;
    }
    return mtype;
  }

  void PrintIndents(std::ostringstream& stream) {
    for (int i = 0; i < indent_; i++) {
      stream << ' ';
    }
  }

  float* GetFloatData(CSIConstant* data) {
    string type_str = data->dtype;
    float* out = NULL;
    if (type_str == "float") {
      uint size = data->size;
      out = reinterpret_cast<float*>(malloc(size));
      std::memcpy(out, data->data_buf, size);
    } else if (type_str == "int64_t") {
      int64_t* data_val = reinterpret_cast<int64_t*>(data->data_buf);
      uint size = data->size / 8;
      out = reinterpret_cast<float*>(malloc(size * 4));
      for (uint i = 0; i < size; i++) {
        out[i] = static_cast<float>(data_val[i]);
      }
    } else if (type_str == "uint8_t") {
      uint8_t* data_val = reinterpret_cast<uint8_t*>(data->data_buf);
      uint size = data->size;
      out = reinterpret_cast<float*>(malloc(size * 4));
      for (uint i = 0; i < size; i++) {
        out[i] = static_cast<float>(data_val[i]);
      }
    } else if (type_str == "int8_t") {
      int8_t* data_val = reinterpret_cast<int8_t*>(data->data_buf);
      uint size = data->size;
      out = reinterpret_cast<float*>(malloc(size * 4));
      for (uint i = 0; i < size; i++) {
        out[i] = static_cast<float>(data_val[i]);
      }
    } else {
      LOG(ERROR) << "get error dtype:" << type_str;
    }
    return out;
  }

  void PrintOneLine(std::ostringstream& stream, string str) {
    PrintIndents(stream);
    stream << str << "\n";
  }

  void PrintOneLine(std::ostringstream& stream, std::ostringstream& str) {
    PrintOneLine(stream, str.str());
    str.str("");
  }

  void PrintNewLine(std::ostringstream& stream) { stream << "\n"; }

  void PushDeclLine(std::ostringstream& decl) {
    decl << ";";
    buf_decl_.push_back(decl.str());
    decl.str("");
  }

  void end_stream(std::ostringstream& decl, string name) {
    std::ostringstream func;
    func << "csi_" << name << decl.str() << ";";

    ext_func_body.push_back(func.str());
    buf_idx_++;
  }

  /*
   * \brief Convert FP32 representation into fixed point representation.
   * \param double_multplier The input FP32 number.
   * \return The pair of multiplier and shift for fixed point representation.
   * \note Converts a floating point number so that it can be represented by
   *       integers. The representation is
   *             float_number = (significand) * 2^(exponent)
   *
   *       The significand is a number between 0.5 and 1. This is represented by
   *       an integer number. For example, if it is int32, then the decimal point
   *       exists between bit 31 and 30 from LSB (or between first and second bit
   *       from the left).
   *
   *       Some examples are
   *           0.25 = (0.5) * 2^(-1)
   *           0.125 = (0.5) * 2^(-2)
   *
   *       Credit to TFLite reference implementation.
   */
  void GetMultiplierAndShift(double double_multiplier, int32_t* multiplier, int32_t* shift) {
    int32_t significand, exponent;
    if (double_multiplier == 0) {
      *multiplier = 0;
      *shift = 0;
      return;
    }

    // Get the significand and exponent.
    double significand_d = frexp(double_multiplier, &exponent);

    // Convert the double significand to int significand, i.e., convert into a
    // integer where the decimal point is between bit 31 and 30. This is done by
    // multiplying the double value with 2^31 and then casting to int.
    significand_d = std::round(significand_d * (1ll << 31));
    int64_t significand_int64 = significand_d;
    if (significand_int64 == (1ll << 31)) {
      significand_int64 /= 2;
      ++exponent;
    }
    significand = significand_int64;
    *multiplier = significand;
    *shift = exponent;
  }

  string double_to_string(double value) {
    std::stringstream ss;
    ss << std::setprecision(15) << value;
    string str = ss.str();
    return str;
  }

  int16_t float32_to_float16(float value) {
    int16_t ret;

    if (value > -6.1e-5 && value < 6.1e-5) {
      return 0;
    }
    int32_t* org_format_addr = reinterpret_cast<int32_t*>(&value);
    int32_t org_format = *org_format_addr;
    int16_t sign = (org_format & 0x80000000) >> 16;
    int16_t frac = (org_format & 0x7fffff) >> 13;
    int16_t exp = (((((org_format >> 23) & 0xff) - 128) + 16) & 0x1f) << 10;
    ret = sign | frac | exp;
    return ret;
  }

  int16_t float32_to_bfloat16(float value) {
    int16_t ret = 0;
    int32_t* org_format_addr = reinterpret_cast<int32_t*>(&value);
    int32_t org_format = *org_format_addr;
    ret = (org_format & 0xffff0000) >> 16;
    return ret;
  }

  int __get_stride(int pos, std::vector<int> data_shape) {
    int size = 1;
    for (uint i = pos + 1; i < data_shape.size(); i++) {
      size *= data_shape[i];
    }
    return size;
  }

  std::vector<int> __get_real_axis(int ndim, Array<Integer> axis) {
    std::vector<int> real_axis;
    for (uint i = 0; i < axis.size(); i++) {
      int ele = axis[i].as<IntImmNode>()->value;
      if (ele < 0) {
        ele += ndim;
      }
      if (ele >= ndim) {
        std::ostringstream tmp_stream;
        for (uint j = 0; j < axis.size(); j++) {
          tmp_stream << to_string(axis[j].as<IntImmNode>()->value) << " ";
        }
        LOG(FATAL) << to_string(ele) << " exceeds the maximum dimension " << to_string(ndim)
                   << " . Received axis=[ " << tmp_stream.str() << "]";
      }
      real_axis.push_back(ele);
    }
    sort(real_axis.begin(), real_axis.end());
    std::vector<int> out;
    for (uint i = 0; i < real_axis.size(); i++) {
      int ele = real_axis[i];
      int flag = 1;
      for (uint j = 0; j < out.size(); j++) {
        int tmp = out[j];
        if (ele == tmp) {
          flag = 0;
        }
      }
      if (flag) {
        out.push_back(ele);
      }
    }
    return out;
  }

  string dirnameOf(const string& filename) {
    size_t pos = filename.find_last_of("\\/");

    return (string::npos == pos) ? "." : filename.substr(0, pos);
  }

  std::vector<int> __convert_list(Array<Integer> data) {
    std::vector<int> out_list;
    for (size_t i = 0; i < data.size(); i++) {
      const auto* intImm = data[i].as<IntImmNode>();
      out_list.push_back(static_cast<int>(intImm->value));
    }
    return out_list;
  }

  std::vector<string> __convert_string_list(Array<String> data) {
    std::vector<string> out_list;
    for (size_t i = 0; i < data.size(); i++) {
      out_list.push_back(data[i]);
    }
    return out_list;
  }

  bool is_depthwise(const std::vector<int>& ishape, const std::vector<int>& kshape, int group,
                    string target_layout) {
    bool is_d = false;
    if (target_layout == "NCHW" && kshape[1] == 1 && group == ishape[1]) {
      is_d = true;
    } else if (target_layout == "NHWC" && kshape[0] == 1 && group == ishape[3]) {
      is_d = true;
    }
    return is_d;
  }

  Array<Array<IndexExpr>> get_quant_params_expr(Array<Array<IndexExpr>> q_params, int index) {
    Array<Array<IndexExpr>> new_params;
    new_params.push_back(q_params[index]);

    return new_params;
  }

  template <typename T>
  bool is_contain_item(std::vector<T> arr, T target_item) {
    for (auto item : arr) {
      if (item == target_item) {
        return true;
      }
    }
    return false;
  }

  string get_complete_layer_name(string op_name, string ori_layer_name) {
    string res = op_name + "_" + ori_layer_name + "_" + std::to_string(params_idx_);
    return res;
  }

 private:
  string target_{""};

  int const_idx_{0};

  std::vector<Output> out_list_;

  string params_path_;

  int indent_{0};

 protected:
  size_t constant_offset{0};
  size_t qinfo_offset{0};
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_CSINN_H_
