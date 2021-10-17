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
 * \file src/relay/backend/contrib/csinn/codegen.h
 * \brief The base class for external codegen tools.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_CODEGEN_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_CODEGEN_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>
#include <tvm/relay/op.h>
#include <tvm/runtime/container.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../codegen_c/codegen_c.h"
using std::string;
using std::to_string;

namespace tvm {
namespace relay {
namespace contrib {

struct CSIConstant {
  string name;
  string dtype;
  size_t size;
  uint8_t* data_buf;
};

class CodegenCSINN : public ExprVisitor, public CodegenCBase {
 public:
  CodegenCSINN(const string& id, const string& layout, const string& target, const string& path) {
    this->ext_func_id_ = id;
    this->layout_ = layout;
    this->target_ = target;
    this->params_path_ = path;
  }
  virtual ~CodegenCSINN() {}

  virtual void VisitExpr_(const VarNode* node);
  virtual void VisitExpr_(const ConstantNode* node);
  virtual void VisitExpr_(const TupleNode* op);
  virtual void VisitExpr_(const CallNode* call);
  virtual void GenerateBackendCFunc(const string& func_name, const Array<Var>& args,
                                    const Output& out);
  virtual string JitImpl(const string& ext_func_id, const Array<Var>& args,
                         const std::vector<string>& buf_decl, const std::vector<string>& body,
                         const std::vector<Output>& out);
  virtual string JIT(const std::vector<Output>& out);
  virtual string JIT(void);

  virtual void SetDim(string name, std::vector<int> shape);
  virtual void CreateConstantTensor(string name, size_t size, std::vector<int> shape);
  virtual void CreateConstantTensor(string name, size_t size, std::vector<int> shape,
                                    int32_t zero_point, double scale);

  virtual void CreateTensor(string name, string data, std::vector<int> shape);
  virtual void CreateTensor(string name, string data, std::vector<int> shape, int32_t zero_point,
                            double scale);
  virtual void CreateTensor(string name, string data, std::vector<int> shape, int32_t zero_point,
                            double scale, double fix_scale);
  virtual Output GetRealInput(const CallNode* call);
  virtual Output GetRealInput(const VarNode* var);
  virtual void PushInput(string name, const CallNode* call);
  virtual void InputTensor(std::ostringstream& decl, const CallNode* call, int input_index);
  virtual string InputTensor(std::ostringstream& decl, const CallNode* call, int input_index,
                             int32_t zero_point, double scale);

  virtual string OutputTensor(std::ostringstream& decl, const CallNode* call);
  virtual string OutputTensor(std::ostringstream& decl, const CallNode* call, int32_t zero_point,
                              double scale);
  template <typename T>
  string OutputTensor(std::ostringstream& decl, const CallNode* call, const T* attr,
                      int32_t zero_point, double scale);
  virtual string OutputTensor(std::ostringstream& decl, const CallNode* call, int32_t zero_point,
                              double scale, double fix_scale);

  virtual void DumpConstant();
  virtual void InputMultiplier(string input, double scale);
  virtual void PushOutput(string name, const CallNode* call, bool push_output = false);
  virtual void PushOutput(std::vector<string> names, const CallNode* call);

  template <typename T>
  void SisoOpU8(std::ostringstream& decl_stream, const CallNode* call, const T* attr);
  virtual void UnaryU8(const CallNode* call, string name);
  virtual void CSINNInit(const CallNode* call);
  virtual void CSINNDeinit(const CallNode* call);
  virtual void DisoOpU8(const CallNode* call, string op_name);
  virtual void Conv2dU8(const CallNode* call);
  virtual void Conv3dU8(const CallNode* call);
  virtual void Conv2dReluU8(const CallNode* call);
  virtual void Conv2dRelu6U8(const CallNode* call);
  virtual void DeConv2dU8(const CallNode* call);
  virtual void DeConv3dU8(const CallNode* call);
  virtual void DenseU8(const CallNode* call);
  virtual void SoftmaxU8(const CallNode* call);
  virtual void LogSoftmaxU8(const CallNode* call);
  virtual void MaxPool2dU8(const CallNode* call);
  virtual void AvgPool2dU8(const CallNode* call);
  virtual void GlobalAvgPool2dU8(const CallNode* call);
  virtual void GlobalMaxPool2dU8(const CallNode* call);
  virtual void Maxpool2dWithArgmaxU8(const CallNode* call);
  virtual void MaxPool2dLocatU8(const CallNode* call);
  virtual void UnPool2dU8(const CallNode* call);
  virtual void PSROIPoolU8(const CallNode* call);
  virtual void ROIPoolU8(const CallNode* call);
  virtual void ProposalU8(const CallNode* call);
  virtual void UpSamplingU8(const CallNode* call);
  virtual void ReluU8(const CallNode* call);
  virtual void Relu6U8(const CallNode* call);
  virtual void PReluU8(const CallNode* call);
  virtual void LeakyReluU8(const CallNode* call);
  virtual void ConcatU8(const CallNode* call);
  virtual void LRNU8(const CallNode* call);
  virtual void FlattenU8(const CallNode* call);
  virtual void SigmoidU8(const CallNode* call);
  virtual void TransposeU8(const CallNode* call);
  virtual void ReshapeU8(const CallNode* call);
  virtual void SqueezeU8(const CallNode* call);
  virtual void SplitU8(const CallNode* call);
  virtual void StridedSliceU8(const CallNode* call);
  virtual void ReverseU8(const CallNode* call);
  virtual void SegmentU8(const CallNode* call, string name);
  virtual void AvgPool3dU8(const CallNode* call);
  virtual void MaxPool3dU8(const CallNode* call);
  virtual void ExpandDimsU8(const CallNode* call);
  virtual void BroadCastToU8(const CallNode* call);
  virtual void FullU8(const CallNode* call);
  virtual void PadU8(const CallNode* call);
  virtual void TakeU8(const CallNode* call);
  virtual void ClipU8(const CallNode* call);
  virtual void BNU8(const CallNode* call);
  virtual void TileU8(const CallNode* call);
  virtual void Dilation2dU8(const CallNode* call);
  virtual void CropResizeU8(const CallNode* call);
  virtual void DepthToSpaceU8(const CallNode* call);
  virtual void SpaceToDepthU8(const CallNode* call);
  virtual void Conv2dChannelU8(const CallNode* call);
  virtual void Conv2dReluChannelU8(const CallNode* call);
  virtual void Conv2dRelu6ChannelU8(const CallNode* call);
  virtual void ReduceU8(const CallNode* call, string name);
  virtual void setup_callback(std::ostringstream& decl, string op_name, string prams_name);
  virtual void params_common_setup(std::ostringstream& decl, string op_name, string params_name);
  template <typename T>
  void SetupConv2dParams(string name, const T* attr);

  template <typename T>
  void SetupConv3dParams(string name, const T* attr);

  template <typename T>
  void SetupDilation2dParams(string name, const T* attr);

  template <typename T>
  void SetupPadding(string name, const T* attr);

  template <typename T>
  void SetupPoolParams(string name, const T* attr);

  template <typename T>
  void SetupPool3DParams(string name, const T* attr);

 protected:
  std::vector<Output> output_list_;
  /*! \brief statement of the function that will be compiled using CSINN kernels. */
  std::vector<string> ext_func_body;
  /*! \brief The arguments used by a wrapped function that calls CSINN kernels. */
  Array<Var> ext_func_args_;

  /*! \brief The declaration of intermeidate buffers. */
  std::vector<string> buf_decl_;

  int buf_idx_{0};
  int layer_index_{0};

  /*! \brief The name of the the outputs. */
  std::vector<Output> out_;

  /*! \brief The name of the the constant. */
  std::vector<CSIConstant> constant_;

  /*! \brief The id of the external csinn ext_func. */
  string ext_func_id_{""};

  std::vector<const CallNode*> call_list_;

  void EnterScope() { indent_ += 2; }

  void ExitScope() {
    CHECK_GE(indent_, 2U) << "Wrong ident found.";
    indent_ -= 2;
  }

  void PrintIndents(std::ostringstream& stream) {
    for (int i = 0; i < indent_; i++) {
      stream << ' ';
    }
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

 private:
  virtual void malloc_buf(string out, int out_size);

  string layout_{""};
  string target_{""};

  int alloc_idx_{0};
  int const_idx_{0};

  std::vector<Output> out_list_;

  std::vector<CSIConstant> constant_list_;
  size_t constant_offset{0};
  string params_path_;

  int indent_{0};

  bool first_visit_expr{true};
};

class CodegenAnole : public CodegenCSINN {
 public:
  CodegenAnole(const string& id, const string& layout, const string& target, const string& path)
      : CodegenCSINN(id, layout, target, path) {}
  virtual ~CodegenAnole() {}
  using CodegenCSINN::JitImpl;
  virtual string JitImpl(const string& ext_func_id, const Array<Var>& args,
                         const std::vector<string>& buf_decl, const std::vector<string>& body,
                         const std::vector<string>& ovx, const std::vector<Output>& out);
  virtual string JIT(const std::vector<Output>& out);
  virtual string JIT(void);
  virtual void VisitExpr_(const CallNode* call);
  void InputMultiplier(string input, double scale) {}

  virtual void CreateTensor(string name, string data, std::vector<int> shape);
  virtual void CreateTensor(string name, string data, std::vector<int> shape, int32_t zero_point,
                            double scale);
  virtual void CreateConstantTensor(string name, size_t size, std::vector<int> shape,
                                    int32_t zero_point, double scale);
  virtual string OutputTensor(std::ostringstream& decl, const CallNode* call, int32_t zero_point,
                              double scale);
  virtual string OutputTensor(std::ostringstream& decl, const CallNode* call, int32_t zero_point,
                              double scale, double fix_scale);
  void malloc_buf(string out, int out_size) {}
  virtual void CSINNInit(const CallNode* call);
  virtual void CSINNDeinit(const CallNode* call);
  virtual void DisoOpU8(const CallNode* call, string op_name);
  virtual void FlattenU8(const CallNode* call);
  virtual void SqueezeU8(const CallNode* call);
  virtual void ReshapeU8(const CallNode* call);
  virtual void StridedSliceU8(const CallNode* call);

  void params_common_setup(std::ostringstream& decl, string op_name, string params_name) {
    std::ostringstream t0;
    t0 << params_name << ".layout = CSINN_NCHW";
    PushDeclLine(t0);
    t0 << params_name << ".api = CSINN_ANOLE";
    PushDeclLine(t0);
    setup_callback(decl, op_name, params_name);
  }

  void nbg_output_tensor(string tensor_name, std::vector<int> shape, int32_t zero_point,
                         double scale) {
    std::ostringstream t0;
    t0 << "struct csi_tensor *" << tensor_name << " = csi_alloc_tensor(sess);\n";
    for (uint32_t i = 0; i < shape.size(); i++) {
      t0 << "  " << tensor_name << "->dim[" << to_string(i) << "] = " << to_string(shape[i])
         << ";\n";
    }
    t0 << "  " << tensor_name << "->dim_count = " << to_string(shape.size()) << ";\n";
    t0 << "  " << tensor_name << "->zero_point = " << to_string(zero_point) << ";\n";
    t0 << "  " << tensor_name << "->scale = " << to_string(scale) << ";\n";

    nbg_buf_decl_.push_back(t0.str());
  }

 private:
  std::vector<string> ovx_body_;
  std::vector<string> nbg_input_tensor_name_;
  std::vector<string> nbg_func_;
  std::vector<string> nbg_output_tensor_name_;
  std::vector<string> nbg_buf_decl_;
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_CODEGEN_H_
