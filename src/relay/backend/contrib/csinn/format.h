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
 * \file src/relay/backend/contrib/csinn/format.h
 * \brief The class for c code emit.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_FORMAT_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_FORMAT_H_

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../codegen_c/codegen_c.h"
#include "csi_nn.h"
#include "shl_utils.h"

using std::string;

namespace tvm {
namespace relay {
namespace contrib {

struct CSIStructedSparsity {
  int8_t* index;
  std::vector<int32_t> shape;
  size_t size{0};
  enum csinn_mem_type_enum type;
};

struct CSIConstant {
  string name;
  string dtype;
  size_t size;
  void* data_buf;
  int32_t layout;

  struct CSIStructedSparsity sparse;
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
  int32_t value_type;
  string name;
  std::vector<int> shape;
  int32_t q_size;
  int32_t offset;
};

class CSINNTensor {
 public:
  CSINNTensor() { tensor = csinn_alloc_tensor(NULL); }

  void append_str(std::ostringstream& decl) {
    decl << ";";
    astr.push_back(decl.str());
    decl.str("");
  }

  string to_mtype(enum csinn_mem_type_enum mtype);
  string to_dtype(enum csinn_dtype_enum dtype);
  string to_layout(int32_t layout);
  string to_layout() { return to_layout(tensor->layout); }
  std::vector<string> serialize(size_t qoffset, size_t coffset);
  virtual void to_file(std::ofstream& file) = 0;
  size_t bm_size() {
    return binary_model_quant_size + binary_model_const_size + binary_model_sparse_index_size;
  }

  void set_const(const struct CSIConstant* cst) {
    const_data = cst->data_buf;
    binary_model_const_size = cst->size;
    if (cst->sparse.size) {
      tensor->mtype = cst->sparse.type;
      sparse_index_data = reinterpret_cast<char*>(cst->sparse.index);
      binary_model_sparse_index_size = cst->sparse.size;
    }
    if (cst->layout == CSINN_LAYOUT_O32HWI32 || cst->layout == CSINN_LAYOUT_O32I32) {
      tensor->layout = cst->layout;
    }
  }

  void set_quant(const struct QuantParams& quant) {
    quant_data = quant.qinfo;
    binary_model_quant_size = quant.q_size * sizeof(Qinfo);
  }

  struct csinn_tensor* tensor;
  size_t const_offset;
  size_t qinfo_offset;
  string name;
  void* const_data;
  char* sparse_index_data;
  struct Qinfo* quant_data;
  size_t binary_model_quant_size{0};
  size_t binary_model_const_size{0};
  size_t binary_model_sparse_index_size{0};

 private:
  void push_str(std::ostringstream& decl) {
    decl << ";";
    str.push_back(decl.str());
    decl.str("");
  }
  std::vector<string> str;
  std::vector<string> astr;
};

class CSINNConstantTensor : public CSINNTensor {
 public:
  void to_file(std::ofstream& file);
};

class CSINNVarTensor : public CSINNTensor {
 public:
  void to_file(std::ofstream& file);
};

class CSINNOP {
 public:
  CSINNTensor* get_tensor(string name);

  size_t size() { return op_size; }
  void set_bm_base(size_t base) {
    op_binary_model_base = base;
    const_offset = op_binary_model_base;
    qinfo_offset = op_binary_model_base;
  }
  void increase_qoffset();
  void increase_coffset();
  void push_input(CSINNVarTensor* in);
  void push_output(CSINNVarTensor* out);
  void push_constant(CSINNConstantTensor* cst);

  std::vector<string> serialize();
  void to_file(std::ofstream& file);

 private:
  size_t op_size{0};
  size_t const_offset{0};
  size_t qinfo_offset{0};
  size_t op_binary_model_base{0};
  std::vector<CSINNVarTensor*> inputs;
  std::vector<CSINNVarTensor*> outputs;
  std::vector<CSINNConstantTensor*> consts;
  std::vector<string> strs;
};

class CSINNBMGraph {
 public:
  CSINNBMGraph() {
    sess = static_cast<struct csinn_session*>(calloc(1, sizeof(struct csinn_session)));
  }
  size_t push_op(CSINNOP* op) {
    ops.push_back(op);
    graph_size += op->size();
    return graph_size;
  }
  void set_layer_align(size_t align) { layer_align = align; }
  void set_input(CSINNTensor* tensor) { return inputs.push_back(tensor); }
  std::vector<CSINNTensor*> get_inputs() { return inputs; }
  void set_output(CSINNTensor* tensor) { return outputs.push_back(tensor); }
  std::vector<CSINNTensor*> get_outputs() { return outputs; }
  size_t dump_params(std::string path);
  size_t size() { return graph_size; }
  size_t dump_graph_info(std::string path);

  CSINNTensor* get_tensor(string name) {
    for (auto op : ops) {
      auto out = op->get_tensor(name);
      if (out) {
        return out;
      }
    }
    return NULL;
  }

  struct csinn_session* sess;

 private:
  std::vector<CSINNTensor*> inputs;
  std::vector<CSINNTensor*> outputs;
  std::vector<CSINNOP*> ops;
  size_t graph_base_size{0};
  size_t graph_size{0};
  size_t layer_align{32};
};

class CSINNCodeFormat {
 public:
  void EnterScope() { indent_ += 2; }

  void ExitScope() {
    CHECK_GE(indent_, 2U) << "Wrong ident found.";
    indent_ -= 2;
  }

  void Indents() {
    for (int i = 0; i < indent_; i++) {
      code_stream_ << ' ';
    }
  }

  void OneLine(string str) {
    Indents();
    code_stream_ << str << "\n";
  }

  void OneLine(std::ostringstream& str) {
    OneLine(str.str());
    str.str("");
  }

  void NewLine() { code_stream_ << "\n"; }

  void PushDecl(const std::vector<string>& decls) {
    for (string decl : decls) {
      buf_decl_.push_back(decl);
    }
  }

  void PushDecl(std::ostringstream& decl) {
    decl << ";";
    buf_decl_.push_back(decl.str());
    decl.str("");
  }

  void PushCall(std::ostringstream& call) {
    call << ";";
    buf_call_.push_back(call.str());
    call.str("");
  }

  void BufToCode() {
    for (auto decl : buf_decl_) {
      OneLine(decl);
    }
    NewLine();
    for (auto stmt : buf_call_) {
      OneLine(stmt);
    }
  }

  string str() { return code_stream_.str(); }

 private:
  std::vector<string> buf_decl_;
  std::vector<string> buf_call_;
  std::ostringstream code_stream_;
  int indent_{0};
};

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_FORMAT_H_
