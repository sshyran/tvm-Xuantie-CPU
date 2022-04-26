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
 * \file src/relay/backend/contrib/csinn/ref.cc
 * \brief Implementation of CSINN codegen APIs.
 */

#include "ref.h"

using namespace tvm::relay::qnn;
namespace tvm {
namespace relay {
namespace contrib {
using namespace backend;

void CodegenRef::CreateTensorSessData() {
  auto iter = tensor_data.begin();
  for (uint i = 0; i < tensor_data.size(); i++) {
    std::ostringstream t0;
    string data = iter->second;
    if (data == "alloc") {
      Output* out = GetOutput(iter->first);
      data = "alloc_" + to_string(alloc_idx_);
      auto out_shape = out->shape;
      // if output is a single number, out_shape.size() here is zero
      if (out_shape.size() == 0) {
        out_shape.push_back(1);
      }
      CreateMallocBuf(data, out_shape, out->dtype);
      alloc_idx_++;
    } else if (data == "hybrid_alloc") {
      iter++;
      continue;
    }
    t0 << iter->first << "->data = " << data << ";";
    ext_func_body.push_back(t0.str());
    iter++;
  }
  tensor_data.clear();
}

void CodegenRef::CreateHybridTensorSessData(std::vector<int> shape, string dtype) {
  std::ostringstream t0;
  for (auto item : tensor_data) {
    t0.str("");
    string data = item.second;
    if (item.second == "hybrid_alloc") {
      data = data + "_" + to_string(alloc_idx_);
      if (shape.size() == 0) {
        shape.push_back(1);
      }

      // get tensor bytes.
      int out_size = 1;
      for (size_t i = 0; i < shape.size(); ++i) {
        out_size *= shape[i];
      }
      if (dtype == "int32_t" || dtype == "float") {
        out_size *= 4;
      } else if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16_t") {
        out_size *= 2;
      }

      std::ostringstream t1;
      t1 << dtype << " *" << data << " = (" << dtype << " *)csi_mem_alloc(" << out_size << ");";
      ext_func_body.push_back(t1.str());

      alloc_idx_++;
    }
    t0 << item.first << "->data = " << data << ";";
    ext_func_body.push_back(t0.str());
  }
  tensor_data.clear();
}

void CodegenRef::malloc_buf(string out, int out_size) {
  std::ostringstream t0;
  t0 << cfg->dtype_input << " *" << out << " = (" << cfg->dtype_input << " *)csi_mem_alloc("
     << out_size << ");";
  ext_func_body.push_back(t0.str());
}

void CodegenRef::CreateMallocBuf(string name, std::vector<int> shape, string dtype) {
  int out_size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    out_size *= shape[i];
  }
  if (dtype == "int32_t" || dtype == "float") {
    out_size *= 4;
  } else if (dtype == "float16" || dtype == "bfloat16" || dtype == "int16_t") {
    out_size *= 2;
  }
  malloc_buf(name, out_size);
}

void CodegenRef::GenerateBackendCFunc(const string& func_name, const Array<Var>& args,
                                      const Output& out) {
  PrintNewLine(code_stream_);
  std::ostringstream t0;
  string in_dtype = cfg->dtype_input;
  string weight_dtype = cfg->dtype_weight;
  PrintNewLine(code_stream_);
  t0 << "int " << func_name << "_runtime_wrapper_(";
  t0 << "int64_t* arg_value, ";
  t0 << "int64_t* arg_type, ";
  t0 << "int64_t* arg_size, ";
  t0 << "int64_t* ret_vale, int64_t* ret_type_code" << args.size() << ") {";
  PrintOneLine(code_stream_, t0);

  EnterScope();
  PrintOneLine(code_stream_, "char** inputs = (char**)arg_value[0];");
  PrintOneLine(code_stream_, "char** outputs = (char**)arg_value[1];");
  PrintOneLine(code_stream_, "char *params_base = (char *)arg_value[2];");

  string out_dtype = GetCSINNDtype(weight_dtype);

  for (uint i = 0; i < args.size(); i++) {
    const auto& dtype_str = GetDtypeString(args[i]);
    std::string new_name = replace(args[i]->name_hint());
    auto iter = io_nodes.find(new_name);
    if (iter == io_nodes.end()) {
      CHECK(0);
    }
    QuantParams q_params = iter->second;
    auto ishape = GetShape(args[i]->checked_type());
    int size = 1;
    if (weight_dtype == "float16" || weight_dtype == "bfloat16" || weight_dtype == "int16_t") {
      size = size * 2;
    }
    for (uint j = 0; j < ishape.size(); j++) {
      size = size * ishape[j];
    }
    if (dtype_str == "float") {
      t0 << weight_dtype << "* __" << new_name << " = (" << weight_dtype << "*)malloc(" << size
         << ");";
      PrintOneLine(code_stream_, t0);
      string in_name = "(" + dtype_str + "*)inputs[" + to_string(i) + "]";

      string in_tensor = "input_" + to_string(i);
      t0 << "struct csi_tensor *" << in_tensor << " = csi_alloc_tensor(NULL);";
      PrintOneLine(code_stream_, t0);
      t0 << in_tensor << "->data = " << in_name << ";";
      PrintOneLine(code_stream_, t0);
      t0 << in_tensor << "->dim_count = " << ishape.size() << ";";
      PrintOneLine(code_stream_, t0);
      for (uint j = 0; j < ishape.size(); j++) {
        t0 << in_tensor << "->dim[" << j << "] = " << ishape[j] << ";";
        PrintOneLine(code_stream_, t0);
      }

      string qin_tensor = "qinput_" + to_string(i);
      PrintOneLine(code_stream_, t0);
      t0 << "struct csi_tensor *" << qin_tensor << " = csi_alloc_tensor(NULL);";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_tensor_copy(" << qin_tensor << ", " << in_tensor << ");";
      PrintOneLine(code_stream_, t0);
      t0 << qin_tensor << "->data = __" << new_name << ";";
      PrintOneLine(code_stream_, t0);

      t0 << qin_tensor << "->qinfo = (struct csi_quant_info *)(params_base + " << q_params.offset
         << ");";
      PrintOneLine(code_stream_, t0);
      t0 << qin_tensor << "->quant_channel =  " << q_params.q_size << ";";
      PrintOneLine(code_stream_, t0);
      t0 << qin_tensor << "->dtype = " << out_dtype << ";";
      PrintOneLine(code_stream_, t0);
      t0 << "csi_ref_nn_init(" << in_tensor << ", " << qin_tensor << ");";
      PrintNewLine(t0);
      PrintOneLine(code_stream_, t0);
    } else if (dtype_str == "uint8_t" || dtype_str == "int8_t") {
      t0 << weight_dtype << "* __" << new_name
         << " = (" + weight_dtype + "*)inputs[" + to_string(i) + "];";
      PrintOneLine(code_stream_, t0);
    }
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    // if (weight_dtype == "float16" || weight_dtype == "bfloat16") {
    //   size = size * 2;
    // }
    uint out_dim_count = output_list_[i].shape.size();
    int size = output_list_[i].size;
    if (output_list_[i].dtype == "float" || output_list_[i].dtype == "int32_t") {
      size *= 4;
    } else if (output_list_[i].dtype == "float16" || output_list_[i].dtype == "bfloat16" ||
               output_list_[i].dtype == "int16_t") {
      size *= 2;
    }
    if (output_list_[i].call != NULL) {
      auto iter = io_nodes.find(output_list_[i].name);
      if (iter == io_nodes.end()) {
        CHECK(0);
      }
      QuantParams q_params = iter->second;

      t0 << output_list_[i].dtype << "* out_q_" << i << " = (" << output_list_[i].dtype
         << " *)malloc(" << size << ");";
      PrintNewLine(t0);
      PrintOneLine(code_stream_, t0);
      auto out_dtype = DType2String(GetType(output_list_[i].call->checked_type()));
      string out_tensor = "output_" + to_string(i);
      t0 << "struct csi_tensor *" << out_tensor << " = csi_alloc_tensor(NULL);";
      PrintOneLine(code_stream_, t0);
      t0 << out_tensor << "->data = "
         << "outputs[" << i << "];";
      PrintOneLine(code_stream_, t0);
      t0 << out_tensor << "->dtype = " << GetCSINNDtype(out_dtype) << ";";
      PrintOneLine(code_stream_, t0);
      t0 << out_tensor << "->layout = " << GetCSINNActLayout(q_params.shape) << ";";
      PrintNewLine(t0);
      PrintOneLine(code_stream_, t0);

      string qout_tensor = "qoutput_" + to_string(i);
      t0 << "struct csi_tensor *" << qout_tensor << " = csi_alloc_tensor(NULL);";
      PrintOneLine(code_stream_, t0);
      t0 << qout_tensor << "->data = out_q_" << i << ";";
      PrintOneLine(code_stream_, t0);
      uint dim_count = out_dim_count == 0 ? 1 : out_dim_count;
      t0 << qout_tensor << "->dim_count = " << dim_count << ";";
      PrintOneLine(code_stream_, t0);
      if (out_dim_count == 0) {
        t0 << qout_tensor << "->dim[" << 0 << "] = 1;";
        PrintOneLine(code_stream_, t0);
      }
      for (uint j = 0; j < out_dim_count; j++) {
        t0 << qout_tensor << "->dim[" << j << "] = " << output_list_[i].shape[j] << ";";
        PrintOneLine(code_stream_, t0);
      }
      t0 << qout_tensor << "->qinfo = (struct csi_quant_info *)(params_base + " << q_params.offset
         << ");";
      PrintOneLine(code_stream_, t0);
      t0 << qout_tensor << "->quant_channel = " << q_params.q_size << ";";
      PrintOneLine(code_stream_, t0);
      t0 << qout_tensor << "->dtype = " << GetCSINNDtype(output_list_[i].dtype) << ";";
      PrintOneLine(code_stream_, t0);
      t0 << qout_tensor << "->layout = " << GetCSINNActLayout(q_params.shape) << ";";
      PrintOneLine(code_stream_, t0);
    } else {
      t0 << in_dtype << " *out_" << i << " = (" << in_dtype << " *)outputs[" << i << "];";
      PrintOneLine(code_stream_, t0);
    }

    PrintNewLine(t0);
    PrintOneLine(code_stream_, t0);
  }

  t0 << func_name << "_(";
  for (const auto& arg : args) {
    std::string new_name = replace(arg->name_hint());
    t0 << "__" << new_name << ", ";
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    if (output_list_[i].call != NULL) {
      t0 << "out_q_" << i << ", ";
    } else {
      t0 << "out_" << i << ", ";
    }
  }
  t0 << "params_base);\n";

  for (uint i = 0; i < output_list_.size(); i++) {
    auto out_node = output_list_[i].call;
    if (out_node != NULL) {
      t0 << "  csi_tensor_data_convert("
         << "output_" << i << ", "
         << "qoutput_" << i << ");\n";
      t0 << "  csi_mem_free(out_q_" << i << ");\n";
    }
  }

  PrintOneLine(code_stream_, t0);
  PrintOneLine(code_stream_, "return 0;");
  ExitScope();
  PrintOneLine(code_stream_, "}");
}

/*!
 * \brief A common interface that is used by various external runtime to
 * generate the wrapper to invoke external kernels.
 *
 * \param ext_func_id The unique id of an external function. It will be used
 * during runtime to pick the correct external function.
 * \param args The arguments used by the external function.
 * \param buf_decl The declaration of temporary buffers that used to store the
 * intermeidate of each external kernel.
 * \param body The statements of the external function.
 * \param out The name and id pairs for output.
 *
 * \return The emitted code string.
 */
string CodegenRef::JitImpl(const string& ext_func_id, const Array<Var>& args,
                           const std::vector<string>& buf_decl, const std::vector<string>& body,
                           const std::vector<Output>& out) {
  string in_dtype = cfg->dtype_weight;
  string hybrid_in_dtype = hybrid_cfg->dtype_weight;
  string base_dtype = GetCSINNDtype(in_dtype);

  // Create headers
  code_stream_ << "#include <csi_ref.h>\n\n";
  if (in_dtype == "float16" || hybrid_in_dtype == "float16") {
    code_stream_ << "#define float16 int16_t\n\n";
  } else if (in_dtype == "bfloat16" || hybrid_in_dtype == "bfloat16") {
    code_stream_ << "#define bfloat16 int16_t\n\n";
  } else if (in_dtype == "int4_t" || hybrid_in_dtype == "int4_t") {
    code_stream_ << "#define int4_t int8_t\n\n";
  }
  std::ostringstream t0;
  t0 << "void *" << ext_func_id << "_(";

  CHECK_EQ(out.size(), 1U) << "Internal error: only single output is support.";

  for (const auto& arg : args) {
    std::string new_name = replace(arg->name_hint());
    t0 << in_dtype << "* __" << new_name << ", ";
  }

  for (uint i = 0; i < output_list_.size(); i++) {
    t0 << output_list_[i].dtype << "* out_" << i << ", ";
  }

  t0 << "char *params_base) {";
  PrintOneLine(code_stream_, t0);
  EnterScope();

  PrintOneLine(code_stream_, "struct csi_session *sess = csi_alloc_session();");
  std::ostringstream sess_dtype;
  sess_dtype << "sess->base_dtype = " << base_dtype << ";";
  PrintOneLine(code_stream_, sess_dtype);
  PrintOneLine(code_stream_, "sess->base_layout = CSINN_LAYOUT_" + layout_ + ";");
  PrintOneLine(code_stream_, "sess->base_run_mode = CSINN_RM_LAYER;");
  PrintOneLine(code_stream_, "sess->base_api = " + target_name_ + ";");
  if (debug_level_ == "INFO") {
    PrintOneLine(code_stream_, "csi_debug_set_level(CSI_DEBUG_LEVEL_INFO);");
  }

  // Function body
  PrintNewLine(code_stream_);
  for (auto decl : buf_decl) {
    PrintOneLine(code_stream_, decl);
  }

  PrintNewLine(code_stream_);
  for (auto stmt : body) {
    PrintOneLine(code_stream_, stmt);
  }

  // free hybrid buffer
  for (auto item : hybrid_buffer_name_) {
    PrintOneLine(code_stream_, "csi_mem_free(" + item + "->data);");
    PrintOneLine(code_stream_, "csi_mem_free(" + item + ");");
  }

  PrintNewLine(code_stream_);

  for (uint i = 0; i < output_list_.size(); i++) {
    int out_size = output_list_[i].size;
    if (output_list_[i].dtype == "int32_t" || output_list_[i].dtype == "float") {
      out_size *= 4;
    } else if (output_list_[i].dtype == "float16" || output_list_[i].dtype == "bfloat16" ||
               output_list_[i].dtype == "int16_t") {
      out_size *= 2;
    }
    t0 << "memcpy("
       << "out_" << i << ", " << output_list_[i].name << "->data, " << out_size << ");";

    PrintOneLine(code_stream_, t0);
    if (!output_list_[i].is_const) {
      t0 << "csi_mem_free(" << output_list_[i].name << "->data);";
      PrintOneLine(code_stream_, t0);
    }
  }

  // Free buffers
  ExitScope();
  PrintOneLine(code_stream_, "}");

  this->GenerateBackendCFunc(ext_func_id, args, out[0]);

  DumpConstant();

  return code_stream_.str();
}

string CodegenRef::JIT(const std::vector<Output>& out) {
  return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body, out);
}

string CodegenRef::JIT(void) { return JIT(out_); }

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
