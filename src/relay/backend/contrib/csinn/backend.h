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
 * \file src/relay/backend/contrib/csinn/backend.h
 * \brief The base class for backend.
 */
#ifndef TVM_RELAY_BACKEND_CONTRIB_CSINN_BACKEND_H_
#define TVM_RELAY_BACKEND_CONTRIB_CSINN_BACKEND_H_

#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/function.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../utils.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

using RelayFunction = tvm::relay::Function;
using RelayExpr = tvm::relay::Expr;
using RelayExprNode = tvm::relay::ExprNode;
using RelayExprVisitor = tvm::relay::ExprVisitor;
using RelayType = tvm::relay::Type;
using RelayCall = tvm::relay::Call;
using RelayCallNode = tvm::relay::CallNode;
using RelayVar = tvm::relay::Var;
using RelayVarNode = tvm::relay::VarNode;
using RelayConstant = tvm::relay::Constant;
using RelayConstantNode = tvm::relay::ConstantNode;
using RelayTuple = tvm::relay::Tuple;
using RelayTupleNode = tvm::relay::TupleNode;
using RelayTupleGetItem = tvm::relay::TupleGetItem;
using RelayTupleGetItemNode = tvm::relay::TupleGetItemNode;

class ExprNode : public RelayExprNode {
 public:
  std::vector<int> get_shape() const { return shape; }
  std::vector<int> shape;
  void* target_data;
  std::string name;
  std::vector<std::string> output_names;
  static constexpr const char* _type_key = "csinn.expr";
  static constexpr const uint32_t _type_child_slots = 0;
  TVM_DECLARE_BASE_OBJECT_INFO(ExprNode, RelayExprNode);
};

class Expr : public RelayExpr {
 public:
  void set_target_data(void* tdata) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->target_data = tdata;
  }
  void set_expr_name(std::string str) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->name = str;
  }
  std::string get_expr_name() {
    auto data = static_cast<ExprNode*>(data_.get());
    return data->name;
  }
  void set_output_names(std::vector<std::string> name) {
    auto data = static_cast<ExprNode*>(data_.get());
    data->output_names = name;
  }
  std::vector<std::string> get_output_names() {
    auto data = static_cast<ExprNode*>(data_.get());
    return data->output_names;
  }
  ExprNode* operator->() { return static_cast<ExprNode*>(data_.get()); }
  ExprNode* get() { return operator->(); }
  TVM_DEFINE_OBJECT_REF_METHODS(Expr, RelayExpr, ExprNode);
};

class CallNode : public ExprNode {
 public:
  RelayExpr op;
  tvm::Array<Expr> args;
  Attrs attrs;
  tvm::Array<Type> type_args;

  static constexpr const char* _type_key = "csinn.call";
  TVM_DECLARE_FINAL_OBJECT_INFO(CallNode, ExprNode);
};

class Call : public Expr {
 public:
  Call(RelayExpr op, Array<Expr> args, std::vector<int> shape, Attrs attrs = Attrs(),
       Array<RelayType> type_args = Array<RelayType>(), Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Call, Expr, CallNode);
};

class VarNode : public ExprNode {
 public:
  Id vid;
  Type type_annotation;
  const String& name_hint() const { return vid->name_hint; }
  static constexpr const char* _type_key = "csinn.var";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarNode, ExprNode);
};

class Var : public Expr {
 public:
  Var(Id vid, std::vector<int> shape, Type type_annotation, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Var, Expr, VarNode);
};

class ConstantNode : public ExprNode {
 public:
  runtime::NDArray data;
  static constexpr const char* _type_key = "csinn.constant";
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantNode, ExprNode);
};

class Constant : public Expr {
 public:
  Constant(runtime::NDArray data, std::vector<int> shape, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Constant, Expr, ConstantNode);
};

class Tuple : public Expr {
 public:
  explicit Tuple(tvm::Array<Expr> fields, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(Tuple, Expr, TupleNode);
};

class TupleNode : public ExprNode {
 public:
  tvm::Array<Expr> fields;

  static constexpr const char* _type_key = "csinn.tuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleNode, ExprNode);
};

class TupleGetItem : public Expr {
 public:
  TupleGetItem(Expr tuple, int index, Span span = Span());
  TVM_DEFINE_OBJECT_REF_METHODS(TupleGetItem, Expr, TupleGetItemNode);
};

class TupleGetItemNode : public ExprNode {
 public:
  Expr tuple;
  int index;

  static constexpr const char* _type_key = "csinn.tuple_get_item";
  TVM_DECLARE_FINAL_OBJECT_INFO(TupleGetItemNode, ExprNode);
};

template <typename FType>
class CSINNExprFunctor;

template <typename R>
class CSINNExprFunctor<R(const Expr& n)> {
 private:
  using TSelf = CSINNExprFunctor<R(const Expr& n)>;
  using FType = tvm::NodeFunctor<R(const ObjectRef& n, TSelf* self)>;

 public:
  /*! \brief the result type of this functor */
  using result_type = R;
  /*! \brief virtual destructor */
  virtual ~CSINNExprFunctor() {}

  virtual R visit(const Expr& n) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this);
  }
  // Functions that can be overriden by subclass
  virtual R visit_expr(const ConstantNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const TupleNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const VarNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const GlobalVarNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const FunctionNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const CallNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const LetNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const IfNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const OpNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const TupleGetItemNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const RefCreateNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const RefReadNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const RefWriteNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const ConstructorNode* op) { return visit_expr_default(op); }
  virtual R visit_expr(const MatchNode* op) { return visit_expr_default(op); }
  virtual R visit_expr_default(const Object* op) {
    LOG(FATAL) << "Do not have a default for " << op->GetTypeKey();
    throw;
  }

 private:
  // initialize the vtable.
  static FType InitVTable() {
    FType vtable;

    vtable.template set_dispatch<ConstantNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const ConstantNode*>(n.get()));
    });
    vtable.template set_dispatch<TupleNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const TupleNode*>(n.get()));
    });
    vtable.template set_dispatch<VarNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const VarNode*>(n.get()));
    });
    vtable.template set_dispatch<GlobalVarNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const GlobalVarNode*>(n.get()));
    });
    vtable.template set_dispatch<FunctionNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const FunctionNode*>(n.get()));
    });
    vtable.template set_dispatch<CallNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const CallNode*>(n.get()));
    });
    vtable.template set_dispatch<LetNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const LetNode*>(n.get()));
    });
    vtable.template set_dispatch<IfNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const IfNode*>(n.get()));
    });
    vtable.template set_dispatch<OpNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const OpNode*>(n.get()));
    });
    vtable.template set_dispatch<TupleGetItemNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const TupleGetItemNode*>(n.get()));
    });
    vtable.template set_dispatch<RefCreateNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const RefCreateNode*>(n.get()));
    });
    vtable.template set_dispatch<RefReadNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const RefReadNode*>(n.get()));
    });
    vtable.template set_dispatch<RefWriteNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const RefWriteNode*>(n.get()));
    });
    vtable.template set_dispatch<ConstructorNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const ConstructorNode*>(n.get()));
    });
    vtable.template set_dispatch<MatchNode>([](const ObjectRef& n, TSelf* self) {
      return self->visit_expr(static_cast<const MatchNode*>(n.get()));
    });
    return vtable;
  }
};

class FromRelay : public ExprFunctor<Expr(const RelayExpr&)> {
 public:
  Expr expr(const RelayExpr& expr);

  virtual Expr VisitRelayExpr(const RelayExpr& expr);
  virtual Expr VisitExpr_(const RelayCallNode* call);
  virtual Expr VisitExpr_(const RelayVarNode* op);
  virtual Expr VisitExpr_(const RelayConstantNode* op);
  // virtual Expr VisitExpr_(const RelayGlobalVarNode* op);
  // virtual Expr VisitExpr_(const RelayOpNode* op);
  virtual Expr VisitExpr_(const RelayTupleNode* op);
  // virtual Expr VisitExpr_(const RelayFunctionNode* op);
  // virtual Expr VisitExpr_(const RelayLetNode* op);
  // virtual Expr VisitExpr_(const RelayIfNode* op);
  virtual Expr VisitExpr_(const RelayTupleGetItemNode* op);
  // virtual Expr VisitExpr_(const RelayRefCreateNode* op);
  // virtual Expr VisitExpr_(const RelayRefReadNode* op);
  // virtual Expr VisitExpr_(const RelayRefWriteNode* op);
  // virtual Expr VisitExpr_(const RelayConstructorNode* op);
  // virtual Expr VisitExpr_(const RelayMatchNode* op);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<RelayExpr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
};

class ToRelay : public CSINNExprFunctor<RelayExpr(const Expr&)> {
 public:
  RelayExpr relay(const Expr& expr);

  virtual RelayExpr visit(const Expr& expr);
  virtual RelayExpr visit_expr(const ConstantNode* call);
  virtual RelayExpr visit_expr(const CallNode* call);
  virtual RelayExpr visit_expr(const VarNode* call);
  virtual RelayExpr visit_expr(const TupleNode* call);
  virtual RelayExpr visit_expr(const TupleGetItemNode* call);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, RelayExpr, ObjectPtrHash, ObjectPtrEqual> memo_;
};

class Function : public RelayFunction, public CSINNExprFunctor<Expr(const Expr&)> {
 public:
  // Function(const RelayFunction& func) { init_from_realy(func); }
  virtual void to_target_graph() {}
  virtual void from_target_graph() {}
  virtual Expr visit_expr(const ConstantNode* call);
  virtual Expr visit_expr(const CallNode* call);
  virtual Expr visit_expr(const VarNode* call);
  virtual Expr visit_expr(const TupleNode* call);
  virtual Expr visit_expr(const TupleGetItemNode* call);
  virtual Expr visit(const Expr& expr);
  virtual void import_realy_expr(const RelayExpr& func);
  virtual RelayExpr export_realy_expr();

  virtual void optimization();
  virtual void phase0();
  virtual void phase1();
  virtual void phase2();
  virtual void phase3();
  virtual void target_define_phase0() {}
  virtual void target_define_phase1() {}
  virtual void target_define_phase2() {}
  virtual void target_define_phase3() {}

 protected:
  Expr expr;
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;

 private:
  class FromRelay get_expr;
  class ToRelay export_relay;
};

}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_BACKEND_CONTRIB_CSINN_BACKEND_H_
