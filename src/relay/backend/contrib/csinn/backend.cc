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
 * \file src/relay/backend/contrib/csinn/backend.cc
 * \brief Implementation of CSINN backend codegen APIs.
 */

#include "backend.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace csinn {

Call::Call(RelayExpr op, Array<Expr> args, std::vector<int> shape, Attrs attrs,
           Array<RelayType> type_args, Span span) {
  ObjectPtr<CallNode> n = make_object<CallNode>();
  n->op = std::move(op);
  n->args = std::move(args);
  n->attrs = std::move(attrs);
  n->type_args = std::move(type_args);
  n->span = std::move(span);
  n->shape = shape;
  data_ = std::move(n);
}

Var::Var(Id vid, std::vector<int> shape, Type type_annotation, Span span) {
  ObjectPtr<VarNode> n = make_object<VarNode>();
  n->vid = std::move(vid);
  n->type_annotation = std::move(type_annotation);
  n->span = std::move(span);
  n->shape = shape;
  data_ = std::move(n);
}

Constant::Constant(runtime::NDArray data, std::vector<int> shape, Span span) {
  ObjectPtr<ConstantNode> n = make_object<ConstantNode>();
  n->data = std::move(data);
  n->span = std::move(span);
  n->shape = shape;
  data_ = std::move(n);
}

Tuple::Tuple(tvm::Array<Expr> fields, Span span) {
  ObjectPtr<TupleNode> n = make_object<TupleNode>();
  n->fields = std::move(fields);
  n->span = std::move(span);
  data_ = std::move(n);
}

TupleGetItem::TupleGetItem(Expr tuple, int index, Span span) {
  ObjectPtr<TupleGetItemNode> n = make_object<TupleGetItemNode>();
  n->tuple = std::move(tuple);
  n->index = index;
  n->span = std::move(span);
  data_ = std::move(n);
}

Expr FromRelay::VisitExpr_(const RelayCallNode* op) {
  tvm::Array<Expr> call_args;
  for (auto arg : op->args) {
    auto new_arg = VisitRelayExpr(arg);
    call_args.push_back(new_arg);
  }
  auto type = op->checked_type();
  std::vector<int> shape;
  if (type.as<TensorTypeNode>()) {
    shape = tvm::relay::backend::GetShape(type);
  }
  auto ret = Call(op->op, call_args, shape, op->attrs, op->type_args, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayVarNode* op) {
  auto shape = tvm::relay::backend::GetShape(op->checked_type());
  auto ret = Var(op->vid, shape, op->type_annotation, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayConstantNode* op) {
  auto shape = tvm::relay::backend::GetShape(op->checked_type());
  auto ret = Constant(op->data, shape, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayTupleNode* op) {
  tvm::Array<Expr> fields;
  for (auto field : op->fields) {
    auto new_field = VisitRelayExpr(field);
    fields.push_back(new_field);
  }
  auto ret = Tuple(fields, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

Expr FromRelay::VisitExpr_(const RelayTupleGetItemNode* get_item) {
  auto new_tuple = VisitRelayExpr(get_item->tuple);
  auto ret = TupleGetItem(new_tuple, get_item->index, get_item->span);
  ret->checked_type_ = get_item->checked_type_;
  return ret;
}

Expr FromRelay::VisitRelayExpr(const RelayExpr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprFunctor::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

Expr FromRelay::expr(const RelayExpr& expr) {
  Expr ret = ExprFunctor::VisitExpr(expr);
  return ret;
}

RelayExpr ToRelay::visit_expr(const CallNode* op) {
  tvm::Array<RelayExpr> call_args;
  for (auto arg : op->args) {
    auto new_arg = visit(arg);
    call_args.push_back(new_arg);
  }
  auto ret = RelayCall(op->op, call_args, op->attrs, op->type_args, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const VarNode* op) {
  auto ret = RelayVar(op->vid, op->type_annotation, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const ConstantNode* op) {
  auto ret = RelayConstant(op->data, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const TupleNode* op) {
  tvm::Array<RelayExpr> fields;
  for (auto field : op->fields) {
    auto new_field = visit(field);
    fields.push_back(new_field);
  }
  auto ret = RelayTuple(fields, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit_expr(const TupleGetItemNode* op) {
  auto new_tuple = visit(op->tuple);
  auto ret = RelayTupleGetItem(new_tuple, op->index, op->span);
  ret->checked_type_ = op->checked_type_;
  return ret;
}

RelayExpr ToRelay::visit(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    RelayExpr new_expr = CSINNExprFunctor::visit(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

RelayExpr ToRelay::relay(const Expr& expr) {
  RelayExpr ret = visit(expr);
  return ret;
}

Expr Function::visit_expr(const CallNode* op) {
  for (auto arg : op->args) {
    visit(arg);
  }
  return GetRef<Expr>(op);
}

Expr Function::visit_expr(const VarNode* op) { return GetRef<Expr>(op); }

Expr Function::visit_expr(const ConstantNode* op) { return GetRef<Expr>(op); }

Expr Function::visit_expr(const TupleNode* op) {
  for (auto field : op->fields) {
    visit(field);
  }
  return GetRef<Expr>(op);
}

Expr Function::visit_expr(const TupleGetItemNode* op) {
  visit(op->tuple);
  return GetRef<Expr>(op);
}

Expr Function::visit(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = CSINNExprFunctor::visit(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}

void Function::import_realy_expr(const RelayExpr& func) { expr = get_expr.expr(func); }

RelayExpr Function::export_realy_expr() {
  RelayExpr ret = export_relay.relay(expr);
  return ret;
}

void Function::phase0() { /* TODO */
}
void Function::phase1() { /* TODO */
}
void Function::phase2() { /* TODO */
}
void Function::phase3() { /* TODO */
}

void Function::optimization() {
  // PHASE 0
  phase0();
  // user-defined phase-0
  target_define_phase0();
  // PHASE 1
  phase1();
  // user-defined phase-1
  target_define_phase1();
  // PHASE 2
  phase2();
  // user-defined phase-2
  target_define_phase2();
  // PHASE 3
  phase3();
  // user-defined phase-3
  target_define_phase3();
}
}  // namespace csinn
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
