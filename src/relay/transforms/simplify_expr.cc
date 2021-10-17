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
 * \file src/relay/transforms/simplify_expr.cc
 * \brief A pass for simplifying the Relay expression.
 */

#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/support/logging.h>

#include "../op/tensor/transform.h"

namespace tvm {
namespace relay {

static Op reshape_op = Op::Get("reshape");
static Op reverse_reshape_op = Op::Get("contrib_reverse_reshape");

/*!
 * \brief SimplifyReshape matches the pattern of consecutive reshape or reverse_reshape ops,
 *   and merges into one reshape op. Meanwhile, remove extra reshape op while the shape of output
 *   in preceding layer equals to the that of the current reshape op.
 */
class SimplifyReshape {
 public:
  SimplifyReshape() {
    x_ = WildcardPattern(make_object<WildcardPatternNode>());
    auto reshape1 = AltPattern(ExprPattern(reshape_op), ExprPattern(reverse_reshape_op));
    auto reshape2 = AltPattern(ExprPattern(reshape_op), ExprPattern(reverse_reshape_op));
    pattern_ = CallPattern(reshape1, {CallPattern(reshape2, {x_}, Attrs{}, {})}, Attrs{}, {});
    pattern_rm_ = CallPattern(reshape1, {x_}, Attrs{}, {});
  }

  Expr callback(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    auto x = node_map[x_][0];
    Array<Integer> newshape = get_out_shape(pre);
    if (newshape.size() != 0) {
      return MakeReshape(x, newshape);
    }
    return post;
  }

  Expr callback_rm(const Expr& pre, const Expr& post, const Map<DFPattern, Array<Expr>>& node_map) {
    auto x = node_map[x_][0];
    Array<Integer> pre_out_shape = get_out_shape(pre);
    Array<Integer> input_out_shape = get_out_shape(x);

    if (pre_out_shape.size() == 0 || input_out_shape.size() == 0) {
      return post;
    }

    if (is_equal(pre_out_shape, input_out_shape)) {
      return x;
    } else {
      return post;
    }
  }

  Array<Integer> get_out_shape(const Expr& expr) {
    Array<Integer> newshape;
    bool const_shape = true;
    // some exprs have no checked_type attribute.
    if (!expr->checked_type_.defined()) {
      return newshape;
    }
    for (auto dim : Downcast<TensorType>(expr->checked_type())->shape) {
      if (dim.as<IntImmNode>() == nullptr) {
        const_shape = false;
        break;
      }
      newshape.push_back(Downcast<Integer>(dim));
    }
    if (!const_shape) {
      newshape.clear();
    }
    return newshape;
  }

  bool is_equal(Array<Integer> left, Array<Integer> right) {
    bool res = true;
    if (left.size() == 0 || left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); i++) {
      if (left[i]->value != right[i]->value) {
        res = false;
        break;
      }
    }
    return res;
  }

  DFPattern pattern() const { return pattern_; }
  DFPattern pattern_rm() const { return pattern_rm_; }

 private:
  /*! \brief Pattern input */
  DFPattern x_;
  /*! \brief Pattern for consecutive reshape or reverse_reshape ops */
  DFPattern pattern_;
  /*! \brief Pattern for reshape or reverse_reshape op that will be removed */
  DFPattern pattern_rm_;
};

/*!
 * \brief ExprSimplifier simplifies the Relay expression.
 */
class ExprSimplifier {
 public:
  explicit ExprSimplifier(IRModule mod) : mod_(mod) {
    auto reshape_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = simplify_reshape_.callback(pre, post, node_map);
    };
    callbacks_.push_back(
        DFPatternCallback(simplify_reshape_.pattern(), PackedFunc(reshape_func), true));

    auto reshape_rm_func = [this](TVMArgs args, TVMRetValue* rv) {
      Expr pre = args[0];
      Expr post = args[1];
      Map<DFPattern, Array<Expr>> node_map = args[2];
      *rv = simplify_reshape_.callback_rm(pre, post, node_map);
    };
    callbacks_rm_.push_back(
        DFPatternCallback(simplify_reshape_.pattern_rm(), PackedFunc(reshape_rm_func), true));
  }

  Expr Simplify(const Expr& expr) { return RewritePatterns(callbacks_, expr, mod_); }
  Expr Simplify_rm(const Expr& expr) { return RewritePatterns(callbacks_rm_, expr, mod_); }

 private:
  IRModule mod_;
  /*! \brief Simplify reshape pattern */
  SimplifyReshape simplify_reshape_;
  /*! \brief Callbacks for expr simplification */
  Array<DFPatternCallback> callbacks_;
  /*! \brief Callbacks for expr remove */
  Array<DFPatternCallback> callbacks_rm_;
};

Expr SimplifyExpr(const Expr& expr, const IRModule& mod) {
  Expr expr_opt = ExprSimplifier(mod).Simplify(expr);
  return ExprSimplifier(mod).Simplify_rm(expr_opt);
}

namespace transform {

Pass SimplifyExpr() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(SimplifyExpr(f, m));
      };
  return CreateFunctionPass(pass_func, 0, "SimplifyExpr", {"InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.SimplifyExpr").set_body_typed(SimplifyExpr);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
