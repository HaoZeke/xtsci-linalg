#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

#include "xtensor/xexpression.hpp"
#include "xtensor/xview.hpp"

namespace xts {
namespace linalg {
namespace precond {

struct IdentityPreconditioner {
  template <typename E> auto solve(const xt::xexpression<E> &x_expr) const {
    const auto &x = x_expr.derived_cast();
    return x; // Identity preconditioning for simplicity
  }
};

} // namespace precond
} // namespace linalg
} // namespace xts
