#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>

#include "xtensor/xexpression.hpp"
#include "xtensor/xview.hpp"

namespace xts {
namespace linalg {
namespace precond {

template <class T> class JacobiPreconditioner {
  xt::xarray<T> M_inv;

public:
  explicit JacobiPreconditioner(const xt::xarray<T> &A) {
    M_inv = 1.0 / xt::diagonal(A);
  }

  template <typename E> auto solve(const xt::xexpression<E> &x_expr) const {
    const auto &x = x_expr.derived_cast();
    return M_inv * x;
  }
};

// Deduction guide via CTAD
template <typename T>
JacobiPreconditioner(const xt::xarray<T> &) -> JacobiPreconditioner<T>;

} // namespace precond
} // namespace linalg
} // namespace xts
