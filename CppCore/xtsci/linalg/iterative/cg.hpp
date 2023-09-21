#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <limits>

#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"
namespace xts {
namespace linalg {
namespace iterative {

template <typename ScalarType> struct ConjugateGradientResult {
  xt::xarray<ScalarType> solution;
  size_t iterations;
  ScalarType final_error;
};

template <typename ScalarType> struct ConjugateGradientParams {
  size_t max_iter;
  ScalarType tol;
};

template <typename E1, typename E2, typename E3, typename Preconditioner>
// Straight port of the Eigen implementation
ConjugateGradientResult<typename E3::value_type> conjugate_gradient(
    const xt::xexpression<E1> &mat_expr, const xt::xexpression<E2> &rhs_expr,
    xt::xexpression<E3> &x_expr, const Preconditioner &precond,
    const ConjugateGradientParams<typename E3::value_type> &params) {

  const auto &mat = mat_expr.derived_cast();
  const auto &rhs = rhs_expr.derived_cast();
  auto &x = x_expr.derived_cast();

  using RealScalar = typename E3::value_type;

  RealScalar tol = params.tol;
  size_t iters{0};

  auto residual_expr = rhs - xt::linalg::dot(mat, x);
  xt::xarray<RealScalar> residual = residual_expr; // Forced evaluation here
  auto norm_sq = [](const auto &vec) { return xt::sum(vec * vec)(); };

  auto rhsNorm2 = norm_sq(rhs);
  if (rhsNorm2 == 0) {
    x.fill(0);
    iters = 0;
    tol = 0;
    return ConjugateGradientResult<typename E3::value_type>{x, iters, tol};
  }

  auto threshold =
      std::max(tol * tol * rhsNorm2, std::numeric_limits<RealScalar>::min());
  auto residualNorm2 = norm_sq(residual);

  if (residualNorm2 < threshold) {
    iters = 0;
    tol = std::sqrt(residualNorm2 / rhsNorm2);
    return ConjugateGradientResult<typename E3::value_type>{x, iters, tol};
  }

  auto searchDirection = precond.solve(residual);
  auto dotProductNew = xt::linalg::dot(residual, searchDirection)();

  while (iters < params.max_iter) {
    auto tmp_expr = xt::linalg::dot(mat, searchDirection);

    auto alpha = dotProductNew / xt::linalg::dot(searchDirection, tmp_expr);

    xt::noalias(x) += alpha * searchDirection;
    xt::noalias(residual) -= alpha * tmp_expr;

    residualNorm2 = norm_sq(residual);
    if (residualNorm2 < threshold)
      break;

    auto z = precond.solve(residual);
    auto dotProductOld = dotProductNew;
    dotProductNew = xt::linalg::dot(residual, z)();
    auto beta = dotProductNew / dotProductOld;

    xt::noalias(searchDirection) = z + beta * searchDirection;
    iters++;
  }

  tol = std::sqrt(residualNorm2 / rhsNorm2);
  return ConjugateGradientResult<typename E3::value_type>{x, iters, tol};
}
} // namespace iterative
} // namespace linalg
} // namespace xts
