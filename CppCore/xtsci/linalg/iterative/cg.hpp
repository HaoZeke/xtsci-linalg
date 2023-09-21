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
  std::function<ScalarType(const xt::xarray<ScalarType> &)> norm_fn;

  // Default constructor
  ConjugateGradientParams()
      : ConjugateGradientParams(1000, static_cast<ScalarType>(1e-6)) {}

  // Constructor for max_iter and tol
  ConjugateGradientParams(size_t max_iter_val, ScalarType tol_val)
      : ConjugateGradientParams(
            max_iter_val, tol_val, [](const xt::xarray<ScalarType> &vec) {
              return xt::sum(vec * vec)(); // Defaults to (L2 norm)**2
            }) {}

  // Fully specified constructor
  ConjugateGradientParams(
      size_t max_iter_val, ScalarType tol_val,
      std::function<ScalarType(const xt::xarray<ScalarType> &)> norm_fn_val)
      : max_iter(max_iter_val), tol(tol_val), norm_fn(norm_fn_val) {}
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
  auto norm_fn = params.norm_fn;

  auto rhsNorm2 = norm_fn(rhs);
  if (rhsNorm2 == 0) {
    x.fill(0);
    iters = 0;
    tol = 0;
    return ConjugateGradientResult<typename E3::value_type>{x, iters, tol};
  }

  auto threshold =
      std::max(tol * tol * rhsNorm2, std::numeric_limits<RealScalar>::min());
  auto residualNorm2 = norm_fn(residual);

  if (residualNorm2 < threshold) {
    iters = 0;
    tol = std::sqrt(residualNorm2 / rhsNorm2);
    return ConjugateGradientResult<typename E3::value_type>{x, iters, tol};
  }

  auto searchDirection = xt::eval(precond.solve(residual));
  auto dotProductNew = xt::linalg::dot(residual, searchDirection);

  while (iters < params.max_iter) {
    auto tmp_expr = xt::linalg::dot(mat, searchDirection);

    auto alpha = dotProductNew / xt::linalg::dot(searchDirection, tmp_expr);

    xt::noalias(x) += alpha * searchDirection;
    xt::noalias(residual) -= alpha * tmp_expr;

    residualNorm2 = norm_fn(residual);
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
