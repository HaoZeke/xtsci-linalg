#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"
namespace xts {
namespace linalg {
namespace iterative {

template <typename ScalarType> struct ConjugateGradientResult {
  xt::xarray<ScalarType> solution;
  int iterations;
  ScalarType final_error;
};

template <typename E1, typename E2, typename E3, typename Preconditioner>
// Straight port of the Eigen implementation
ConjugateGradientResult<typename E3::value_type>
conjugate_gradient(const xt::xexpression<E1> &mat_expr,
                   const xt::xexpression<E2> &rhs_expr,
                   xt::xexpression<E3> &x_expr, const Preconditioner &precond,
                   int &iterations, typename E3::value_type &final_error) {
  const auto &mat = mat_expr.derived_cast();
  const auto &rhs = rhs_expr.derived_cast();
  auto &x = x_expr.derived_cast();

  using RealScalar = typename E3::value_type;

  RealScalar tol = final_error;
  int maxIters = iterations;

  xt::xarray<RealScalar> residual = rhs - xt::linalg::dot(mat, x);

  RealScalar rhsNorm2 = xt::linalg::norm(rhs, 2);
  rhsNorm2 *= rhsNorm2;

  if (rhsNorm2 == 0) {
    x.fill(0);
    iterations = 0;
    final_error = 0;
    return ConjugateGradientResult<typename E3::value_type>{x, iterations, final_error};
  }

  RealScalar threshold =
      std::max(tol * tol * rhsNorm2, std::numeric_limits<RealScalar>::min());

  RealScalar residualNorm2 = xt::linalg::norm(residual, 2);
  residualNorm2 *= residualNorm2;

  if (residualNorm2 < threshold) {
    iterations = 0;
    final_error = std::sqrt(residualNorm2 / rhsNorm2);
    return ConjugateGradientResult<typename E3::value_type>{x, iterations, final_error};
  }

  auto searchDirection = precond.solve(residual);

  RealScalar dotProductNew = xt::linalg::dot(residual, searchDirection)();
  int i = 0;

  while (i < maxIters) {
    auto tmp = xt::linalg::dot(mat, searchDirection);
    RealScalar alpha = dotProductNew / xt::linalg::dot(searchDirection, tmp)();

    x += alpha * searchDirection;
    residual -= alpha * tmp;

    residualNorm2 = xt::linalg::norm(residual, 2);
    residualNorm2 *= residualNorm2;

    if (residualNorm2 < threshold)
      break;

    auto z = precond.solve(residual);
    RealScalar dotProductOld = dotProductNew;
    dotProductNew = xt::linalg::dot(residual, z)();
    RealScalar beta = dotProductNew / dotProductOld;
    searchDirection = z + beta * searchDirection;

    i++;
  }

  final_error = std::sqrt(residualNorm2 / rhsNorm2);
  iterations = i;
  return ConjugateGradientResult<typename E3::value_type>{x, i, final_error};
}
} // namespace iterative
} // namespace linalg
} // namespace xts
