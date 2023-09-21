// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/linalg/iterative/cg.hpp"

#include <catch2/catch_all.hpp>

// A mock preconditioner for testing
struct MockPreconditioner {
  template <typename E> auto solve(const xt::xexpression<E> &x_expr) const {
    const auto &x = x_expr.derived_cast();
    return x; // Identity preconditioning for simplicity
  }
};

TEST_CASE("Conjugate Gradient Tests", "[conjugate_gradient]") {
  xt::xarray<double> A = {{4, 1}, {1, 3}};
  xt::xarray<double> b = {1, 2};
  xt::xarray<double> x0 = {2, 1}; // Initial guess
  MockPreconditioner precond;

  SECTION("Basic convergence test") {
    int maxIters = 1000;
    double tol = 1e-5;
    auto res = xts::linalg::iterative::conjugate_gradient(A, b, x0, precond, maxIters,
                                               tol);

    xt::xarray<double> expected_solution = {0.090909090909091,
                                            0.636363636363636};
    REQUIRE(xt::allclose(res.solution, expected_solution, 1e-15));
    REQUIRE(res.iterations < 100); // Should converge in fewer than 1000 iterations
  }

  SECTION("Zero rhs test") {
    xt::xarray<double> zero_b = {0, 0};
    int maxIters = 1000;
    double tol = 1e-5;
    auto res = xts::linalg::iterative::conjugate_gradient(A, zero_b, x0, precond, maxIters,
                                               tol);

    REQUIRE(xt::all(xt::equal(res.solution, 0.0)));
    REQUIRE(res.iterations == 0);
  }
}
