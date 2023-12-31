// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "xtensor-io/xnpz.hpp"
#include "xtensor/xarray.hpp"

#include "include/xtensor_fmt.hpp"
#include "xtsci/linalg/iterative/cg.hpp"
#include "xtsci/linalg/precond/identity.hpp"
#include "xtsci/linalg/precond/jacobi.hpp"

#include <catch2/catch_all.hpp>

// To get an ill-conditioned matrix
xt::xarray<double> create_hilbert_matrix(std::size_t n) {
  xt::xarray<double> H = xt::empty<double>({n, n});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      H(i, j) = 1.0 / (i + j + 1);
    }
  }
  return H;
}

TEST_CASE("Conjugate Gradient Tests", "[conjugate_gradient]") {
  xt::xarray<double> A = {{4, 1}, {1, 3}};
  xt::xarray<double> b = {1, 2};
  xt::xarray<double> x0 = {2, 1}; // Initial guess
  xts::linalg::precond::IdentityPreconditioner precond;
  const xt::xarray<double> default_expected_solution{
      0.09090909090909093936971885341336019,
      0.63636363636363624252112458634655923};

  SECTION("Basic convergence test") {
    size_t maxIters = 1000;
    double tol = 1e-5;
    auto res = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol}, precond);

    REQUIRE(xt::allclose(res.solution, default_expected_solution, 1e-15));
    REQUIRE(res.iterations <
            100); // Should converge in fewer than 1000 iterations
  }

  SECTION("Zero rhs test") {
    xt::xarray<double> zero_b = {0, 0};
    size_t maxIters = 1000;
    double tol = 1e-5;
    auto res = xts::linalg::iterative::conjugate_gradient(
        A, zero_b, x0, {maxIters, tol}, precond);

    REQUIRE(xt::all(xt::equal(res.solution, 0.0)));
    REQUIRE(res.iterations == 0);
  }

  SECTION("Custom Norm Function Test - L1 norm") {
    size_t maxIters = 1000;
    double tol = 1e-5;
    auto l1_norm = [](const xt::xarray<double> &vec) {
      return xt::sum(xt::abs(vec))();
    };
    auto res = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol, false, l1_norm}, precond);
    REQUIRE(xt::allclose(res.solution, default_expected_solution, 1e-15));
  }

  SECTION("Custom Norm Function Test - xt::linalg::norm") {
    size_t maxIters = 1000;
    double tol = 1e-5;
    auto l2_norm = [](const xt::xarray<double> &vec) {
      return xt::linalg::norm(vec, 2);
    };
    auto res = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol, false, l2_norm}, precond);
    REQUIRE(xt::allclose(res.solution, default_expected_solution, 1e-15));
  }

  SECTION("High Tolerance Test") {
    size_t maxIters = 1000;
    double tol = 1.0; // An exaggeratedly high tolerance.
    auto res = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol, false}, precond);

    // Solver should stop early and the solution might be far from accurate.
    REQUIRE(res.iterations < maxIters);
    REQUIRE(res.final_error <= tol);
  }

  SECTION("Low Tolerance Test with Hilbert matrix") {
    size_t n = 10;
    xt::xarray<double> Hilbert = create_hilbert_matrix(n);
    xt::xarray<double> b = xt::ones<double>({n});
    xt::xarray<double> x0 = xt::zeros<double>({n}); // Initial guess

    size_t maxIters = 500;
    double tol = 0.0; // Overly stringent

    auto res = xts::linalg::iterative::conjugate_gradient(
        Hilbert, b, x0, {maxIters, tol, false}, precond);

    // Doesn't converge exactly before maxIters due to the stringent
    // tolerance:
    REQUIRE(res.iterations == maxIters);
    // Ensure the final error is above the stringent tolerance (this is a safety
    // check)
    REQUIRE(res.final_error > tol);
  }

  SECTION("Non-Convergence Test") {
    // Create an ill-conditioned matrix, for example:
    xt::xarray<double> ill_conditioned_A = {{1e-50, 0}, {0, 1}};
    xt::xarray<double> ill_conditioned_b = {1, 20};
    xt::xarray<double> x0_ill = {1, 1}; // Initial guess

    size_t maxIters = 2;
    double tol = 1e-5;
    auto res = xts::linalg::iterative::conjugate_gradient(
        ill_conditioned_A, ill_conditioned_b, x0_ill, {maxIters, tol, false},
        precond);

    // It should stop at max iterations due to non-convergence.
    REQUIRE(res.iterations == maxIters);
  }

  SECTION("Convergence with the Jacobi preconditioner") {
    size_t maxIters = 1000;
    double tol = 1e-5;
    xts::linalg::precond::JacobiPreconditioner jacobi_precond(A);
    auto res = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol, false}, jacobi_precond);
    REQUIRE(xt::allclose(res.solution, default_expected_solution, 1e-15));
    REQUIRE(res.iterations < 50); // Should converge in fewer than 1000
                                  // iterations with the Jacobi preconditioner
  }

  SECTION("Preconditioner Comparison") {
    xt::xarray<double> A = {{4, 1}, {1, 3}};
    xt::xarray<double> b = {1, 2};
    xt::xarray<double> x0 = {2, 1}; // Initial guess

    xts::linalg::precond::IdentityPreconditioner identity_precond;
    xts::linalg::precond::JacobiPreconditioner jacobi_precond(A);

    size_t maxIters = 1000;
    double tol = 1e-5;

    // Solving using Identity Preconditioner
    auto res_identity = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol, false}, identity_precond);

    // Solving using Jacobi Preconditioner
    auto res_jacobi = xts::linalg::iterative::conjugate_gradient(
        A, b, x0, {maxIters, tol, false}, jacobi_precond);

    // Check solutions
    REQUIRE(
        xt::allclose(res_identity.solution, default_expected_solution, 1e-15));
    REQUIRE(
        xt::allclose(res_jacobi.solution, default_expected_solution, 1e-15));

    // Verify that the Jacobi preconditioner converged faster than the Identity
    // preconditioner
    REQUIRE(res_jacobi.iterations < res_identity.iterations);
  }
}
