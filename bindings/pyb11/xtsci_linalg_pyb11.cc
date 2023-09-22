// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pyvectorize.hpp"

#include "xtsci/linalg/iterative/base.hpp"
#include "xtsci/linalg/iterative/cg.hpp"
#include "xtsci/linalg/precond/jacobi.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_xts_linalg, m) {
  xt::import_numpy();
  py::module_ iterative = m.def_submodule("iterative");
  py::module_ linalg = m.def_submodule("linalg");
  linalg.attr("iterative") = iterative;

  using RealScalar = double;
  using CGParams = xts::linalg::iterative::ConjugateGradientParams<RealScalar>;
  using IterativeRes = xts::linalg::iterative::IterativeResult<RealScalar>;

  py::class_<xts::linalg::iterative::IterativeMethodState<RealScalar>>
      iterative_method_state_RealScalar(iterative,
                                        "IterativeMethodStateRealScalar");
  iterative_method_state_RealScalar.def(py::init<>())
      .def(py::init<size_t, RealScalar, const xt::xarray<RealScalar> &,
                    const xt::xarray<RealScalar> &>())
      .def_readwrite(
          "iteration",
          &xts::linalg::iterative::IterativeMethodState<RealScalar>::iteration)
      .def_readwrite(
          "error",
          &xts::linalg::iterative::IterativeMethodState<RealScalar>::error)
      .def_readwrite("search_direction",
                     &xts::linalg::iterative::IterativeMethodState<
                         RealScalar>::search_direction)
      .def_readwrite("candidate_solution",
                     &xts::linalg::iterative::IterativeMethodState<
                         RealScalar>::candidate_solution);

  py::class_<xts::linalg::iterative::IterativeMethodHistory<RealScalar>>
      iterative_method_history_RealScalar(iterative,
                                          "IterativeMethodHistoryRealScalar");
  iterative_method_history_RealScalar.def(py::init<>())
      .def("record",
           &xts::linalg::iterative::IterativeMethodHistory<RealScalar>::record)
      .def_readwrite(
          "states",
          &xts::linalg::iterative::IterativeMethodHistory<RealScalar>::states);

  py::class_<xts::linalg::iterative::IterativeResult<RealScalar>>
      iterative_result_RealScalar(iterative, "IterativeResultRealScalar");
  iterative_result_RealScalar.def(py::init<>())
      .def_readwrite(
          "solution",
          &xts::linalg::iterative::IterativeResult<RealScalar>::solution)
      .def_readwrite(
          "iterations",
          &xts::linalg::iterative::IterativeResult<RealScalar>::iterations)
      .def_readwrite(
          "final_error",
          &xts::linalg::iterative::IterativeResult<RealScalar>::final_error)
      .def_readwrite(
          "history",
          &xts::linalg::iterative::IterativeResult<RealScalar>::history);

  py::class_<CGParams> cg_params(iterative, "ConjugateGradientParams");
  cg_params.def(py::init<>())
      .def(py::init<size_t, RealScalar>())
      .def(py::init<size_t, RealScalar, bool>())
      .def(
          py::init<size_t, RealScalar, bool,
                   std::function<RealScalar(const xt::xarray<RealScalar> &)>>())
      .def_readwrite("max_iter", &CGParams::max_iter)
      .def_readwrite("tol", &CGParams::tol)
      .def_readwrite("keep_history", &CGParams::keep_history)
      .def_readwrite("norm_fn", &CGParams::norm_fn);

  // py::class_<IterativeRes>(iterative, "IterativeResult")
  //     .def_readonly("solution", &IterativeRes::solution)
  //     .def_readonly("iterations", &IterativeRes::iterations)
  //     .def_readonly("final_error", &IterativeRes::final_error);
  iterative.def(
      "conjugate_gradient",
      [](const xt::pyarray<RealScalar> &mat, const xt::pyarray<RealScalar> &rhs,
         xt::xarray<RealScalar> &x, const CGParams &params) {
        xts::linalg::precond::IdentityPreconditioner precond;
        IterativeRes result = xts::linalg::iterative::conjugate_gradient(
            mat, rhs, x, params, precond);
        return result;
      },
      py::arg("mat"), py::arg("rhs"), py::arg("x"), py::arg("params"));
};
