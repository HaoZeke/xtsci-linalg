#pragma once
// MIT License
// Copyright 2023--present Rohit Goswami <HaoZeke>
#include <algorithm>
#include <limits>
#include <vector>

#include "xtensor/xexpression.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xnoalias.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xtensor-blas/xlinalg.hpp"
#include "xtsci/linalg/precond/identity.hpp"

namespace xts {
namespace linalg {
namespace iterative {

template <typename ScalarType> struct IterativeMethodState {
  size_t iteration;
  ScalarType error;
  xt::xarray<ScalarType> search_direction;
  xt::xarray<ScalarType> candidate_solution;

  IterativeMethodState()
      : iteration(0), error(std::numeric_limits<ScalarType>::max()) {}

  IterativeMethodState(size_t it, ScalarType err,
                       const xt::xarray<ScalarType> &dir,
                       const xt::xarray<ScalarType> &sol)
      : iteration(it), error(err), search_direction(dir),
        candidate_solution(sol) {}
};

template <typename ScalarType> struct IterativeMethodHistory {
  std::vector<IterativeMethodState<ScalarType>> states;

  void record(const IterativeMethodState<ScalarType> &state) {
    states.push_back(state);
  }
};

template <typename ScalarType> struct IterativeResult {
  xt::xarray<ScalarType> solution;
  size_t iterations;
  ScalarType final_error;
  IterativeMethodHistory<ScalarType> history;
};

} // namespace iterative
} // namespace linalg
} // namespace xts
