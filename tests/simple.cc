// Copyright (C) 2009 by Thomas Moulard, AIST, CNRS, INRIA.
//
// This file is part of the roboptim.
//
// roboptim is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// roboptim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with roboptim.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <boost/numeric/ublas/io.hpp>
#include <boost/variant/detail/variant_io.hpp>

#include <roboptim/core/fwd.hh>
#include <roboptim/core/solver-error.hh>
#include <roboptim/core/result.hh>
#include <roboptim/core/result-with-warnings.hh>

#include <roboptim/core/plugin/cfsqp.hh>

#include "common.hh"
#include "shared-tests/hs071.hh"

int run_test ()
{
  F f;

  CFSQPSolver::problem_t pb (f);
  initialize_problem<CFSQPSolver::problem_t,
    roboptim::DerivableFunction> (pb);

  // Initialize solver
  CFSQPSolver solver (pb);

  // Display solver information.
  std::cout << solver << std::endl;

  // Compute the minimum and retrieve the result.
  CFSQPSolver::result_t res = solver.minimum ();

  // Check if the minimization has succeed.
  std::cout << res << std::endl;
  system("PAUSE");
  return 0;
}


GENERATE_TEST ()
