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
#include <boost/mpl/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/variant/get.hpp>

#include <roboptim/core/solver-factory.hh>

#include "common.hh"
#include "hs071.hh"

using namespace roboptim;
typedef boost::mpl::vector<LinearFunction, DerivableFunction> clist_t;
typedef Solver<DerivableFunction, clist_t> solver_t;

int run_test ()
{
  F f;

  solver_t::problem_t pb (f);
  initialize_problem<solver_t::problem_t,
    roboptim::DerivableFunction> (pb);

  // Initialize solver
  SolverFactory<solver_t> factory ("cfsqp", pb);
  solver_t& solver = factory ();

  // Compute the minimum and retrieve the result.
  solver_t::result_t res = solver.minimum ();

  // Display solver information.
  std::cout << solver << std::endl;

  // Display the result.
  std::cout << res << std::endl;
  return 0;
}


GENERATE_TEST ()
