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

#include <roboptim/core/numeric-linear-function.hh>
#include <roboptim/core/solver-factory.hh>

#include "common.hh"

using namespace roboptim;

typedef boost::mpl::vector<LinearFunction, DerivableFunction> clist_t;
typedef Solver<DerivableFunction, clist_t> solver_t;

int run_test ()
{
  using namespace boost;

  Function::matrix_t a (1, 1);
  a.clear ();
  a (0, 0) = 1.;
  Function::vector_t b (1);
  b.clear ();
  NumericLinearFunction cost (a, b);


  solver_t::problem_t pb (cost);

  shared_ptr<NumericLinearFunction> constraint
    (new NumericLinearFunction (a, b));

  pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		    Function::makeLowerInterval (-42.));
  pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		    Function::makeUpperInterval (42.));
  pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		    Function::makeInterval (-42., 42.));

  // Initialize solver
  SolverFactory<solver_t> factory ("cfsqp", pb);
  solver_t& solver = factory ();

  // Display solver information.
  std::cout << solver << std::endl;

  // Compute the minimum and retrieve the result.
  solver_t::result_t res = solver.minimum ();

  // Check if the minimization has succeed.
  std::cout << res << std::endl;

  return 0;
}


GENERATE_TEST ()
