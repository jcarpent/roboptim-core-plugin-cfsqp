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

#include <roboptim/core/twice-derivable-function.hh>

#include <roboptim/core/solver-factory.hh>

#include "common.hh"

using namespace roboptim;

typedef boost::mpl::vector<LinearFunction, DerivableFunction> clist_t;
typedef Solver<DerivableFunction, clist_t> solver_t;

struct F : public TwiceDerivableFunction
{
public:
  F ()
    : TwiceDerivableFunction (2, 1)
  {}
  ~F () throw () {}

  void
  impl_compute (result_t& res, const argument_t& x) const throw ()
  {
    res.clear ();
    res (0) = x[0] * x[0] + x[1] * x[1];
  }

  void
  impl_gradient (gradient_t& grad, const argument_t& x, size_type) const throw ()
  {
    grad.clear ();
    grad (0) = 2 * x[0];
    grad (1) = 2 * x[1];
  }

  void
  impl_hessian (hessian_t& hessian, const argument_t& res, size_type) const throw ()
  {
    hessian.clear ();
    hessian (0, 0) = hessian (1, 1) = 2;
    hessian (0, 1) = hessian (1, 0) = 0;
  }
};

struct G : public DerivableFunction
{
public:
  G ()
    : DerivableFunction (2, 1)
  {}
  ~G ()  throw () {}

  void
  impl_compute (result_t& res, const argument_t& x) const throw ()
  {
    res.clear ();
    res (0) = x[0] + x[1] - 1;
  }

  void
  impl_gradient (gradient_t& grad, const argument_t& x, size_type) const throw ()
  {
    grad.clear ();
    grad (0) = 1.;
    grad (1) = 1.;
  }
};

int run_test ()
{
  F f;

  solver_t::problem_t pb (f);

  boost::shared_ptr<G> g (new G ());
  pb.addConstraint (boost::static_pointer_cast<DerivableFunction> (g),
		    Function::makeInterval (0., 0.));

  // Initialize solver
  SolverFactory<solver_t> factory ("cfsqp", pb);
  solver_t& solver = factory ();

  // Compute the minimum and retrieve the result.
  solver_t::result_t res = solver.minimum ();

  // Display solver information.
  std::cout << solver << std::endl;

  // Check if the minimization has succeed.
  switch (res.which ())
    {
    case GenericSolver::SOLVER_VALUE:
      {
	Result& result = boost::get<Result> (res);
	std::cout << result << std::endl;
	break;
      }

    case GenericSolver::SOLVER_NO_SOLUTION:
      {
	std::cerr << "No solution" << std::endl;
	return 1;
      }
    case GenericSolver::SOLVER_VALUE_WARNINGS:
      {
	ResultWithWarnings& result = boost::get<ResultWithWarnings> (res);
	std::cout << result.x << std::endl;
	std::cerr << result << std::endl;
	break;
      }

    case GenericSolver::SOLVER_ERROR:
      {
	SolverError& result = boost::get<SolverError> (res);
	std::cout << result << std::endl;
      return 1;
      }
    }

  return 0;
}


GENERATE_TEST ()
