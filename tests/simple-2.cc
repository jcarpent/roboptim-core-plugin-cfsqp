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

#include "shared-tests/common.hh"

#include <iostream>
#include <boost/format.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <roboptim/core/numeric-linear-function.hh>
#include <roboptim/core/solver-factory.hh>

using namespace roboptim;

typedef boost::mpl::vector<LinearFunction, DifferentiableFunction> clist_t;
typedef Solver<DifferentiableFunction, clist_t> solver_t;

struct SumArgs : public DifferentiableFunction
{
public:
  SumArgs ()
    : DifferentiableFunction (8, 1, "\\sum_i x_i")
  {}
  ~SumArgs () throw () {}

  void
  impl_compute (result_t& res, const argument_t& x) const throw ()
  {
    res.setZero ();

    for (unsigned i = 0; i < inputSize (); ++i)
      res (0) += x[i];
  }

  void
  impl_gradient (gradient_t& grad, const argument_t&, size_type) const throw ()
  {
    grad.setZero ();
    for (unsigned i = 0; i < inputSize (); ++i)
      grad (i) = 1.;
  }

  unsigned i_;
};

struct F : public DifferentiableFunction
{
public:
  F (unsigned i, bool opposite)
    : DifferentiableFunction (8, 1, (boost::format("%1%.5 * x_%2%^2")
				% (opposite ? "-" : "")
				% i).str ()),
      i_ (i),
      opposite_ (opposite)
  {}
  ~F () throw () {}

  void
  impl_compute (result_t& res, const argument_t& x) const throw ()
  {
    res.setZero ();
    res (0) = .5 * x[i_] * x[i_];
    if (opposite_)
      res (0) *= -1.;
  }

  void
  impl_gradient (gradient_t& grad, const argument_t& x, size_type) const throw ()
  {
    grad.setZero ();
    grad (i_) = x[i_];
    if (opposite_)
      grad (i_) *= -1.;
  }

  unsigned i_;
  bool opposite_;
};

void setAB (unsigned i, Function::matrix_t& a, Function::vector_t& b);

void setAB (unsigned i, Function::matrix_t& a, Function::vector_t& b)
{
  a.setZero ();
  a (0, i) = 3.;
  b.setZero ();
  b[0] = 2.;
}

int run_test ()
{
  using namespace boost;

  SumArgs cost;

  Function::matrix_t a (1, 8);
  Function::vector_t b (1);

  solver_t::problem_t pb (cost);

  {
    setAB (0, a, b);
    shared_ptr<NumericLinearFunction> constraint
      (new NumericLinearFunction (a, b));
    pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		      Function::makeLowerInterval (4.));
  }

  {
    setAB (1, a, b);
    a (0, 1) *= -1.;
    shared_ptr<NumericLinearFunction> constraint
      (new NumericLinearFunction (a, b));
    pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		      Function::makeUpperInterval (6.));
  }

  {
    setAB (2, a, b);
    shared_ptr<NumericLinearFunction> constraint
      (new NumericLinearFunction (a, b));
    pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		      Function::makeInterval (8., 8.));
  }

  {
    setAB (3, a, b);
    shared_ptr<NumericLinearFunction> constraint
      (new NumericLinearFunction (a, b));
    pb.addConstraint (static_pointer_cast<LinearFunction> (constraint),
		      Function::makeInterval (-10., 10.));
  }





  {
    shared_ptr<F> constraint (new F (4, true));
    pb.addConstraint (static_pointer_cast<DifferentiableFunction> (constraint),
		      Function::makeLowerInterval (-12.));
  }

  {
    shared_ptr<F> constraint (new F (5, false));
    pb.addConstraint (static_pointer_cast<DifferentiableFunction> (constraint),
		      Function::makeUpperInterval (14.));
  }

  {
    shared_ptr<F> constraint (new F (6, false));
    pb.addConstraint (static_pointer_cast<DifferentiableFunction> (constraint),
		      Function::makeInterval (16., 16.));
  }

  {
    shared_ptr<F> constraint (new F (7, false));
    pb.addConstraint (static_pointer_cast<DifferentiableFunction> (constraint),
		      Function::makeInterval (-18., 18.));
  }

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
