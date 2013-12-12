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

#ifndef ROBOPTIM_CORE_CFSQP_HH
# define ROBOPTIM_CORE_CFSQP_HH
# include <roboptim/core/sys.hh>
# include <roboptim/core/portability.hh>
# include <iostream>
# include <string>
# include <utility>
# include <vector>

# include <boost/mpl/vector.hpp>

# include <boost/variant.hpp>
# include <roboptim/core/derivable-function.hh>
# include <roboptim/core/linear-function.hh>
# include <roboptim/core/solver.hh>
# include <roboptim/core/util.hh>

namespace roboptim
{
  /// \addtogroup roboptim_solver
  /// @{

  /// \brief CFSQP based solver.
  ///
  /// Instantiate this class to solve an optimization
  /// problem using CFSQP.
  ///
  /// This solver works with DerivableFunction for the cost function;
  /// constraints can be linear or derivable functions.
  class ROBOPTIM_DLLEXPORT CFSQPSolver
    : public Solver<DerivableFunction,
		    boost::mpl::vector<LinearFunction, DerivableFunction> >
  {
    /// \brief CLIST parameter passed to parent.
    typedef boost::mpl::vector<LinearFunction, DerivableFunction> clist_t;

  public:
    /// \brief Categorize constraints.
    ///
    /// Used with the which method of the Boost.Variant, it
    /// allows to check for a constraint's real type.
    ///
    /// \warning Make sure to keep enum values in the
    /// same order than the MPL vector used to specify CLIST.
    enum ConstraintType
      {
	/// \brief Constraint is a linear function.
	LINEAR = 0,
	/// \brief Constraint is a derivable function.
	NONLINEAR = 1
      };

    /// \brief Parent type.
    typedef Solver<DerivableFunction, clist_t> parent_t;

    /// \brief Instantiate the solver from a problem.
    ///
    /// \param problem problem that will be solved
    /// \param print verbosity level
    explicit CFSQPSolver (const problem_t& problem, int print = 0) throw ();

    explicit CFSQPSolver (const CFSQPSolver&) throw ();

    virtual ~CFSQPSolver () throw ();

    /// \brief Solve the problem.
    virtual void solve () throw ();

    /// \brief Retrieve interval constraint representation.
    ///
    /// \note This method provides is  an advance feature, most users
    /// can safely ignore it.
    ///
    /// This additional vector is built to transform the provided
    /// constraint vector into a vector that CFSQP can accept:
    /// - only \f$g \leq X\f$ constraints
    /// - the order has to be:
    ///  - non-linear inequalities,
    ///  - linear inequalities,
    ///  - non-linear equalities,
    ///  - linear equalities
    /// .
    ///
    /// This vector is filled by the constructor. It duplicates
    /// the constraints that have both a lower and an upper bound
    /// into two constraints that CFSQP can handle.
    /// It also detects the constraints where the lower
    /// and upper bounds are similar and handle them correctly
    /// by considering them as an equality. Linear and non-linear
    /// constraints are also separated to preserve the order CFSQP
    /// expects.
    ///
    /// The representation used in the vector is:
    /// - pair of integers: constraint index in the original vector
    ///                     and function id in the case the constraint
    ///                     is non-scalar (outputSize > 1)
    /// - bool: is it a lower constraint?
    ///  If the constraint is an equality, the boolean has no meaning.
    const std::vector<std::pair<std::pair<int, int>, bool> >& cfsqpConstraints ()
      const throw ();

    /// \brief Number of linear inequalities constraints.
    const int& nineq () const throw ();
    /// \brief Number of non-linear inequalities constraints.
    const int& nineqn () const throw ();
    /// \brief Number of linear equalities constraints.
    const int& neq () const throw ();
    /// \brief Number of non-linear equalities constraints.
    const int& neqn () const throw ();

    /// \brief Tag gradient computation as invalid.
    ///
    /// If gradients are tagged as invalid, the #solve () method
    /// will fail (i.e. result will be a SolverError object).
    void invalidateGradient () throw ();

    /// \brief Display the solver on the specified output stream.
    ///
    /// \param o output stream used for display
    /// \return output stream
    virtual std::ostream& print (std::ostream& o) const throw ();

    void setIterationCallback (callback_t callback) throw (std::runtime_error);
  private:
    /// \brief Initialize parameters.
    ///
    /// Add solver parameters. Called during construction.
    void initializeParameters () throw ();

    /// \brief Initialize bounds.
    ///
    /// Fill the two bounds array as required by CFSQP.
    /// \param bl lower bounds array
    /// \param bu upper bounds array
    void initializeBounds (double* bl, double* bu) const throw ();

    /// \brief Copy CFSQP final constraints values into a vector.
    ///
    /// Copy constraints final values from CFSQP internal representation
    /// to the generic representation.
    ///
    /// Used to fill the result that will be returned to the user.
    ///
    /// \param constraints generic representation
    /// \param g CFSQP representation
    void fillConstraints (vector_t& constraints, double* g) const throw ();

    void perIterationCallback (const double* x);

    /// \brief Number of non linear inegality constraints (including linear's).
    int nineq_;
    /// \brief Number of linear inegality constraints.
    int nineqn_;
    /// \brief Number of non linear equality constraints (including linear ones).
    int neq_;
    /// \brief Number of linear equality constraints.
    int neqn_;

    /// \brief Internal representation of constraints.
    std::vector<std::pair<std::pair<int, int>, bool> > cfsqpConstraints_;

    /// \brief Set to true if a gradient check has failed.
    /// Always false if gradient checks are not enabled.
    bool invalidGradient_;

    /// \brief Callback object, should be called at each iteration.
    callback_t callback_;

    /// \brief Current state of the solver (used by the callback function).
    solverState_t solverState_;
  };

  /// @}

} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_CFSQP_HH

//  LocalWords:  CFSQP
