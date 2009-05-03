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


/**
 * \brief Implementation of the CFSQP module.
 */

#ifndef ROBOPTIM_CORE_CFSQP_HH
# define ROBOPTIM_CORE_CFSQP_HH
# include <iostream>
# include <string>
# include <utility>
# include <vector>

# include <boost/variant.hpp>
# include <roboptim/core/derivable-function.hh>
# include <roboptim/core/linear-function.hh>
# include <roboptim/core/solver.hh>
# include <roboptim/core/util.hh>

namespace roboptim
{
  /**
     \addtogroup roboptim_solver
     @{
  */

  /// CFSQP solver.
  class CFSQPSolver : public Solver<DerivableFunction,
                                    boost::variant<const DerivableFunction*,
                                                   const LinearFunction*> >
  {
  public:
    /// Variant of both Linear and NonLinear functions.
    typedef boost::variant<const DerivableFunction*,
                           const LinearFunction*> constraint_t;

    typedef Solver<DerivableFunction, constraint_t> parent_t;

    /// Constructor.
    explicit CFSQPSolver (const problem_t&, int = 0) throw ();
    /// Destructor.
    virtual ~CFSQPSolver () throw ();
    /// Solve the problem.
    virtual void solve () throw ();


    const std::vector<std::pair<int, bool> >& cfsqpConstraints ()
      const throw ();

    const int& nineq () const throw ();
    const int& nineqn () const throw ();
    const int& neq () const throw ();
    const int& neqn () const throw ();

    int& mode () throw ();
    const int& mode () const throw ();

    int& iprint () throw ();
    const int& iprint () const throw ();

    int& miter () throw ();
    const int& miter () const throw ();

    double& bigbnd () throw ();
    const double& bigbnd () const throw ();

    double& eps () throw ();
    const double& eps () const throw ();

    double& epseqn () throw ();
    const double& epseqn () const throw ();

    double& udelta () throw ();
    const double& udelta () const throw ();

    virtual std::ostream& print (std::ostream&) const throw ();
  private:
    /// Initialize bounds.
    void initialize_bounds (double* bl, double* bu) const throw ();

    /// Number of non linear inegality constraints (including linear ones).
    int nineq_;
    /// Number of linear inegality constraints.
    int nineqn_;
    /// Number of non linear equality constraints (including linear ones).
    int neq_;
    /// Number of linear equality constraints.
    int neqn_;
    /// CFSQP mode.
    int mode_;
    /// Logging level.
    int iprint_;
    /// Number of iterations.
    int miter_;
    /// Symbolizes infinity.
    double bigbnd_;
    /// Final norm requirement for the Newton direction.
    double eps_;
    /// Maximum violation of nonlinear equality constraint.
    double epseqn_;
    /// Perturbation size used in finite difference.
    double udelta_;

    std::vector<std::pair<int, bool> > cfsqpConstraints_;
  };

/**
   @}
*/

} // end of namespace roboptim

#endif //! ROBOPTIM_CORE_CFSQP_HH
