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

#include <cstring>
#include <limits>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <roboptim/core/function.hh>
#include <roboptim/core/indent.hh>
#include <roboptim/core/result.hh>
#include <roboptim/core/result-with-warnings.hh>
#include <roboptim/core/util.hh>

#include "roboptim/core/plugin/cfsqp.hh"
#include "ofsqp.hh"

#ifdef ROBOPTIM_CORE_CFSQP_PLUGIN_CHECK_GRADIENT
# include <boost/format.hpp>
# include <roboptim/core/finite-difference-gradient.hh>
#endif //!ROBOPTIM_CORE_CFSQP_PLUGIN_CHECK_GRADIENT


namespace roboptim
{
  typedef void (*fct_t) (int, int, double*, double*, void*);
  typedef void (*grad_t) (int, int, double*, double*, fct_t, void*);

  namespace detail
  {
    void obj (int, int, double*, double*, void*);
    double evaluate_inequality (double, bool, double, double);
    void constr (int, int, double*, double*, void*);
    void gradob (int, int, double*, double*, fct_t, void*);
    void gradcn (int, int, double*, double*, fct_t, void*);


    void CFSQPCheckGradient (const DerivableFunction& function,
			     unsigned functionId,
			     Function::vector_t& x,
			     int constraintId,
			     CFSQPSolver& solver) throw ();

    /// \internal
    void CFSQPCheckGradient (const DerivableFunction& function,
			     unsigned functionId,
			     Function::vector_t& x,
			     int constraintId,
			     CFSQPSolver& solver) throw ()
    {
#ifdef ROBOPTIM_CORE_CFSQP_PLUGIN_CHECK_GRADIENT
      using boost::format;
      try
	{
	  checkGradientAndThrow (function, functionId, x, 1.);
	}
      catch (BadGradient& bg)
	{
	  solver.invalidateGradient ();
	  std::cerr
	    << ((functionId < 0)
		? "Invalid cost function gradient:"
		: (format ("Invalid constraint function gradient (id = %1%):")
		   % constraintId).str ())
	    << std::endl
	    << function.getName ()
	    << std::endl
	    << bg
	    << std::endl;
	}
#endif //!ROBOPTIM_CORE_CFSQP_PLUGIN_CHECK_GRADIENT
    }

    /// \internal
    /// CFSQP objective function.
    void obj (int nparam, int j , double* x, double* fj, void* cd)
    {
      assert (cd);
      CFSQPSolver* solver = static_cast<CFSQPSolver*> (cd);

      Function::vector_t x_ (nparam);
      array_to_vector (x_, x);
      *fj = solver->problem ().function () (x_)[0];
    }


    /// \internal
    /// Evaluate an inequality defined from lower/upper bounds.
    double
    evaluate_inequality (double x, bool is_lower, double l, double u)
    {
      if (is_lower)
        // g(x) >= b, -g(x) + b <= 0
        return -x + l;
      else
        // g(x) <= b, g(x) - b <= 0
        return x - u;
    }

    /// \internal
    /// CFSQP constraints function.
    void constr (int nparam, int j,
                 double* x, double* gj, void* cd)
    {
      using namespace boost;

      assert (cd && !!gj && !!x && nparam >= 0 && j > 0);

      CFSQPSolver* solver = static_cast<CFSQPSolver*> (cd);
      assert (j > 0 && solver->cfsqpConstraints ().size () - j >= 0);

      Function::vector_t x_ (nparam);
      array_to_vector (x_, x);

      // Decrement j to have C style indexation (0...size - 1).
      j--;

      // Constraint index in the generic representation.
      int j_ = solver->cfsqpConstraints ()[j].first;

      assert (j >= 0 && solver->cfsqpConstraints ().size () - j > 0);
      assert (j_ >= 0 && solver->problem ().bounds ().size () - j_ > 0);
      assert (j_ >= 0 && solver->problem ().constraints ().size () - j_ > 0);

      if (0 <= j && j < solver->nineqn ())
        {
	  shared_ptr<DerivableFunction> f =
	    get<shared_ptr<DerivableFunction> >
            (solver->problem ().constraints ()[j_]);
	  assert (f);
          Function::vector_t res = (*f) (x_);
          *gj = evaluate_inequality
            (res (0),
             solver->cfsqpConstraints ()[j].second,
             solver->problem ().bounds ()[j_].first,
             solver->problem ().bounds ()[j_].second);
          return;
        }

      if (solver->nineqn () <= j && j < solver->nineq ())
        {
	  shared_ptr<LinearFunction> f =
            get<shared_ptr<LinearFunction> >
            (solver->problem ().constraints ()[j_]);
	  assert (f);
          Function::vector_t res = (*f) (x_);
          *gj = evaluate_inequality
            (res (0),
             solver->cfsqpConstraints ()[j].second,
             solver->problem ().bounds ()[j_].first,
             solver->problem ().bounds ()[j_].second);
          return;
        }

      j -= solver->nineq ();
      assert (j >= 0);
      if (0 <= j && j < solver->neqn ())
        {
          shared_ptr<DerivableFunction> f =
	    get<shared_ptr<DerivableFunction> >
            (solver->problem ().constraints ()[j_]);
	  assert (f);
          Function::vector_t res = (*f) (x_);
          *gj = res (0) - solver->problem ().bounds ()[j_].first;
          return;
        }

      if (solver->neqn () <= j && j < solver->neq ())
        {
          shared_ptr<LinearFunction> f =
            get<shared_ptr<LinearFunction> >
            (solver->problem ().constraints ()[j_]);
	  assert (f);
          Function::vector_t res = (*f) (x_);
          *gj = res (0) - solver->problem ().bounds ()[j_].first;
          return;
        }
      assert (0);
    }

    /// \internal
    /// CFSQP objective function gradient.
    void gradob (int nparam, int j,
                 double* x, double* gradf, fct_t dummy, void* cd)
    {
      assert (nparam >= 0 && j == 1 && !!x && !!gradf && !!cd);

      CFSQPSolver* solver = static_cast<CFSQPSolver*> (cd);

      Function::vector_t x_ (nparam);
      array_to_vector (x_, x);
      DerivableFunction::gradient_t grad =
        solver->problem ().function ().gradient (x_, 0);

      vector_to_array (gradf, grad);

      CFSQPCheckGradient (solver->problem ().function (), 0, x_, -1, *solver);
    }

    /// \internal
    /// CFSQP constraints function gradient.
    void gradcn (int nparam, int j,
                 double* x, double* gradgj, fct_t dummy, void* cd)
    {
      using namespace boost;

      assert (nparam >= 0 && !!x && !!gradgj && !!cd);

      CFSQPSolver* solver = static_cast<CFSQPSolver*> (cd);
      assert (j > 0 && solver->cfsqpConstraints ().size () - j >= 0);


      Function::vector_t x_ (nparam);
      array_to_vector (x_, x);

      Function::vector_t grad (nparam);

      // Decrement j to have C style indexation (0...size - 1).
      j--;
      // Constraint index in the generic representation.
      int j_ = solver->cfsqpConstraints ()[j].first;
      bool is_lower = solver->cfsqpConstraints ()[j].second;

      if (solver->problem ().constraints ()[j_].which () == CFSQPSolver::NONLINEAR)
        {
	  shared_ptr<DerivableFunction> f =
            get<shared_ptr<DerivableFunction> >
            (solver->problem ().constraints ()[j_]);
          grad = f->gradient (x_, 0);
	  CFSQPCheckGradient (*f, 0, x_, j_, *solver);
        }
      else
        {
          shared_ptr<LinearFunction> f =
            get<shared_ptr<LinearFunction> >
            (solver->problem ().constraints ()[j_]);
          grad = f->gradient (x_, 0);
	  CFSQPCheckGradient (*f, 0, x_, j_, *solver);
        }

      if (j < solver->nineq () && is_lower)
	grad = -grad;

      vector_to_array (gradgj, grad);
    }

  }

  CFSQPSolver::CFSQPSolver (const problem_t& pb, int iprint) throw ()
    : parent_t (pb),
      nineq_ (0),
      nineqn_ (0),
      neq_ (0),
      neqn_ (0),
      cfsqpConstraints_ (),
      invalidGradient_ (false)
  {
    // Initialize parameters.
    initializeParameters ();

    // Add non-linear inequalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == NONLINEAR)
	if (problem ().bounds ()[i].first != problem ().bounds ()[i].second)
	  {
	    if (problem ().bounds ()[i].first != -Function::infinity ())
	      cfsqpConstraints_.push_back (std::make_pair (i, true));
	    if (problem ().bounds ()[i].second != Function::infinity ())
	      cfsqpConstraints_.push_back (std::make_pair (i, false));
	  }
    nineqn_ = cfsqpConstraints_.size ();

    // Add linear inequalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == LINEAR)
	if (problem ().bounds ()[i].first != problem ().bounds ()[i].second)
	  {
	    if (problem ().bounds ()[i].first != -Function::infinity ())
	      cfsqpConstraints_.push_back (std::make_pair (i, true));
	    if (problem ().bounds ()[i].second != Function::infinity ())
	      cfsqpConstraints_.push_back (std::make_pair (i, false));
	  }
    nineq_ = cfsqpConstraints_.size ();

    // Add non-linear equalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == NONLINEAR)
        if (problem ().bounds ()[i].first == problem ().bounds ()[i].second)
          cfsqpConstraints_.push_back (std::make_pair (i, true));
    neqn_ = cfsqpConstraints_.size () - nineq_;

    // Add linear equalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == LINEAR)
        if (problem ().bounds ()[i].first == problem ().bounds ()[i].second)
          cfsqpConstraints_.push_back (std::make_pair (i, true));
    neq_ = cfsqpConstraints_.size () - nineq_;

    assert (nineq_ >= nineqn_);
    assert (neq_ >= neqn_);
  }

  CFSQPSolver::~CFSQPSolver () throw ()
  {
  }

  CFSQPSolver::CFSQPSolver (const CFSQPSolver& solver) throw ()
    : parent_t (solver.problem_),
      nineq_ (solver.nineq_),
      nineqn_ (solver.nineqn_),
      neq_ (solver.neq_),
      neqn_ (solver.neqn_),
      cfsqpConstraints_ (solver.cfsqpConstraints_),
      invalidGradient_ (solver.invalidGradient_)
  {}



#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)	\
  do {							\
    parameters_[KEY].description = DESCRIPTION;		\
    parameters_[KEY].value = VALUE;			\
  } while (0)

  void
  CFSQPSolver::initializeParameters () throw ()
  {
    parameters_.clear ();

    // Shared parameters.
    DEFINE_PARAMETER ("max-iterations", "number of iterations", 3000);

    // CFSQP specific.
    DEFINE_PARAMETER ("cfsqp.mode", "CFSQP mode", 100);
    DEFINE_PARAMETER ("cfsqp.iprint", "logging level", 0);
    DEFINE_PARAMETER ("cfsqp.bigbnd", "symbolizes infinity", 1e10);
    DEFINE_PARAMETER ("cfsqp.eps",
		      "final norm requirement for the Newton direction", 1e-8);

    DEFINE_PARAMETER ("cfsqp.epseqn",
		      "maximum violation of nonlinear equality constraint", 1e-8);

    DEFINE_PARAMETER ("cfsqp.udelta",
		      "perturbation size used in CFSQP finite differences algorithm", 1e-8);
  }

#undef DEFINE_PARAMETER


  void
  CFSQPSolver::initializeBounds (double* bl, double* bu) const throw ()
  {
    typedef problem_t::intervals_t::const_iterator citer_t;

    Function::size_type i = 0;
    for (citer_t it = problem ().argumentBounds ().begin ();
         it != problem ().argumentBounds ().end (); ++it)
      {
        bl[i] = (*it).first, bu[i] = (*it).second;
        ++i;
      }
  }


  void
  CFSQPSolver::fillConstraints (vector_t& constraints, double* g) const throw ()
  {
    constraints.resize (problem ().constraints ().size ());
    constraints.clear ();

    // Copy constraints final values from the CFSQP representation
    // to the generic representation.
    for (Function::size_type i = 0; i < cfsqpConstraints ().size (); ++i)
      {
	int j = cfsqpConstraints ()[i].first;
	assert (j >= 0 && problem ().constraints ().size () - j > 0);

	if (cfsqpConstraints ()[i].second)
	  // g(x) >= b, -g(x) + b <= 0
	  constraints[j] =
	    Function::getLowerBound (problem ().bounds ()[j]) - g[i];
	else
	  // g(x) <= b, g(x) - b <= 0
	  constraints[j] =
	    g[i] + Function::getUpperBound (problem ().bounds ()[j]);

      }
  }

  // Recopy lambda values for constraints only as expected by roboptim-core.
#define FILL_RESULT()					\
  res.value (0) = f[0];					\
  array_to_vector (res.x, x);				\
  fillConstraints (res.constraints, g);			\
  res.lambda.resize (neq_ + nineq_);			\
  array_to_vector (res.lambda, lambda+nparam+1);	\
  result_ = res

#define SWITCH_ERROR(NAME, ERROR)		\
  case NAME:					\
  result_ = SolverError (ERROR);		\
  break

#define SWITCH_WARNING(NAME, ERROR)			\
  case NAME:						\
  {							\
    ResultWithWarnings res (nparam, 1);			\
    SolverWarning warning (ERROR);			\
    res.warnings.push_back (warning);			\
    FILL_RESULT ();					\
  }							\
  break

#define MAP_CFSQP_ERRORS(MACRO)						\
  MACRO (1, "Infeasible guess for linear constraints.");		\
  MACRO (2, "Infeasible guess for linear and non-linear constraints.");	\
  MACRO (5, "Failed to construct d0.");					\
  MACRO (6, "Failed to construct d1.");					\
  MACRO (7, "Input data are not consistent.");				\
  MACRO (9, "One penalty parameter has exceeded bigbng.");

#define MAP_CFSQP_WARNINGS(MACRO)			\
  MACRO (3, "Max iteration has been reached.");		\
  MACRO (4, "Failed to find a new iterate.");		\
  MACRO (8, "New iterate equivalent to previous one.");



  void
  CFSQPSolver::solve () throw ()
  {
    using namespace detail;

    const int nparam = problem ().function ().inputSize ();
    const int nf = 1; //FIXME: only one objective function.
    const int nfsr = 0;

    const int ncsrl = 0;
    const int ncsrn = 0;
    int mesh_pts[1];
    int inform = 0;
    double* bl = (double*) malloc (nparam * sizeof (double));
    double* bu = (double*) malloc (nparam * sizeof (double));
    double* x = (double*) malloc (nparam * sizeof (double));
    double f[1];
    double* g = (double*) malloc ((nineq_ + neq_) * sizeof (double));
    double* lambda = (double*) malloc ((nparam + 1 + nineq_ + neq_) * sizeof (double));
    fct_t obj = detail::obj;
    fct_t constr = detail::constr;
    grad_t gradob = detail::gradob;
    grad_t gradcn = detail::gradcn;

    // Clear memory.
    memset (mesh_pts, 0, sizeof (int));
    memset (bl, 0, nparam * sizeof (double));
    memset (bu, 0, nparam * sizeof (double));
    memset (x, 0, nparam * sizeof (double));
    memset (f, 0, sizeof (double));
    memset (g, 0, (nineq_ + neq_) * sizeof (double));
    memset (lambda, 0, (nparam + 1 + nineq_ + neq_) * sizeof (double));

    // Initialize bounds.
    initializeBounds (bl, bu);

    // Copy starting point.
    if (!!problem ().startingPoint ())
      detail::vector_to_array (x, *problem ().startingPoint ());

    OFSQP myfsqp;

    // Retrieve parameters.
    const int& miter = getParameter<int> ("max-iterations");

    const int& mode = getParameter<int> ("cfsqp.mode");
    const int& iprint = getParameter<int> ("cfsqp.iprint");
    const double& bigbnd = getParameter<double> ("cfsqp.bigbnd");
    const double& eps = getParameter<double> ("cfsqp.eps");
    const double& epseqn = getParameter<double> ("cfsqp.epseqn");
    const double& udelta = getParameter<double> ("cfsqp.udelta");

    // Run optimization process.
    myfsqp.cfsqp (nparam, nf, nfsr, nineqn_, nineq_, neqn_, neq_, ncsrl,  ncsrn,
		  mesh_pts, mode,  iprint, miter, &inform, bigbnd, eps, epseqn,
		  udelta, bl, bu, x, f, g, lambda,
		  obj, constr, gradob, gradcn, this);

    if (invalidGradient_)
      result_ = SolverError ("gradient checks have failed.");
    else
      switch (inform)
	{
	  // Normal termination.
	case 0:
	  {
	    Result res (nparam, 1);
	    FILL_RESULT ();
	  }
	  break;

	  MAP_CFSQP_WARNINGS(SWITCH_WARNING);
	  MAP_CFSQP_ERRORS(SWITCH_ERROR);
	}

    free (lambda);
    free (g);
    free (x);
    free (bu);
    free (bl);
  }

#undef SWITCH_ERROR
#undef SWITCH_FATAL
#undef MAP_CFSQP_ERRORS
#undef MAP_CFSQP_WARNINGS
#undef FILL_RESULT



  const std::vector<std::pair<int, bool> >&
  CFSQPSolver::cfsqpConstraints () const throw ()
  {
    return cfsqpConstraints_;
  }

  const int&
  CFSQPSolver::nineqn () const throw ()
  {
    return nineqn_;
  }

  const int&
  CFSQPSolver::nineq () const throw ()
  {
    return nineq_;
  }

  const int&
  CFSQPSolver::neqn () const throw ()
  {
    return neqn_;
  }

  const int&
  CFSQPSolver::neq () const throw ()
  {
    return neq_;
  }

  void
  CFSQPSolver::invalidateGradient () throw ()
  {
    invalidGradient_ = true;
  }


  std::ostream&
  CFSQPSolver::print (std::ostream& o) const throw ()
  {
    parent_t::print (o);

    o << iendl << "CFSQP specific variables: " << incindent << iendl
      << "Nineq: " << nineq () << iendl
      << "Nineqn: " << nineqn () << iendl
      << "Neq: " << neq () << iendl
      << "Neqn: " << neqn () << iendl
      << "CFSQP constraints: " << cfsqpConstraints ();

    return o;
  }

} // end of namespace roboptim

extern "C"
{
  using namespace roboptim;
  typedef CFSQPSolver::parent_t solver_t;

  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ();
  ROBOPTIM_DLLEXPORT solver_t* create (const CFSQPSolver::problem_t&);
  ROBOPTIM_DLLEXPORT void destroy (solver_t*);


  ROBOPTIM_DLLEXPORT unsigned getSizeOfProblem ()
  {
    return sizeof (CFSQPSolver::problem_t);
  }

  ROBOPTIM_DLLEXPORT solver_t* create (const CFSQPSolver::problem_t& pb)
  {
    return new CFSQPSolver (pb);
  }

  ROBOPTIM_DLLEXPORT void destroy (solver_t* p)
  {
    delete p;
  }
}
