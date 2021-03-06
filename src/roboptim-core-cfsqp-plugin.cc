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

#include <boost/mpl/assert.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/size.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

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

#ifdef NDEBUG
# define DEBUG_ONLY(x)
#else
# define DEBUG_ONLY(x) x
#endif // NDEBUG

namespace roboptim
{
  typedef void (*fct_t) (int, int, double*, double*, void*);
  typedef void (*grad_t) (int, int, double*, double*, fct_t, void*);

  namespace detail
  {
    ROBOPTIM_DLLLOCAL void obj (int, int, double*, double*, void*);
    ROBOPTIM_DLLLOCAL double evaluate_inequality (double, bool, double, double);
    ROBOPTIM_DLLLOCAL void constr (int, int, double*, double*, void*);
    ROBOPTIM_DLLLOCAL void gradob (int, int, double*, double*, fct_t, void*);
    ROBOPTIM_DLLLOCAL void gradcn (int, int, double*, double*, fct_t, void*);


    ROBOPTIM_DLLLOCAL void CFSQPCheckGradient
    (const DerivableFunction& function,
     unsigned functionId,
     Function::vector_t& x,
     int constraintId,
     CFSQPSolver& solver);

    /// \internal
#ifdef ROBOPTIM_CORE_CFSQP_PLUGIN_CHECK_GRADIENT
    void CFSQPCheckGradient (const DerivableFunction& function,
			     unsigned functionId,
			     Function::vector_t& x,
			     int constraintId,
			     CFSQPSolver& solver)
    {
      using boost::format;
      try
	{
	  checkGradientAndThrow (function, functionId, x, 1.);
	}
      catch (BadGradient<EigenMatrixDense>& bg)
	{
	  solver.invalidateGradient ();
	  std::cerr
	    << ((constraintId < 0)
		? "Invalid cost function gradient:"
		: (format ("Invalid constraint function gradient (id = %1%):")
		   % constraintId).str ())
	    << std::endl
	    << function.getName ()
	    << std::endl
	    << bg
	    << std::endl;
	}
    }
#else
    void CFSQPCheckGradient (const DerivableFunction&,
			     unsigned,
			     Function::vector_t&,
			     int,
			     CFSQPSolver&)
    {}
#endif //!ROBOPTIM_CORE_CFSQP_PLUGIN_CHECK_GRADIENT


    struct ComputeConstraintsSizeVisitor
      : public boost::static_visitor<Function::size_type>
    {
      template <typename U>
      Function::size_type operator () (const U& constraint)
      {
	return constraint->outputSize ();
      }
    };

    /// \internal
    ////
    /// Compute the total size of the constraints.
    template  <typename T>
    Function::size_type
    computeConstraintsOutputSize (const T& pb)
    {
      Function::size_type result = 0;
      ComputeConstraintsSizeVisitor visitor;
      for (std::size_t i = 0; i < pb.constraints ().size (); ++i)
	result += boost::apply_visitor (visitor, pb.constraints ()[i]);
      return result;
    }

    /// \internal
    ///
    /// Count the number of constraints bounds.
    template <typename T>
    Function::size_type
    computeBoundsSize (const T& problem)
    {
      Function::size_type res = 0;
      for (unsigned i = 0; i < problem.boundsVector ().size (); ++i)
	res += problem.boundsVector ()[i].size ();
      return res;
    }

    /// \internal
    /// CFSQP objective function.
    void obj (int nparam, int, double* x, double* fj, void* cd)
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

      assert (j > 0 &&
	      solver->cfsqpConstraints ().size () >=
	      static_cast<unsigned int>(j));

      Function::vector_t x_ (nparam);
      array_to_vector (x_, x);

      // Decrement j to have C style indexation (0...size - 1).
      j--;

      // Constraint index in the generic representation.
      std::size_t j_ = static_cast<std::size_t> (j);
      std::pair<int, int> constraintId = solver->cfsqpConstraints ()[j_].first;

      assert (constraintId.first >= 0 && constraintId.second >= 0);
      assert (solver->cfsqpConstraints ().size () - constraintId.first > 0);
      assert (solver->problem ().boundsVector ().size ()
	      - constraintId.first > 0);
      assert (solver->problem ().boundsVector ()[constraintId.first].size ()
	      - constraintId.second > 0);

      if (0 <= j && j < solver->nineqn ())
        {
	  shared_ptr<DerivableFunction> f =
	    get<shared_ptr<DerivableFunction> >
            (solver->problem ().constraints ()[constraintId.first]);
	  assert (f);

	  //FIXME: this is inefficient.
          double res = (*f) (x_)[constraintId.second];

          *gj = evaluate_inequality
            (res,
             solver->cfsqpConstraints ()[j].second,
             solver->problem ().boundsVector ()
	     [constraintId.first][constraintId.second].first,
             solver->problem ().boundsVector ()
	     [constraintId.first][constraintId.second].second);
          return;
        }

      if (solver->nineqn () <= j && j < solver->nineq ())
        {
	  shared_ptr<LinearFunction> f =
            get<shared_ptr<LinearFunction> >
            (solver->problem ().constraints ()[constraintId.first]);
	  assert (f);

	  //FIXME: this is inefficient.
          double res = (*f) (x_)[constraintId.second];

          *gj = evaluate_inequality
            (res,
             solver->cfsqpConstraints ()[j].second,
             solver->problem ().boundsVector ()
	     [constraintId.first][constraintId.second].first,
             solver->problem ().boundsVector ()
	     [constraintId.first][constraintId.second].second);
          return;
        }

      j -= solver->nineq ();
      assert (j >= 0);
      if (0 <= j && j < solver->neqn ())
        {
          shared_ptr<DerivableFunction> f =
	    get<shared_ptr<DerivableFunction> >
            (solver->problem ().constraints ()[constraintId.first]);
	  assert (f);

	  //FIXME: this is inefficient.
          double res = (*f) (x_)[constraintId.second];
          *gj = res - solver->problem ().boundsVector ()
	    [constraintId.first][constraintId.second].first;
          return;
        }

      if (solver->neqn () <= j && j < solver->neq ())
        {
          shared_ptr<LinearFunction> f =
            get<shared_ptr<LinearFunction> >
            (solver->problem ().constraints ()[constraintId.first]);
	  assert (f);

	  //FIXME: this is inefficient.
          double res = (*f) (x_)[constraintId.second];
          *gj = res - solver->problem ().boundsVector ()
	    [constraintId.first][constraintId.second].first;
          return;
        }
      assert (0);
    }

    /// \internal
    /// CFSQP objective function gradient.
    void gradob (int nparam, int DEBUG_ONLY(j),
                 double* x, double* gradf, fct_t, void* cd)
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
                 double* x, double* gradgj, fct_t, void* cd)
    {
      using namespace boost;

      assert (nparam >= 0 && !!x && !!gradgj && !!cd);

      CFSQPSolver* solver = static_cast<CFSQPSolver*> (cd);
      assert (j > 0 &&
	      solver->cfsqpConstraints ().size () >=
	      static_cast<unsigned int>(j));


      Function::vector_t x_ (nparam);
      array_to_vector (x_, x);

      Function::vector_t grad (nparam);

      // Decrement j to have C style indexation (0...size - 1).
      j--;
      // Constraint index in the generic representation.
      std::pair<int, int> constraintId = solver->cfsqpConstraints ()[j].first;
      bool is_lower = solver->cfsqpConstraints ()[j].second;

      if (solver->problem ().constraints ()[constraintId.first].which ()
	  == CFSQPSolver::NONLINEAR)
        {
	  shared_ptr<DerivableFunction> f =
            get<shared_ptr<DerivableFunction> >
            (solver->problem ().constraints ()[constraintId.first]);
          grad = f->gradient (x_, constraintId.second);
	  CFSQPCheckGradient (*f, 0, x_, constraintId.first, *solver);
        }
      else
        {
          shared_ptr<LinearFunction> f =
            get<shared_ptr<LinearFunction> >
            (solver->problem ().constraints ()[constraintId.first]);
          grad = f->gradient (x_, constraintId.second);
	  CFSQPCheckGradient (*f, 0, x_, constraintId.first, *solver);
        }

      if (j < solver->nineq () && is_lower)
	grad = -grad;

      vector_to_array (gradgj, grad);
    }

  }

  CFSQPSolver::CFSQPSolver (const problem_t& pb, int)
    : parent_t (pb),
      nineq_ (0),
      nineqn_ (0),
      neq_ (0),
      neqn_ (0),
      cfsqpConstraints_ (),
      invalidGradient_ (false),
      callback_ (),
      solverState_ (this->problem ())
  {
    // Initialize parameters.
    initializeParameters ();

    // Add non-linear inequalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == NONLINEAR)
	for (unsigned j = 0; j < problem ().boundsVector ()[i].size (); ++j)
	  if (problem ().boundsVector ()[i][j].first
	      != problem ().boundsVector ()[i][j].second)
	    {
	      if (problem ().boundsVector ()[i][j].first
		  != -Function::infinity ())
		cfsqpConstraints_.push_back
		  (std::make_pair (std::make_pair (i, j), true));
	      if (problem ().boundsVector ()[i][j].second
		  != Function::infinity ())
		cfsqpConstraints_.push_back
		  (std::make_pair (std::make_pair (i, j), false));
	    }
    nineqn_ = static_cast<int> (cfsqpConstraints_.size ());

    // Add linear inequalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == LINEAR)
	for (unsigned j = 0; j < problem ().boundsVector ()[i].size (); ++j)
	  if (problem ().boundsVector ()[i][j].first
	      != problem ().boundsVector ()[i][j].second)
	    {
	      if (problem ().boundsVector ()[i][j].first
		  != -Function::infinity ())
		cfsqpConstraints_.push_back
		  (std::make_pair (std::make_pair (i, j), true));
	      if (problem ().boundsVector ()[i][j].second
		  != Function::infinity ())
		cfsqpConstraints_.push_back
		  (std::make_pair (std::make_pair (i, j), false));
	    }
    nineq_ = static_cast<int> (cfsqpConstraints_.size ());

    // Add non-linear equalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == NONLINEAR)
	for (unsigned j = 0; j < problem ().boundsVector ()[i].size (); ++j)
	  if (problem ().boundsVector ()[i][j].first
	      == problem ().boundsVector ()[i][j].second)
	    cfsqpConstraints_.push_back
	      (std::make_pair (std::make_pair (i, j), true));
    neqn_ = static_cast<int> (cfsqpConstraints_.size ()) - nineq_;

    // Add linear equalities.
    for (unsigned i = 0; i < problem ().constraints ().size (); ++i)
      if (problem ().constraints ()[i].which () == LINEAR)
	for (unsigned j = 0; j < problem ().boundsVector ()[i].size (); ++j)
	  if (problem ().boundsVector ()[i][j].first
	      == problem ().boundsVector ()[i][j].second)
	    cfsqpConstraints_.push_back
	      (std::make_pair (std::make_pair (i, j), true));
    neq_ = static_cast<int> (cfsqpConstraints_.size ()) - nineq_;

    assert (nineq_ >= nineqn_);
    assert (neq_ >= neqn_);
  }

  CFSQPSolver::~CFSQPSolver ()
  {
  }


#define DEFINE_PARAMETER(KEY, DESCRIPTION, VALUE)	\
  do {							\
    parameters_[KEY].description = DESCRIPTION;		\
    parameters_[KEY].value = VALUE;			\
  } while (0)

  void
  CFSQPSolver::initializeParameters ()
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

    DEFINE_PARAMETER ("cfsqp.objeps", "N/A", 0.);
    DEFINE_PARAMETER ("cfsqp.objrep", "N/A", 0.);
    DEFINE_PARAMETER ("cfsqp.gLgeps", "N/A", 0.);
    DEFINE_PARAMETER ("cfsqp.nstop", "N/A", 0);
  }

#undef DEFINE_PARAMETER


  void
  CFSQPSolver::initializeBounds (double* bl, double* bu) const
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
  CFSQPSolver::fillConstraints (vector_t& constraints, double* g) const
  {
    constraints.resize (detail::computeConstraintsOutputSize (problem ()));
    constraints.setZero ();

    detail::ComputeConstraintsSizeVisitor visitor;

    // Copy constraints final values from the CFSQP representation
    // to the generic representation.
    for (std::size_t i = 0; i < cfsqpConstraints ().size (); ++i)
      {
	std::size_t constraintId = cfsqpConstraints ()[i].first.first;
	std::size_t functionId = cfsqpConstraints ()[i].first.second;
	assert (problem ().constraints ().size () - constraintId > 0);
	bool is_lower = cfsqpConstraints ()[i].second;

	std::size_t index = 0;
	for (std::size_t j = 0; j < constraintId; ++j)
	  index += boost::apply_visitor (visitor, problem ().constraints ()[j]);
	index += functionId;

	if (is_lower)
	  // g(x) >= b, -g(x) + b <= 0
	  constraints[index] =
	    Function::getLowerBound
	    (problem ().boundsVector ()[constraintId][functionId]) - g[i];
	else
	  // g(x) <= b, g(x) - b <= 0
	  constraints[index] =
	    g[i] + Function::getUpperBound
	    (problem ().boundsVector ()[constraintId][functionId]);

	++functionId;
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

#define SWITCH_WARNING(NAME, ERROR)             \
  case NAME:                                    \
  {                                             \
    ResultWithWarnings res (nparam, 1);         \
    SolverWarning warning (ERROR);              \
    res.warnings.push_back (warning);           \
    FILL_RESULT ();                             \
  }                                             \
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
  CFSQPSolver::solve ()
  {
    using namespace detail;

    const int nparam = static_cast<int> (problem ().function ().inputSize ());
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

    OFSQP::callback_t cb =
      boost::bind
      (&CFSQPSolver::perIterationCallback, this, boost::arg<1> ());
    OFSQP myfsqp (cb);

    // Retrieve parameters.
    const int& miter = getParameter<int> ("max-iterations");

    const int& mode = getParameter<int> ("cfsqp.mode");
    const int& iprint = getParameter<int> ("cfsqp.iprint");
    const double& bigbnd = getParameter<double> ("cfsqp.bigbnd");
    const double& eps = getParameter<double> ("cfsqp.eps");
    const double& epseqn = getParameter<double> ("cfsqp.epseqn");
    const double& udelta = getParameter<double> ("cfsqp.udelta");

    const double& objeps = getParameter<double> ("cfsqp.objeps");
    const double& objrep = getParameter<double> ("cfsqp.objrep");
    const double& gLgeps = getParameter<double> ("cfsqp.gLgeps");
    const int& nstop = getParameter<int> ("cfsqp.nstop");

    myfsqp.objeps = objeps;
    myfsqp.objrep = objrep;
    myfsqp.gLgeps = gLgeps;
    myfsqp.nstop = nstop;

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



  const std::vector<std::pair<std::pair<int, int>, bool> >&
  CFSQPSolver::cfsqpConstraints () const
  {
    return cfsqpConstraints_;
  }

  const int&
  CFSQPSolver::nineqn () const
  {
    return nineqn_;
  }

  const int&
  CFSQPSolver::nineq () const
  {
    return nineq_;
  }

  const int&
  CFSQPSolver::neqn () const
  {
    return neqn_;
  }

  const int&
  CFSQPSolver::neq () const
  {
    return neq_;
  }

  void
  CFSQPSolver::invalidateGradient ()
  {
    invalidGradient_ = true;
  }


  std::ostream&
  CFSQPSolver::print (std::ostream& o) const
  {
    parent_t::print (o);

    // Increase indent level to make it look like this is
    // part of the solver's parameters.
    o << incindent;

    o << iendl << "CFSQP specific variables:" << incindent;

    o << iendl << "Nineq: " << nineq ()
      << iendl << "Nineqn: " << nineqn ()
      << iendl << "Neq: " << neq ()
      << iendl << "Neqn: " << neqn ()
      << iendl << "CFSQP constraints: " << cfsqpConstraints ();

    o << decindent << decindent;

    return o;
  }

  void
  CFSQPSolver::setIterationCallback (callback_t callback)
  {
    callback_ = callback;
  }

  void
  CFSQPSolver::perIterationCallback (const double* x)
  {
    if (!callback_)
      return;
    vector_t x_ = Eigen::Map<const Eigen::VectorXd>
      (x, this->problem ().function ().inputSize ());
    this->solverState_.x () = x_;
    this->callback_ (this->problem (), this->solverState_);
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

  ROBOPTIM_DLLEXPORT const char* getTypeIdOfConstraintsList ()
  {
    return typeid (CFSQPSolver::problem_t::constraintsList_t).name ();
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
