// Copyright (C) 2008, 2009 by Adrien Escande, Francois Keith,
// Abderrahmane Kheddar, Thomas Moulard, AIST, CNRS, INRIA.
//
// This file is part of the roboptim.
//
// CAUTION: this file is a derivative work of CFSQP.
// It is *not* free software. See LICENSE.CFSQP for more details.

#ifndef OFSQP_HH
# define OFSQP_HH
# include <cstdio>
# include <cmath>
# include <algorithm>
# include <cstdlib>

/* As MSVC does not declare properly __STDC__, let's define
   it manually. */
# ifndef __STDC__
#  ifdef _WIN32
#   define __STDC__
#  endif //! WIN32
# endif //! __STDC__

# ifndef __STDC__
#  ifdef apollo
extern char *calloc();
# else
#  include <malloc.h>
# endif //! apollo
#endif //! __STDC__



/***************************************************************/
/*     Macros                                                  */
/***************************************************************/

# define DMAX1(a, b) ((a) > (b) ? (a) : (b))
# define DMIN1(a, b) ((a) < (b) ? (a) : (b))
# ifndef TRUE
#  define TRUE 1
# endif //! TRUE
# ifndef FALSE
#  define FALSE 0
# endif //! FALSE

class OFSQP
{
public:
  OFSQP(void);
  ~OFSQP(void);

  static bool isXNew(void);
  static void resetNewX(void);

  int __numIteration;

private:
  static const int NONE = 0;
  static const int OBJECT = 1;
  static const int CONSTR = 2;

  struct _objective {
    double val;
    double *grad;
    double mult;
    double mult_L; /* mode A=1 */
    int act_sip;   /* SIP      */
  };

  struct _constraint {
    double val;
    double *grad;
    double mult;
    int act_sip;   /* SIP      */
    int d1bind;    /* SR constraints  */
  };

  struct _parameter {
    double *x;
    double *bl;
    double *bu;
    double *mult;
    void *cd;      /* Client data pointer */
  };

  struct _violation {    /* SIP      */
    int type;
    int index;
  };

  struct {
    int nnineq, M, ncallg, ncallf, mode, modec;
    int tot_actf_sip,tot_actg_sip,nfsip,ncsipl,ncsipn; /* SIP */
  } glob_info;

  struct {
    int iprint,info,ipd,iter,initvl,iter_mod;
    FILE *io;
  } glob_prnt;

  struct {
    double epsmac,rteps,udelta,valnom;
  } glob_grd;

  struct {
    int dlfeas,local,update,first,rhol_is1,d0_is0,get_ne_mult;
  } glob_log;

  /**************************************************************/
  /*     Gradients - Finite Difference                          */
  /**************************************************************/

#ifdef __STDC__
  void    grobfdSort(
		     void (*g)(int,int,double *,double *,
			       void (*)(int,int,double *,double *,void *),void *),
		     int,int,double *,double *,void (*)(int,int,
							double *,double *,void *),void *);
  void    grcnfdSort(
		     void (*g)(int,int,double *,double *,
			       void (*)(int,int,double *,double *,void *),void *),
		     int,int,double *,double *,void (*)(int,int,
							double *,double *,void *),void *);
#else
  void    grobfd();
  void    grcnfd();
#endif

public :
  /**************************************************************/
  /*     Gradients - Finite Difference                          */
  /**************************************************************/

#ifdef __STDC__
  void    grobfd(int,int,double *,double *,void (*)(int,int,
						    double *,double *,void *),void *);
  void    grcnfd(int,int,double *,double *,void (*)(int,int,
						    double *,double *,void *),void *);
#else
  void    grobfd();
  void    grcnfd();
#endif

  /**************************************************************/
  /*     Prototype for CFSQP -   	                        */
  /**************************************************************/

#ifdef __STDC__
  void
  cfsqp(int nparam,int nf,int nfsr,int nineqn,int nineq,int neqn,
	int neq,int ncsrl,int ncsrn,int *mesh_pts,
	int mode,int iprint,int miter,int *inform,double bigbnd,
	double eps,double epseqn,double udelta,double *bl,double *bu,
	double *x,double *f,double *g,double *lambda,
	void (*obj)(int, int, double *, double *,void *),
	void (*constr)(int,int,double *,double *,void *),
	void (*gradob)(int,int,double *,double *,
		       void (*)(int,int,double *,double *,void *),void *),
	void (*gradcn)(int,int,double *,double *,
		       void (*)(int,int,double *,double *,void *),void *),
	void *cd);
#else
  void    cfsqp();
#endif


  /****RobOptim : change form char[] to char* ****/
  const char* cfsqp_version;
  double  bgbnd,tolfea;
  int  maxit;

  /* Declare and initialize user-accessible flag indicating    */
  /* whether x sent to user functions has been changed within  */
  /* CFSQP.				 		     */
  static int x_is_new;

  /* Declare and initialize user-accessible stopping criterion */
  double objeps;
  double objrep;
  double gLgeps;
  int nstop;

  /* Workspace                                                     */
  int     *iw;
  double  *w;
  int     lenw, leniw;

  /***************************************************************/
  /*     Memory Utilities                                        */
  /***************************************************************/
#ifdef __STDC__
  static int      *make_iv(int);
  static double   *make_dv(int);
  static double   **make_dm(int, int);
  static void     free_iv(int *);
  static void     free_dv(double *);
  static void     free_dm(double **, int);
  static double   *convert(double **, int, int);
#else
  static int      *make_iv();
  static double   *make_dv();
  static double   **make_dm();
  static void     free_iv();
  static void     free_dv();
  static void     free_dm();
  static double   *convert();
#endif

  /***************************************************************/
  /*     Utility Subroutines                                     */
  /***************************************************************/

#ifdef __STDC__
  static void     diagnl(int, double, double **);
  void			error(const char string[],int *);
  void
  estlam(int,int,int *,double,double **,double *,double *,double *,
	 struct _constraint *,double *,double *,double *,double *);
  static double   *colvec(double **,int,int);
  static double   scaprd(int,double *,double *);
  static double   small(void);
  static int      fuscmp(double,double);
  static int      indexs(int,int);
  static void     matrcp(int,double **,int,double **);
  static void     matrvc(int,int,double **,double *,double *);
  static void     nullvc(int,double *);
  void
  resign(int,int,double *,double *,double *,struct _constraint *,
	 double *,int,int);
  static void     sbout1(FILE *,int,const char *,double,double *,int,int);
  static void     sbout2(FILE *,int,int,const char *,const char *,double *);
  static void     shift(int,int,int *);
  double
  slope(int,int,int,int,int,struct _objective *,double *,double *,
	double *,double,double,int,double *,int);
  static int      element(int *,int,int);
#else
  static void     diagnl();
  void			error();
  void			estlam();
  static double   *colvec();
  static double   scaprd();
  static double   small();
  static int      fuscmp();
  static int      indexs();
  static void     matrcp();
  static void     matrvc();
  static void     nullvc();
  void			resign();
  static void     sbout1();
  static void     sbout2();
  static void     shift();
  double			slope();
  static int      element();
#endif


  /**************************************************************/
  /*     Main routines for optimization -                       */
  /**************************************************************/

#ifdef __STDC__
  void
  cfsqp1(int,int,int,int,int,int,int,int,int,int,int *,int,
	 int,int,int,double,double,int *,int *,struct _parameter *,
	 struct _constraint *,struct _objective *,double *,
	 void (*)(int,int,double *,double *,void *),
	 void (*)(int,int,double *,double *,void *),
	 void (*)(int,int,double *,double *,
		  void (*)(int,int,double *,double *,void *),void *),
	 void (*)(int,int,double *,double *,
		  void (*)(int,int,double *,double *,void *),void *));
  void
  check(int,int,int,int *,int,int,int,int,int,int,int,int *,double,
	double,struct _parameter *);
  void
  initpt(int,int,int,int,int,int,int,struct _parameter *,
	 struct _constraint *,void (*)(int,int,double *,double *,void *),
	 void (*)(int,int,double *,double *,
		  void (*)(int,int,double *,double *,void *),void *));
  void
  dir(int,int,int,int,int,int,int,int,int,int,int,int,double *,
      double,double,double *,double *,double,double *,double *,int *,
      int *,int *,int *,int *,int *,struct _parameter *,double *,
      double *,struct _constraint *,struct _objective *,double *,
      double *,double *,double *,double *,double *,double **,double *,
      double *,double *,double *,double **,double **,double *,
      double *,struct _violation *,void (*)(int,int,double *,double *,
					    void *),void (*)(int,int,double *,double *,void *));
  void
  step1(int,int,int,int,int,int,int,int,int,int,int,int *,int *,int *,
	int *,int *,int *,int *,int *,int,double,struct _objective *,
	double *,double *,double *,double *,double *,double *,double *,
	double *,double *,double *,double *,struct _constraint *,
	double *,double *,struct _violation *viol,
	void (*)(int,int,double *,double *,void *),
	void (*)(int,int,double *,double *,void *),void *);
  void
  hessian(int,int,int,int,int,int,int,int,int,int,int,int *,int,
	  double *,struct _parameter *,struct _objective *,
	  double,double *,double *,double *,double *,double *,
	  struct _constraint *,double *,int *,int *,double *,
	  double *,double *,double **,double *,double,int *,
	  double *,double *,void (*)(int,int,double *,double *,void *),
	  void (*)(int,int,double *,double *,void *),
	  void (*)(int,int,double *,double *,
		   void (*)(int,int,double *,double *,void *),void *),
	  void (*)(int,int,double *,double *,
		   void (*)(int,int,double *,double *,void *),void *),
	  double **,double *,double *,struct _violation *);
  void
  out(int,int,int,int,int,int,int,int,int,int,int,int *,double *,
      struct _constraint *,struct _objective *,double,
      double,double,double,double,int);
  void
  update_omega(int,int,int,int *,int,int,int,int,double,double,
	       struct _constraint *,struct _objective *,double *,
	       struct _violation *,void (*)(int,int,double *,double *,
					    void *),void (*)(int,int,double *,double *,void *),
	       void (*)(int,int,double *,double *,
			void (*)(int,int,double *,double *,void *),void *),
	       void (*)(int,int,double *,double *,
			void (*)(int,int,double *,double *,void *),void *),
	       void *,int);
#else
  void			cfsqp1();
  void			check();
  void			initpt();
  void			dir();
  void			step1();
  void			hessian();
  void			out();
  void			update_omega();
#endif

#ifdef __STDC__
  static void
  dealloc(int,int,double *,int *,int *,struct _constraint *cs,
	  struct _parameter *);
#else
  static void dealloc();
#endif

#ifdef __STDC__
  static void
  dealloc1(int,int,double **,double **,double **,double *,double *,
	   double *,double *,double *,double *,double *,double *,
	   double *,double *,double *,double *,int *,int *,int *);
#else
  static void dealloc1();
#endif

#ifdef __STDC__
  void
  dqp(int,int,int,int,int,int,int,int,int,int,int,int,int,
      int,int,int *,struct _parameter *,double *,int,
      struct _objective *,double,double *,struct _constraint *,
      double **,double *,double *,double *,double *,
      double **,double **,double *,double,int);
  void
  di1(int,int,int,int,int,int,int,int,int,int,int,int,int *,
      int,struct _parameter *,double *,struct _objective *,
      double,double *,struct _constraint *,double *,
      double *,double *,double *,double **,double *,double);
#else
  void		dqp();
  void		di1();
#endif


  /************************************************************/
  /*    Utility functions used by CFSQP -                     */
  /*    Available functions:                                  */
  /*      diagnl        error         estlam                  */
  /*      colvec        scaprd        small                   */
  /*      fool          matrvc        matrcp                  */
  /*      nullvc        resign        sbout1                  */
  /*      sbout2        shift         slope                   */
  /*      fuscmp        indexs        element                 */
  /************************************************************/

#ifdef __STDC__
  static void fool(double, double, double *);
#else
  static void fool();
#endif



  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  /****************************************************************
   *						       QLD define *
   ***************************************************************/
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /*  -- translated by f2c (version of 22 July 1992  22:54:52).
   */

  /* umd
     Must include math.h before f2c.h - f2c does a #define abs.
     (Thanks go to Martin Wauchope for providing this correction)
     We manually included AT&T's f2c.h in this source file, i.e.
     it does not have to be present separately in order to compile.
  */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

  typedef int integer;
  typedef char *address;
  typedef short int shortint;
  typedef float real;
  typedef double doublereal;
  typedef struct { real r, i; } complex;
  typedef struct { doublereal r, i; } doublecomplex;
  typedef long int logical;
  typedef short int shortlogical;

#define TRUE_ (1)
#define FALSE_ (0)

  /* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

  /* I/O stuff */

#ifdef f2c_i2
  /* for -i2 */
  typedef short flag;
  typedef short ftnlen;
  typedef short ftnint;
#else
  typedef long flag;
  typedef long ftnlen;
  typedef long ftnint;
#endif


  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  /****************************************************************
   *					QLD struct definitions	  *
   ***************************************************************/
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */



  /*external read, write*/
  typedef struct
  {	flag cierr;
    ftnint ciunit;
    flag ciend;
    char *cifmt;
    ftnint cirec;
  } cilist;

  /*internal read, write*/
  typedef struct
  {	flag icierr;
    char *iciunit;
    flag iciend;
    char *icifmt;
    ftnint icirlen;
    ftnint icirnum;
  } icilist;

  /*open*/
  typedef struct
  {	flag oerr;
    ftnint ounit;
    char *ofnm;
    ftnlen ofnmlen;
    char *osta;
    char *oacc;
    char *ofm;
    ftnint orl;
    char *oblnk;
  } olist;

  /*close*/
  typedef struct
  {	flag cerr;
    ftnint cunit;
    char *csta;
  } cllist;

  /*rewind, backspace, endfile*/
  typedef struct
  {	flag aerr;
    ftnint aunit;
  } alist;

  /* inquire */
  typedef struct
  {	flag inerr;
    ftnint inunit;
    char *infile;
    ftnlen infilen;
    ftnint	*inex;	/*parameters in standard's order*/
    ftnint	*inopen;
    ftnint	*innum;
    ftnint	*innamed;
    char	*inname;
    ftnlen	innamlen;
    char	*inacc;
    ftnlen	inacclen;
    char	*inseq;
    ftnlen	inseqlen;
    char 	*indir;
    ftnlen	indirlen;
    char	*infmt;
    ftnlen	infmtlen;
    char	*inform;
    ftnint	informlen;
    char	*inunf;
    ftnlen	inunflen;
    ftnint	*inrecl;
    ftnint	*innrec;
    char	*inblank;
    ftnlen	inblanklen;
  } inlist;


#define VOID void

  union Multitype {	/* for multiple entry points */
    shortint h;
    integer i;
    real r;
    doublereal d;
    complex c;
    doublecomplex z;
  };

  typedef union Multitype Multitype;

  typedef long Long;

  struct Vardesc {	/* for Namelist */
    char *name;
    char *addr;
    Long *dims;
    int  type;
  };
  typedef struct Vardesc Vardesc;

  struct Namelist {
    char *name;
    Vardesc **vars;
    int nvars;
  };
  typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal)abs(x)
#define dmin(a,b) (doublereal)std::min(a,b)
#define dmax(a,b) (doublereal)std::max(a,b)

  /* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
  typedef int /* Unknown procedure type */ (*U_fp)(...);
  typedef shortint (*J_fp)(...);
  typedef integer (*I_fp)(...);
  typedef real (*R_fp)(...);
  typedef doublereal (*D_fp)(...), (*E_fp)(...);
  typedef /* Complex */ VOID (*C_fp)(...);
  typedef /* Double Complex */ VOID (*Z_fp)(...);
  typedef logical (*L_fp)(...);
  typedef shortlogical (*K_fp)(...);
  typedef /* Character */ VOID (*H_fp)(...);
  typedef /* Subroutine */ int (*S_fp)(...);
#else
  typedef int /* Unknown procedure type */ (*U_fp)();
  typedef shortint (*J_fp)();
  typedef integer (*I_fp)();
  typedef real (*R_fp)();
  typedef doublereal (*D_fp)(), (*E_fp)();
  typedef /* Complex */ VOID (*C_fp)();
  typedef /* Double Complex */ VOID (*Z_fp)();
  typedef logical (*L_fp)();
  typedef shortlogical (*K_fp)();
  typedef /* Character */ VOID (*H_fp)();
  typedef /* Subroutine */ int (*S_fp)();
#endif	//__cplusplus
	/* E_fp is for real functions when -R is not specified */
  typedef VOID C_f;	/* complex function */
  typedef VOID H_f;	/* character function */
  typedef VOID Z_f;	/* double complex function */
  typedef doublereal E_f;	/* real function with -R not specified */

  /* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
#undef cray
#undef gcos
#undef mc68010
#undef mc68020
#undef mips
#undef pdp11
#undef sgi
#undef sparc
#undef sun
#undef sun2
#undef sun3
#undef sun4
#undef u370
#undef u3b
#undef u3b2
#undef u3b5
#undef unix
#undef vax
#endif	//Skip_f2c_Undefs
#endif	//F2C_INCLUDE



	/* Common Block Declarations */

  struct {
    doublereal eps;
  } cmache_;

#define cmache_1 cmache_

  /* Table of constant values */

  /****RobOptim : add const keyword****/
  static const integer c__1 = 1;




  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
  /****************************************************************
   *					QLD function definitions  *
   ***************************************************************/
  /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

  /* umd */
  /*
    ql0002_ is declared here to provide ANSI C compliance.
    (Thanks got to Martin Wauchope for providing this correction)
  */
#ifdef __STDC__

  int ql0002_(integer *n,integer *m,integer *meq,integer *mmax,
	      integer *mn,integer *mnn,integer *nmax,
	      logical *lql,
	      doublereal *a,doublereal *b,doublereal *grad,
	      doublereal *g,doublereal *xl,doublereal *xu,doublereal *x,
	      integer *nact,integer *iact,integer *maxit,
	      doublereal *vsmall,
	      integer *info,
	      doublereal *diag, doublereal *w,
	      integer *lw);
#else
  int ql0002_();
#endif
  /* umd */
  /*
    When the fortran code was f2c converted, the use of fortran COMMON
    blocks was no longer available. Thus an additional variable, eps1,
    was added to the parameter list to account for this.
  */
  /* umd */
  /*
    Two alternative definitions are provided in order to give ANSI
    compliance.
  */

  /* CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
     !!!! NOTICE !!!!

     1. The routines contained in this file are due to Prof. K.Schittkowski
     of the University of Bayreuth, Germany (modification of routines
     due to Prof. MJD Powell at the University of Cambridge).  They can
     be freely distributed.

     2. A few minor modifications were performed at the University of
     Maryland. They are marked in the code by "umd".

     A.L. Tits, J.L. Zhou, and
     Craig Lawrence
     University of Maryland

     ***********************************************************************



     SOLUTION OF QUADRATIC PROGRAMMING PROBLEMS



     QL0001 SOLVES THE QUADRATIC PROGRAMMING PROBLEM

     MINIMIZE        .5*X'*C*X + D'*X
     SUBJECT TO      A(J)*X  +  B(J)   =  0  ,  J=1,...,ME
     A(J)*X  +  B(J)  >=  0  ,  J=ME+1,...,M
     XL  <=  X  <=  XU

     HERE C MUST BE AN N BY N SYMMETRIC AND POSITIVE MATRIX, D AN N-DIMENSIONAL
     VECTOR, A AN M BY N MATRIX AND B AN M-DIMENSIONAL VECTOR. THE ABOVE
     SITUATION IS INDICATED BY IWAR(1)=1. ALTERNATIVELY, I.E. IF IWAR(1)=0,
     THE OBJECTIVE FUNCTION MATRIX CAN ALSO BE PROVIDED IN FACTORIZED FORM.
     IN THIS CASE, C IS AN UPPER TRIANGULAR MATRIX.

     THE SUBROUTINE REORGANIZES SOME DATA SO THAT THE PROBLEM CAN BE SOLVED
     BY A MODIFICATION OF AN ALGORITHM PROPOSED BY POWELL (1983).


     USAGE:

     QL0001(M,ME,MMAX,N,NMAX,MNN,C,D,A,B,XL,XU,X,U,IOUT,IFAIL,IPRINT,
     WAR,LWAR,IWAR,LIWAR)


     DEFINITION OF THE PARAMETERS:

     M :        TOTAL NUMBER OF CONSTRAINTS.
     ME :       NUMBER OF EQUALITY CONSTRAINTS.
     MMAX :     ROW DIMENSION OF A. MMAX MUST BE AT LEAST ONE AND GREATER
     THAN M.
     N :        NUMBER OF VARIABLES.
     NMAX :     ROW DIMENSION OF C. NMAX MUST BE GREATER OR EQUAL TO N.
     MNN :      MUST BE EQUAL TO M + N + N.
     C(NMAX,NMAX): OBJECTIVE FUNCTION MATRIX WHICH SHOULD BE SYMMETRIC AND
     POSITIVE DEFINITE. IF IWAR(1) = 0, C IS SUPPOSED TO BE THE
     CHOLESKEY-FACTOR OF ANOTHER MATRIX, I.E. C IS UPPER
     TRIANGULAR.
     D(NMAX) :  CONTAINS THE CONSTANT VECTOR OF THE OBJECTIVE FUNCTION.
     A(MMAX,NMAX): CONTAINS THE DATA MATRIX OF THE LINEAR CONSTRAINTS.
     B(MMAX) :  CONTAINS THE CONSTANT DATA OF THE LINEAR CONSTRAINTS.
     XL(N),XU(N): CONTAIN THE LOWER AND UPPER BOUNDS FOR THE VARIABLES.
     X(N) :     ON RETURN, X CONTAINS THE OPTIMAL SOLUTION VECTOR.
     U(MNN) :   ON RETURN, U CONTAINS THE LAGRANGE MULTIPLIERS. THE FIRST
     M POSITIONS ARE RESERVED FOR THE MULTIPLIERS OF THE M
     LINEAR CONSTRAINTS AND THE SUBSEQUENT ONES FOR THE
     MULTIPLIERS OF THE LOWER AND UPPER BOUNDS. ON SUCCESSFUL
     TERMINATION, ALL VALUES OF U WITH RESPECT TO INEQUALITIES
     AND BOUNDS SHOULD BE GREATER OR EQUAL TO ZERO.
     IOUT :     INTEGER INDICATING THE DESIRED OUTPUT UNIT NUMBER, I.E.
     ALL WRITE-STATEMENTS START WITH 'WRITE(IOUT,... '.
     IFAIL :    SHOWS THE TERMINATION REASON.
     IFAIL = 0 :   SUCCESSFUL RETURN.
     IFAIL = 1 :   TOO MANY ITERATIONS (MORE THAN 40*(N+M)).
     IFAIL = 2 :   ACCURACY INSUFFICIENT TO SATISFY CONVERGENCE
     CRITERION.
     IFAIL = 5 :   LENGTH OF A WORKING ARRAY IS TOO SHORT.
     IFAIL > 10 :  THE CONSTRAINTS ARE INCONSISTENT.
     IPRINT :   OUTPUT CONTROL.
     IPRINT = 0 :  NO OUTPUT OF QL0001.
     IPRINT > 0 :  BRIEF OUTPUT IN ERROR CASES.
     WAR(LWAR) : REAL WORKING ARRAY. THE LENGTH LWAR SHOULD BE GRATER THAN
     3*NMAX*NMAX/2 + 10*NMAX + 2*MMAX.
     IWAR(LIWAR): INTEGER WORKING ARRAY. THE LENGTH LIWAR SHOULD BE AT
     LEAST N.
     IF IWAR(1)=0 INITIALLY, THEN THE CHOLESKY DECOMPOSITION
     WHICH IS REQUIRED BY THE DUAL ALGORITHM TO GET THE FIRST
     UNCONSTRAINED MINIMUM OF THE OBJECTIVE FUNCTION, IS
     PERFORMED INTERNALLY. OTHERWISE, I.E. IF IWAR(1)=1, THEN
     IT IS ASSUMED THAT THE USER PROVIDES THE INITIAL FAC-
     TORIZATION BY HIMSELF AND STORES IT IN THE UPPER TRIAN-
     GULAR PART OF THE ARRAY C.

     A NAMED COMMON-BLOCK  /CMACHE/EPS   MUST BE PROVIDED BY THE USER,
     WHERE EPS DEFINES A GUESS FOR THE UNDERLYING MACHINE PRECISION.


     AUTHOR:    K. SCHITTKOWSKI,
     MATHEMATISCHES INSTITUT,
     UNIVERSITAET BAYREUTH,
     8580 BAYREUTH,
     GERMANY, F.R.


     VERSION:   1.4  (MARCH, 1987)
  */
  /* f2c.h  --  Standard Fortran to C header file */

  /**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

       - From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */
#ifdef __STDC__
  int ql0001_(int *m,int *me,int *mmax,int *n,int *nmax,int *mnn,
	      double *c,double *d,double *a,double *b,double *xl,
	      double *xu,double *x,double *u,int *iout,int *ifail,
	      int *iprint,double *war,int *lwar,int *iwar,int *liwar,
	      double *eps1);
#else
  /* Subroutine */
  int ql0001_(m, me, mmax, n, nmax, mnn, c, d, a, b, xl, xu, x,
	      u, iout, ifail, iprint, war, lwar, iwar, liwar, eps1)
    integer *m, *me, *mmax, *n, *nmax, *mnn;
  doublereal *c, *d, *a, *b, *xl, *xu, *x, *u;
  integer *iout, *ifail, *iprint;
  doublereal *war;
  integer *lwar, *iwar, *liwar;
  doublereal *eps1;
#endif
};

#endif //! OFSQP_HH
