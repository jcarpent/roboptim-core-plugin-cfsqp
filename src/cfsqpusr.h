/*************************************************************/
/*  CFSQP - Header file to be included in user's main        */
/*          program.                                         */
/*************************************************************/

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#ifndef __STDC__
#ifdef apollo
extern char *calloc();
#else
#include <malloc.h>
#endif
#endif

#define TRUE 1
#define FALSE 0

/* Declare and initialize user-accessible flag indicating    */
/* whether x sent to user functions has been changed within  */
/* CFSQP.				 		     */

extern int x_is_new;

extern void set_x_is_new(int n);
extern int get_x_is_new();

/* Declare and initialize user-accessible stopping criterion */
extern double objeps;
extern double objrep;
extern double gLgeps;
extern int nstop;

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
/*     Prototype for QL0001_ -                                */
/**************************************************************/

#ifdef __STDC__
int
ql0001_(int *,int *,int *,int *,int *,int *,double *,double *,
        double *,double *,double *,double *,double *,double *,
        int *,int *,int *,double *,int *,int *,int *,double *);
#else
int  	ql0001_();
#endif

/**************************************************************/
/*     Prototype for CFSQP -   	                              */
/**************************************************************/

#ifdef __STDC__
void    cfsqp(int,int,int,int,int,int,int,int,int,int *,int,int,
              int,int *,double,double,double,double,double *,
              double *,double *,double *,double *,double *,
              void (*)(int,int,double *,double *,void *),
              void (*)(int,int,double *,double *,void *),
              void (*)(int,int,double *,double *,
                   void (*)(int,int,double *,double *,void *),void *),
              void (*)(int,int,double *,double *,
                   void (*)(int,int,double *,double *,void *),void *),
              void *);
#else
void    cfsqp();
#endif

#ifdef __cplusplus
}
#endif
