/*

  vecLibFort
  https://github.com/mcg1969/vecLibFort
  Run-time F2C/GFORTRAN translation for Apple's vecLib BLAS/LAPACK
  Copyright (c) 2014 Michael C. Grant

  See README.md for full background and usage details.

  Use, modification and distribution is subject to the Boost Software 
  License, Version 1.0. See the accompanying file LICENSE or

      http://www.booost.org/LICENSE_1_0.txt

*/

#include <stdio.h>
#include "cloak.h"
/* Don't load the CLAPACK header, because we are using a different calling
   convention for the replaced functions than the ones listed there. */
#define __CLAPACK_H
#include "vecLib-760.100.h"
#include <Accelerate/Accelerate.h>
#include <AvailabilityMacros.h>

/* Add a SGEMV fix for Mavericks. See
  http://www.openradar.me/radar?id=5864367807528960 */

#if !defined(VECLIBFORT_SGEMV) && \
    defined(MAC_OS_X_VERSION_10_9) && \
    MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9 && \
    !(defined(MAC_OS_X_VERSION_10_10) && \
      MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_10)
#define VECLIBFORT_SGEMV
#endif

#define VOIDS_(s,i,id) COMMA_IF(i) void*
#define VOIDS(n) IF(n)(EXPR_S(0)(REPEAT_S(0,DEC(n),VOIDS_,~)),void)
#define VOIDA_(s,i,id) COMMA_IF(i) void *a ## i
#define VOIDA(n) IF(n)(EXPR_S(0)(REPEAT_S(0,DEC(n),VOIDA_,~)),void)
#define PARAM_(s,i,id) COMMA_IF(i)a ## i
#define PARAM(n) IF(n)(EXPR_S(0)(REPEAT_S(0,DEC(n),PARAM_,~)),)

#ifdef VECLIBFORT_VERBOSE
#define DEBUG(...) fprintf(stderr,__VA_ARGS__);
static const char* dynamic_msg = "Entering dynamic %s replacement\n";
static const char* static_msg = "Entering static %s replacement\n";
#define DEBUG_S(x) DEBUG( static_msg, x )
#define DEBUG_D(x) DEBUG( dynamic_msg, x )

#else
#define DEBUG(...)
#define DEBUG_S(x)
#define DEBUG_D(x)
#endif

#include <complex.h>
typedef float complex c_float;
typedef double complex c_double;

#ifdef VECLIBFORT_INTERPOSE

/*
 * INTERPOSING MODE
 *
 * In this mode, dyld is instructed to preload this library even before the
 * executable itself. It reads the __DATA.__interpose section of the library
 * for the interpose information, which it uses to swap out the offending
 * BLAS/LAPACK functions with our replacements. Because vecLib provides two
 * aliases for each function---one with a trailing underscore, and one
 * without---we need two interpose records for each replacement.
 *
 * For instance, for "sdot", we define a static function
 *    static float my_sdot( const int* N, const float* X, const int* incX )
 * add interpose data to signify two substitutions:
 *    sdot_ -> my_sdot
 *    sdot  -> my_sdot
 */

typedef struct interpose_t_ {
  const void *replacement;
  const void *original;
} interpose_t;

#define INTERPOSE(name) \
__attribute__((used)) interpose_t interpose_ ## name [] \
__attribute__((section ("__DATA,__interpose"))) = \
{ { (const void*)&my_ ## name, (const void*)&name }, \
  { (const void*)&my_ ## name, (const void*)&name ## _ } };

#define D2F_CALL(name,n) \
extern double name( VOIDS(n) ); \
extern double name ## _( VOIDS(n) ); \
static float my_ ## name ( VOIDA(n) ) \
{ return (float)name ## _( PARAM(n) ); } \
INTERPOSE(name)

#define CPLX_CALL(type,name,n) \
extern void name( VOIDS(INC(n)) ); \
extern void name ## _( VOIDS(INC(n)) ); \
static c_ ## type my_ ## name ( VOIDA(n) ) \
{ \
  c_ ## type cplx; \
  name ## _( &cplx, PARAM(n) ); \
  return cplx; \
} \
INTERPOSE(name)

/*
 * DYNAMIC BLAS SUBSTITUTION
 *
 * For the interpose library we need to use the same techniques for the BLAS
 * as we do for the LAPACK routines. However, because we have CBLAS versions
 * available to use, we can use the wrappers already created in "static.h"
 * by prepending them with the "my_" prefixes.
 */

#define BLS_CALL(type,name,n) \
extern type name( VOIDS(n) ); \
extern type name ## _( VOIDS(n) ); \
INTERPOSE(name)
  
#define ADD_PREFIX
#include "static.h"
#undef ADD_PREFIX

BLS_CALL(float,sdsdot,6)
BLS_CALL(float,sdot,5)
BLS_CALL(float,snrm2,3)
BLS_CALL(float,sasum,3)
BLS_CALL(c_float,cdotu,5)
BLS_CALL(c_float,cdotc,5)
BLS_CALL(float,scnrm2,3)
BLS_CALL(float,scasum,3)
BLS_CALL(c_double,zdotu,5)
BLS_CALL(c_double,zdotc,5)
#if defined(VECLIBFORT_SGEMV)
BLS_CALL(void,sgemv,11)
#endif

#else

/*
 * STATIC BLAS SUBSTITUTION
 * 
 * For BLAS functions, we have access to CBLAS versions of each function.
 * So the hoops we need to jump through to resolve the name clashes in the
 * dynamic substitution mode can be avoided. Instead, we simply create the
 * replacement functions to call the CBLAS counterparts instead.
 *
 * To void duplicating code, we include the functions in "static.h" twice:
 * once for the functions with trailing underscores (e.g., "sdot_"), and once 
 * without (e.g., "sdot"). In theory, we could create just one replacement
 * with two aliases, but clang has thus far been uncooperative. Any assistance 
 * on this matter would be appreciated.
 */

#include "static.h"
#define ADD_UNDERSCORE
#include "static.h"

/*
 * DYNAMIC LAPACK SUBSTITUTION
 * 
 * In this mode, we give our functions identical names, and rely on link
 * order to ensure that these take precedence over those declared in vecLib.
 * Thus whenever the main code attempts to call one of the covered functions,
 * it will be directed to one of our wrappers instead.
 *
 * Because vecLib provides two aliases for each function---one with a
 * trailing underscore, and one without---we actually need two separate
 * replacement functions (at least until we can figure out how to do aliases
 * cleanly in clang.) Each pair of replacements controls a single static
 * pointer to the replacement function. On the first invocation of either,
 * this pointer is retrieved using a dlsym() command.
 *
 * For instance, for "sdot", we define two functions
 *    float sdot_( const int* N, const float* X, const int* incX )
 *    float sdot ( const int* N, const float* X, const int* incX )
 * On the first invocation of either, the "sdot_" symbol from vecLib is
 * retrieved using the dlsym() command and stored in
 *    static void* fp_dot;
 * In theory, we could create just one replacement with two aliases, but 
 * clang has thus far been uncooperative. Any assistance on this matter would
 * be appreciated. 
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

#define VECLIB_FILE "/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/vecLib"

static void * veclib = 0;

static void unloadlib(void)
{
  DEBUG( "Unloading vecLib\n" );
  dlclose (veclib);
}

static void loadlib(void)
{
  static const char* veclib_loc = VECLIB_FILE;
  DEBUG( "Loading library: %s\n", veclib_loc )
  veclib = dlopen (veclib_loc, RTLD_LOCAL | RTLD_FIRST);
  if ( veclib == 0 ) {
    fprintf( stderr, "Failed to open vecLib library; aborting.\n   Location: %s\n", veclib );
    abort ();
  }
  atexit(unloadlib); 
}

static void* loadsym( const char* nm )
{
  if ( veclib == 0 ) loadlib();
  DEBUG( "Loading function: %s\n", nm )
  void *ans = dlsym( veclib, nm );
  if ( ans != 0 ) return ans;
  fprintf( stderr, "vecLib symbol '%s' could not be resolved; aborting.\n", nm );
  abort();
}

#define D2F_CALL_(fname,name,n) \
float fname( VOIDA(n) ) \
{ \
  DEBUG_D( #name "_" ) \
  if ( !fp_ ## name ) fp_ ## name = loadsym( #name "_" ); \
  return ((ft_ ## name)fp_ ## name)( PARAM(n) ); \
}

#define D2F_CALL(name,n) \
typedef double (*ft_ ## name)( VOIDS(n) ); \
static void *fp_ ## name = 0; \
D2F_CALL_(name,name,n) \
D2F_CALL_(name ## _,name,n)

#define CPLX_CALL_(type,fname,name,n) \
c_ ## type fname( VOIDA(n) ) \
{ \
  c_ ## type cplx; \
  DEBUG_D( #name "_" ) \
  if ( !fp_ ## name ) fp_ ## name = loadsym( #name "_" ); \
  ((ft_ ## name)fp_ ## name)( &cplx, PARAM(n) ); \
  return cplx; \
}

#define CPLX_CALL(type,name,n) \
typedef void (*ft_ ## name)( VOIDS(INC(n)) ); \
static void *fp_ ## name = 0; \
CPLX_CALL_(type,name,name,n) \
CPLX_CALL_(type,name ## _,name,n)

#endif

D2F_CALL(clangb,7)
D2F_CALL(clange,6)
D2F_CALL(clangt,5)
D2F_CALL(clanhb,7)
D2F_CALL(clanhe,6)
D2F_CALL(clanhp,5)
D2F_CALL(clanhs,5)
D2F_CALL(clanht,4)
D2F_CALL(clansb,7)
D2F_CALL(clansp,5)
D2F_CALL(clansy,6)
D2F_CALL(clantb,8)
D2F_CALL(clantp,6)
D2F_CALL(clantr,8)

D2F_CALL(scsum1,3)
#if defined(MAC_OS_X_VERSION_10_6) && \
    MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_6
D2F_CALL(slaneg,6)
#endif
D2F_CALL(slangb,7)
D2F_CALL(slange,6)
D2F_CALL(slangt,5)
D2F_CALL(slanhs,5)
D2F_CALL(slansb,7)
D2F_CALL(slansp,5)
D2F_CALL(slanst,4)
D2F_CALL(slansy,6)
D2F_CALL(slantb,8)
D2F_CALL(slantp,6)
D2F_CALL(slantr,8)
D2F_CALL(slapy2,2)
D2F_CALL(slapy3,3)
D2F_CALL(slamch,1)
D2F_CALL(slamc3,2)

#if defined(MAC_OS_X_VERSION_10_7) && \
    MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_7
D2F_CALL(clanhf,6)
D2F_CALL(slansf,6)
#endif

CPLX_CALL(float,cladiv,2)
CPLX_CALL(double,zladiv,2)


