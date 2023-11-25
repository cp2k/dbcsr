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

#if defined(ADD_UNDERSCORE)
#define FNAME(x) x ## _
#define STATIC 
#elif defined(ADD_PREFIX)
#define FNAME(x) my_ ## x
#define STATIC static
#else
#define FNAME(x) x
#define STATIC 
#endif

STATIC float FNAME(sdsdot)( const int* N, const float* alpha, const float* X, const int* incX, const float* Y, const int* incY )
{
  DEBUG_S( "sdsdot" )
  return cblas_sdsdot( *N, *alpha, X, *incX, Y, *incY );
}

STATIC float FNAME(sdot)( const int* N, const float* X, const int* incX, const float* Y, const int* incY )
{
  DEBUG_S( "sdot" )
  return cblas_sdot( *N, X, *incX, Y, *incY );
}

STATIC float FNAME(snrm2)( const int* N, const float* X, const int* incX )
{
  DEBUG_S( "snrm2" )
  return cblas_snrm2( *N, X, *incX );
}

STATIC float FNAME(sasum)( const int* N, const float *X, const int* incX )
{
  DEBUG_S( "sasum" )
  return cblas_sasum( *N, X, *incX );
}

STATIC c_float FNAME(cdotu)( const int* N, const void* X, const int* incX, const void* Y, const int* incY )
{
  DEBUG_S( "cdotu" )
  c_float ans;
  cblas_cdotu_sub( *N, X, *incX, Y, *incY, &ans );
  return ans;
}

STATIC c_float FNAME(cdotc)( const int* N, const void* X, const int* incX, const void* Y, const int* incY )
{
  DEBUG_S( "cdotc" )
  c_float ans;
  cblas_cdotc_sub( *N, X, *incX, Y, *incY, &ans );
  return ans;
}

STATIC float FNAME(scnrm2)( const int* N, const void* X, const int* incX )
{
  DEBUG_S( "scnrm2" )
  return cblas_scnrm2( *N, X, *incX );
}

STATIC float FNAME(scasum)( const int* N, const void *X, const int* incX )
{
  DEBUG_S( "scasum" )
  return cblas_scasum( *N, X, *incX );
}

STATIC c_double FNAME(zdotu)( const int* N, const void* X, const int* incX, const void* Y, const int* incY )
{
  DEBUG_S( "zdotu" )
  c_double ans;
  cblas_zdotu_sub( *N, X, *incX, Y, *incY, &ans );
  return ans;
}

STATIC c_double FNAME(zdotc)( const int* N, const void* X, const int* incX, const void* Y, const int* incY )
{
  DEBUG_S( "zdotc" )
  c_double ans;
  cblas_zdotc_sub( *N, X, *incX, Y, *incY, &ans );
  return ans;
}

#ifdef VECLIBFORT_SGEMV
STATIC void FNAME(sgemv)( const char* trans, const int* m, const int* n, 
  const float* alpha, const float* A, const int* ldA,
  const float* X, const int* incX,
  const float* beta, float* Y, const int* incY )
{
  DEBUG_S( "sgemv" )
  enum CBLAS_TRANSPOSE T;
  switch ( trans[0] ) {
    case 'T': case 't':
      if ( ((intptr_t)X|(intptr_t)A|(intptr_t)Y)%32 == 0 ) { T = CblasTrans; break; }
      /* Implement as alpha * X^T * A + beta * Y^T */
      cblas_sgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, 1, *n, *m, *alpha, X, *incX, A, *ldA, *beta, Y, *incY );
      return;
    default:
      T = CblasNoTrans;
  }
  cblas_sgemv( CblasColMajor, T, *m, *n, *alpha, A, *ldA, X, *incX, *beta, Y, *incY );
}
#endif

#undef FNAME
#undef STATIC

