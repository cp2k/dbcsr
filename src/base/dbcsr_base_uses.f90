! Basic use statements and preprocessor macros
! should be included in the use statements

  USE dbcsr_base_hooks, ONLY: dbcsr__a, &
     dbcsr__b, &
     dbcsr__w, &
     dbcsr__l, &
     dbcsr_abort, &
     dbcsr_warn, &
     timeset, &
     timestop

! Dangerous: Full path can be arbitrarily long and might overflow Fortran line.
#if !defined(__SHORT_FILE__)
#define __SHORT_FILE__ __FILE__
#endif

#define __LOCATION__ dbcsr__l(__SHORT_FILE__,__LINE__)
#define DBCSR_WARN(msg) CALL dbcsr__w(__SHORT_FILE__,__LINE__,msg)
#define DBCSR_ABORT(msg) CALL dbcsr__b(__SHORT_FILE__,__LINE__,msg)

! DBCSR_ASSERT can be elided if NDEBUG is defined.
#if defined(NDEBUG)
# define DBCSR_ASSERT(cond)
#else
#define DBCSR_ASSERT(cond) IF(.NOT.(cond))CALL dbcsr__a(__SHORT_FILE__,__LINE__)
#endif

! The MARK_USED macro can be used to mark an argument/variable as used.
! It is intended to make it possible to switch on -Werror=unused-dummy-argument,
! but deal elegantly with e.g. library wrapper routines that take arguments only used if the library is linked in.
! This code should be valid for any Fortran variable, is always standard conforming,
! and will be optimized away completely by the compiler
#define MARK_USED(foo) IF(.FALSE.)THEN; DO ; IF(SIZE(SHAPE(foo))==-1) EXIT ;  END DO ; ENDIF

! Calculate version number from 3-components. Can be used for comparison e.g.,
! TO_VERSION(4, 9, 0) <= TO_VERSION(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#define TO_VERSION(MAJOR, MINOR, UPDATE) ((MAJOR) * 10000 + (MINOR) * 100 + (UPDATE))

! LIBXSMM has a FORTRAN-suitable header with macro/version definitions (since v1.8.2).
! Allows macro-toggles (in addition to parameters).
#if defined(__LIBXSMM)
#include <libxsmm_config.h>
#endif
