!--------------------------------------------------------------------------------------------------!
!   CP2K: A general program to perform molecular dynamics simulations                              !
!   Copyright (C) 2000 - 2018  CP2K developers group                                               !
!--------------------------------------------------------------------------------------------------!

! **************************************************************************************************
!> \brief type to store parallelization informations (at the moment assumes 1d
!>      position and uses mpi)
!> \par History
!>      07.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
MODULE dbcsr_para_env
   USE dbcsr_message_passing,           ONLY: mp_comm_free,&
                                              mp_environ
   USE dbcsr_para_types,                ONLY: dbcsr_para_cart_type,&
                                              dbcsr_para_env_type
#include "base/dbcsr_base_uses.f90"

   IMPLICIT NONE
   PRIVATE

   LOGICAL, PRIVATE, PARAMETER :: debug_this_module = .TRUE.
   CHARACTER(len=*), PARAMETER, PRIVATE :: moduleN = 'dbcsr_para_env'

   PUBLIC :: dbcsr_para_env_retain, dbcsr_para_env_release, dbcsr_para_env_create
   PUBLIC :: dbcsr_cart_create, dbcsr_cart_release
!***
CONTAINS

! **************************************************************************************************
!> \brief creates a new para environment
!> \param para_env the new parallel environment
!> \param group the id of the actual mpi_group
!> \param source the id of the special (master) processor (defaults to 0)
!> \param mepos the id of the actual processor
!> \param num_pe the number of processors in the group
!> \param owns_group if the group is owned by this object (defaults to true)
!> \par History
!>      08.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE dbcsr_para_env_create(para_env, group, source, mepos, num_pe, &
                                    owns_group)
      TYPE(dbcsr_para_env_type), POINTER                 :: para_env
      INTEGER, INTENT(in)                                :: group
      INTEGER, INTENT(in), OPTIONAL                      :: source, mepos, num_pe
      LOGICAL, INTENT(in), OPTIONAL                      :: owns_group

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_para_env_create', &
         routineP = moduleN//':'//routineN

      DBCSR_ASSERT(.NOT. ASSOCIATED(para_env))
      ALLOCATE (para_env)
      para_env%group = group
      para_env%source = 0
      para_env%ref_count = 1
      para_env%owns_group = .TRUE.
      IF (PRESENT(source)) para_env%source = source
      IF (PRESENT(owns_group)) para_env%owns_group = owns_group
      IF (.NOT. (PRESENT(mepos) .AND. PRESENT(num_pe))) THEN
         CALL dbcsr_para_env_update(para_env)
      ELSE
         para_env%mepos = mepos
         para_env%num_pe = num_pe
      END IF
      para_env%ionode = para_env%mepos == para_env%source
   END SUBROUTINE dbcsr_para_env_create

! **************************************************************************************************
!> \brief retains the para object (to be called when you want to keep a
!>      shared copy of this object)
!> \param para_env the new group
!> \par History
!>      08.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE dbcsr_para_env_retain(para_env)
      TYPE(dbcsr_para_env_type), POINTER                 :: para_env

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_para_env_retain', &
         routineP = moduleN//':'//routineN

      DBCSR_ASSERT(ASSOCIATED(para_env))
      DBCSR_ASSERT(para_env%ref_count > 0)
      para_env%ref_count = para_env%ref_count+1
   END SUBROUTINE dbcsr_para_env_retain

! **************************************************************************************************
!> \brief releases the para object (to be called when you don't want anymore
!>      the shared copy of this object)
!> \param para_env the new group
!> \par History
!>      08.2002 created [fawzi]
!> \author Fawzi Mohamed
!> \note
!>      to avoid circular dependencies dbcsr_log_handling has a private copy
!>      of this method (see dbcsr_log_handling:my_dbcsr_para_env_release)!
! **************************************************************************************************
   SUBROUTINE dbcsr_para_env_release(para_env)
      TYPE(dbcsr_para_env_type), POINTER                 :: para_env

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_para_env_release', &
         routineP = moduleN//':'//routineN

      IF (ASSOCIATED(para_env)) THEN
         DBCSR_ASSERT(para_env%ref_count > 0)
         para_env%ref_count = para_env%ref_count-1
         IF (para_env%ref_count < 1) THEN
            IF (para_env%owns_group) THEN
               CALL mp_comm_free(para_env%group)
            END IF
            DEALLOCATE (para_env)
         END IF
      END IF
      NULLIFY (para_env)
   END SUBROUTINE dbcsr_para_env_release

! **************************************************************************************************
!> \brief gets again the position and size of the group from the mpi_group
!> \param para_env the new group
!> \par History
!>      08.2002 created [fawzi]
!> \author Fawzi Mohamed
! **************************************************************************************************
   SUBROUTINE dbcsr_para_env_update(para_env)
      TYPE(dbcsr_para_env_type), POINTER                 :: para_env

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_para_env_update', &
         routineP = moduleN//':'//routineN

      DBCSR_ASSERT(ASSOCIATED(para_env))
      DBCSR_ASSERT(para_env%ref_count > 0)
      CALL mp_environ(taskid=para_env%mepos, numtask=para_env%num_pe, &
                      groupid=para_env%group)
      para_env%ionode = para_env%mepos == para_env%source
   END SUBROUTINE dbcsr_para_env_update

! **************************************************************************************************
!> \brief creates a cart (multidimensional parallel environment)
!> \param cart the cart environment to create
!> \param group the mpi communicator
!> \param ndims the number of dimensions of the cart
!> \param owns_group if this object owns the underlying cart (and should
!>        free it)
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE dbcsr_cart_create(cart, group, ndims, owns_group)
      TYPE(dbcsr_para_cart_type), POINTER                :: cart
      INTEGER, INTENT(in)                                :: group, ndims
      LOGICAL, INTENT(in), OPTIONAL                      :: owns_group

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_cart_create', &
         routineP = moduleN//':'//routineN

      DBCSR_ASSERT(.NOT. ASSOCIATED(cart))
      ALLOCATE (cart)
      cart%owns_group = .TRUE.
      IF (PRESENT(owns_group)) cart%owns_group = owns_group
      cart%ndims = ndims
      cart%group = group

      ALLOCATE (cart%source(ndims), cart%periodic(ndims), cart%mepos(ndims), &
                cart%num_pe(ndims))

      cart%source = 0
      cart%mepos = 0
      cart%periodic = .FALSE.
      cart%ref_count = 1
      cart%ntask = 1
      CALL dbcsr_cart_update(cart)
   END SUBROUTINE dbcsr_cart_create

! **************************************************************************************************
!> \brief updates the information about the given cart
!> \param cart the cart to update
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE dbcsr_cart_update(cart)
      TYPE(dbcsr_para_cart_type), POINTER                :: cart

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_cart_update', &
         routineP = moduleN//':'//routineN

      DBCSR_ASSERT(ASSOCIATED(cart))
      DBCSR_ASSERT(cart%ref_count > 0)
      CALL mp_environ(cart%group, cart%ndims, cart%num_pe, task_coor=cart%mepos, &
                      periods=cart%periodic)
      CALL mp_environ(numtask=cart%ntask, taskid=cart%rank, groupid=cart%group)
   END SUBROUTINE dbcsr_cart_update

! **************************************************************************************************
!> \brief releases the given cart
!> \param cart the cart to release
!> \author fawzi
! **************************************************************************************************
   SUBROUTINE dbcsr_cart_release(cart)
      TYPE(dbcsr_para_cart_type), POINTER                :: cart

      CHARACTER(len=*), PARAMETER :: routineN = 'dbcsr_cart_release', &
         routineP = moduleN//':'//routineN

      IF (ASSOCIATED(cart)) THEN
         DBCSR_ASSERT(cart%ref_count > 0)
         cart%ref_count = cart%ref_count-1
         IF (cart%ref_count == 0) THEN
            IF (cart%owns_group) THEN
               CALL mp_comm_free(cart%group)
            END IF
            DEALLOCATE (cart%source, cart%periodic, cart%mepos, cart%num_pe)
            DEALLOCATE (cart)
         END IF
      END IF
      NULLIFY (cart)
   END SUBROUTINE dbcsr_cart_release

END MODULE dbcsr_para_env
