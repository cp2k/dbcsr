# 1 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F" 1
!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# 1 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.fypp" 1
# 9 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.fypp"

# 56 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.fypp"
# 11 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F" 2

MODULE dbcsr_btree
   !! B-tree
   IMPLICIT NONE
   PRIVATE

   !API
   PUBLIC :: btree_new, btree_add, btree_find, &
             btree_delete, btree_get_entries
# 21 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      PUBLIC :: btree_i8_sp2d
# 22 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 21 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      PUBLIC :: btree_i8_dp2d
# 22 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 21 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      PUBLIC :: btree_i8_cp2d
# 22 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 21 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      PUBLIC :: btree_i8_zp2d
# 22 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 23 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

   INTEGER, PARAMETER :: keyt = SELECTED_INT_KIND(10)
   INTEGER, PARAMETER :: valt = SELECTED_INT_KIND(5)
   INTEGER, PARAMETER :: sp = KIND(0.0)
   INTEGER, PARAMETER :: dp = KIND(0.0d0)

# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
TYPE btree_data_sp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
REAL(KIND=sp), DIMENSION(:,:), POINTER :: p => NULL()
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
LOGICAL :: tr = .FALSE.
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
END TYPE btree_data_sp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
PUBLIC :: btree_data_sp2d
# 31 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
TYPE btree_data_dp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
REAL(KIND=dp), DIMENSION(:,:), POINTER :: p => NULL()
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
LOGICAL :: tr = .FALSE.
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
END TYPE btree_data_dp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
PUBLIC :: btree_data_dp2d
# 31 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
TYPE btree_data_cp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
COMPLEX(KIND=sp), DIMENSION(:,:), POINTER :: p => NULL()
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
LOGICAL :: tr = .FALSE.
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
END TYPE btree_data_cp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
PUBLIC :: btree_data_cp2d
# 31 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
TYPE btree_data_zp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
COMPLEX(KIND=dp), DIMENSION(:,:), POINTER :: p => NULL()
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
LOGICAL :: tr = .FALSE.
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
END TYPE btree_data_zp2d
# 30 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
PUBLIC :: btree_data_zp2d
# 31 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 32 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

   INTERFACE btree_new
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_new_i8_sp2d
# 36 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_new_i8_dp2d
# 36 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_new_i8_cp2d
# 36 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 35 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_new_i8_zp2d
# 36 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 37 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
   END INTERFACE

   INTERFACE btree_add
# 41 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_add_i8_sp2d
# 42 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 41 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_add_i8_dp2d
# 42 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 41 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_add_i8_cp2d
# 42 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 41 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_add_i8_zp2d
# 42 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 43 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
   END INTERFACE

   INTERFACE btree_find
# 47 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_find_i8_sp2d
# 48 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 47 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_find_i8_dp2d
# 48 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 47 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_find_i8_cp2d
# 48 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 47 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_find_i8_zp2d
# 48 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 49 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
   END INTERFACE

   INTERFACE btree_delete
# 53 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_delete_i8_sp2d
# 54 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 53 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_delete_i8_dp2d
# 54 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 53 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_delete_i8_cp2d
# 54 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 53 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_delete_i8_zp2d
# 54 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 55 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
   END INTERFACE

   INTERFACE btree_get_entries
# 59 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_get_entries_i8_sp2d
# 60 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 59 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_get_entries_i8_dp2d
# 60 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 59 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_get_entries_i8_cp2d
# 60 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 59 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         MODULE PROCEDURE btree_get_entries_i8_zp2d
# 60 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 61 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
   END INTERFACE

# 64 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      TYPE btree_node_i8_sp2d
# 65 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: id = -1
         INTEGER :: filled = -1
         INTEGER(KIND=keyt), DIMENSION(:), POINTER :: keys => NULL()
# 68 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), DIMENSION(:), POINTER :: values => NULL()
# 69 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_p_i8_sp2d), DIMENSION(:), POINTER :: subtrees => NULL()
# 70 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: parent => NULL()
# 71 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_i8_sp2d
# 72 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_p_i8_sp2d
# 74 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node => NULL()
# 75 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_p_i8_sp2d
# 76 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_structure_i8_sp2d
# 78 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: min_fill = -1, max_fill = -1
         INTEGER :: n = -1
         INTEGER :: lastid = -1
         INTEGER :: refcount = -1
         TYPE(btree_node_i8_sp2d), POINTER :: root => NULL()
# 83 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_structure_i8_sp2d
# 84 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_i8_sp2d
# 86 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_structure_i8_sp2d) :: b = btree_node_structure_i8_sp2d ()
# 87 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_i8_sp2d
# 88 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 64 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      TYPE btree_node_i8_dp2d
# 65 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: id = -1
         INTEGER :: filled = -1
         INTEGER(KIND=keyt), DIMENSION(:), POINTER :: keys => NULL()
# 68 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), DIMENSION(:), POINTER :: values => NULL()
# 69 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_p_i8_dp2d), DIMENSION(:), POINTER :: subtrees => NULL()
# 70 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: parent => NULL()
# 71 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_i8_dp2d
# 72 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_p_i8_dp2d
# 74 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node => NULL()
# 75 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_p_i8_dp2d
# 76 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_structure_i8_dp2d
# 78 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: min_fill = -1, max_fill = -1
         INTEGER :: n = -1
         INTEGER :: lastid = -1
         INTEGER :: refcount = -1
         TYPE(btree_node_i8_dp2d), POINTER :: root => NULL()
# 83 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_structure_i8_dp2d
# 84 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_i8_dp2d
# 86 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_structure_i8_dp2d) :: b = btree_node_structure_i8_dp2d ()
# 87 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_i8_dp2d
# 88 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 64 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      TYPE btree_node_i8_cp2d
# 65 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: id = -1
         INTEGER :: filled = -1
         INTEGER(KIND=keyt), DIMENSION(:), POINTER :: keys => NULL()
# 68 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), DIMENSION(:), POINTER :: values => NULL()
# 69 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_p_i8_cp2d), DIMENSION(:), POINTER :: subtrees => NULL()
# 70 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: parent => NULL()
# 71 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_i8_cp2d
# 72 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_p_i8_cp2d
# 74 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node => NULL()
# 75 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_p_i8_cp2d
# 76 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_structure_i8_cp2d
# 78 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: min_fill = -1, max_fill = -1
         INTEGER :: n = -1
         INTEGER :: lastid = -1
         INTEGER :: refcount = -1
         TYPE(btree_node_i8_cp2d), POINTER :: root => NULL()
# 83 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_structure_i8_cp2d
# 84 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_i8_cp2d
# 86 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_structure_i8_cp2d) :: b = btree_node_structure_i8_cp2d ()
# 87 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_i8_cp2d
# 88 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 64 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      TYPE btree_node_i8_zp2d
# 65 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: id = -1
         INTEGER :: filled = -1
         INTEGER(KIND=keyt), DIMENSION(:), POINTER :: keys => NULL()
# 68 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), DIMENSION(:), POINTER :: values => NULL()
# 69 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_p_i8_zp2d), DIMENSION(:), POINTER :: subtrees => NULL()
# 70 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: parent => NULL()
# 71 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_i8_zp2d
# 72 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_p_i8_zp2d
# 74 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node => NULL()
# 75 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_p_i8_zp2d
# 76 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_node_structure_i8_zp2d
# 78 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: min_fill = -1, max_fill = -1
         INTEGER :: n = -1
         INTEGER :: lastid = -1
         INTEGER :: refcount = -1
         TYPE(btree_node_i8_zp2d), POINTER :: root => NULL()
# 83 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_node_structure_i8_zp2d
# 84 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      TYPE btree_i8_zp2d
# 86 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_structure_i8_zp2d) :: b = btree_node_structure_i8_zp2d ()
# 87 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END TYPE btree_i8_zp2d
# 88 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 89 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

CONTAINS

# 93 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      SUBROUTINE btree_new_i8_sp2d (tree, order)
# 94 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(OUT) :: tree
# 95 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: order
         INTEGER :: maxs, mins
         !
         IF (PRESENT(order)) THEN
            maxs = order - 1
         ELSE
            maxs = 15
         END IF
         mins = ISHFT(maxs, -1)
         IF (mins*2 .GT. maxs) maxs = 2*maxs
         IF (mins .LT. 1) mins = 1
         IF (maxs .LT. 3) maxs = 3
         tree%b%min_fill = mins
         tree%b%max_fill = maxs
         tree%b%refcount = 1
         tree%b%n = 0
         NULLIFY (tree%b%root)
         tree%b%lastid = 0
      END SUBROUTINE btree_new_i8_sp2d
# 114 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      FUNCTION btree_get_entries_i8_sp2d (tree) RESULT(num_entries)
# 116 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(INOUT) :: tree
# 117 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: num_entries
         num_entries = tree%b%n
      END FUNCTION btree_get_entries_i8_sp2d
# 120 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is a non-leaf node
      SUBROUTINE btree_adopt_subtrees_i8_sp2d (node)
# 123 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 124 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ! Assume that node is not a leaf!
         DO i = 1, node%filled + 1
            !IF (ASSOCIATED (node%subtrees(i)%node)) THEN
            !IF (.NOT. ASSOCIATED (node%subtrees(i)%node%parent,&
            ! node)) THEN
            node%subtrees(i)%node%parent => node
            !ENDIF
            !ENDIF
         END DO
      END SUBROUTINE btree_adopt_subtrees_i8_sp2d
# 136 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_delete_i8_sp2d (tree, keys, values)
# 138 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(INOUT) :: tree
# 139 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(OUT), OPTIONAL :: keys
# 140 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), DIMENSION(:), INTENT(OUT), OPTIONAL :: values
# 141 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: pos
         !
         IF (ASSOCIATED(tree%b%root)) THEN
            pos = 0
            IF (PRESENT(keys) .AND. PRESENT(values)) THEN
               pos = 1
               CALL btree_delete_node_i8_sp2d (tree%b%root, pos, keys, values)
# 148 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_delete_node_i8_sp2d (tree%b%root)
# 150 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
         END IF
         NULLIFY (tree%b%root)
      END SUBROUTINE btree_delete_i8_sp2d
# 154 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_delete_node_i8_sp2d (node, pos, keys, values)
# 156 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 157 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(INOUT), OPTIONAL :: pos
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(INOUT), OPTIONAL :: keys
# 159 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), DIMENSION(:), INTENT(INOUT), OPTIONAL :: values
# 160 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         INTEGER :: i
         !
         IF (node%filled .GT. 0 .AND. ASSOCIATED(node%subtrees(1)%node)) THEN
            DO i = 1, node%filled + 1
               IF (PRESENT(pos)) THEN
                  CALL btree_delete_node_i8_sp2d (node%subtrees(i)%node, pos, keys, values)
# 167 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_delete_node_i8_sp2d (node%subtrees(i)%node)
# 169 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               END IF
               IF (PRESENT(pos) .AND. i .LE. node%filled) THEN
                  keys(pos) = node%keys(i)
                  values(pos) = node%values(i)
                  pos = pos + 1
               END IF
            END DO
         ELSEIF (PRESENT(pos) .AND. node%filled .GT. 0) THEN
            keys(pos:pos + node%filled - 1) = node%keys(1:node%filled)
            values(pos:pos + node%filled - 1) = node%values(1:node%filled)
            pos = pos + node%filled
         END IF
         CALL btree_free_node_i8_sp2d (node)
# 182 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END SUBROUTINE btree_delete_node_i8_sp2d
# 183 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! Find the key
      ! IF node still has space, insert & update the node
      ! else
      ! 1. select median
      ! 2. split keys into two nodes (one is new)
      ! 3. insert separation key put into parent, and repeat upwards

      SUBROUTINE btree_add_i8_sp2d (tree, key, value, exists, existing_value, replace)
# 192 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(INOUT) :: tree
# 193 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 194 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(IN) :: value
# 195 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         TYPE(btree_data_sp2d), INTENT(OUT), OPTIONAL :: existing_value
# 197 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(IN), OPTIONAL :: replace
         !
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 200 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, position
         !
         IF (PRESENT(exists)) THEN
            CALL btree_find_full_i8_sp2d (tree, key, node, position, ge_pos, short=.TRUE.)
# 204 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (position .GT. 0) THEN
               exists = .TRUE.
               existing_value = node%values(position)
               IF (PRESENT(replace)) THEN
                  IF (replace) THEN
                     node%values(position) = value
                  END IF
               END IF
               RETURN
            ELSE
               exists = .FALSE.
            END IF
         ELSE
            CALL btree_find_leaf_i8_sp2d (tree, key, node, ge_pos)
# 218 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         CALL btree_add_into_i8_sp2d (tree, node, key, value, before=ge_pos)
# 220 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) existing_value = value
         tree%b%n = tree%b%n + 1
      END SUBROUTINE btree_add_i8_sp2d
# 223 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_add_into_i8_sp2d (tree, node, key, value, before, subtree)
# 225 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(INOUT) :: tree
# 226 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 227 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 228 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(IN) :: value
# 229 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: before
         TYPE(btree_node_i8_sp2d), POINTER, OPTIONAL :: subtree
# 231 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         TYPE(btree_node_i8_sp2d), POINTER :: new_node
# 233 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt) :: upgrade_key
# 234 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, split_pos
         TYPE(btree_data_sp2d) :: upgrade_value
# 236 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL :: leaf
         !
         ! Root is special
         IF (.NOT. ASSOCIATED(node)) THEN
            CALL btree_new_root_i8_sp2d (tree, key, value)
# 241 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (PRESENT(subtree)) THEN
               tree%b%root%subtrees(2)%node => subtree
               subtree%parent => tree%b%root
            END IF
            RETURN
         END IF
         ! Where the insertion takes place.
         IF (PRESENT(before)) THEN
            ge_pos = before
         ELSE
            CALL btree_node_find_gt_pos_i8_sp2d (node%keys, key, ge_pos, node%filled)
# 252 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         ! Addition is easy if the node has enough space.
         leaf = .NOT. ASSOCIATED(node%subtrees(1)%node)
         IF (node%filled .LT. tree%b%max_fill) THEN
            IF (PRESENT(subtree)) THEN
               CALL btree_simple_insertion_i8_sp2d (node, key, value, ge_pos, subtree)
# 258 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_simple_insertion_i8_sp2d (node, key, value, ge_pos)
# 260 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            RETURN
         ELSE
            split_pos = ISHFT(tree%b%max_fill + 1, -1)
            ! I assert that split_pos <= SIZE(node%keys)
            CALL btree_new_node_i8_sp2d (tree, new_node)
# 266 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! The key to be added falls in the left node
            node%filled = split_pos - 1
            IF (ge_pos .LE. split_pos) THEN
               IF (ge_pos .EQ. split_pos) THEN
                  upgrade_key = key
                  upgrade_value = value
               ELSE
                  upgrade_key = node%keys(split_pos - 1)
                  upgrade_value = node%values(split_pos - 1)
               END IF
               IF (PRESENT(subtree)) THEN
                  CALL btree_left_insertion_i8_sp2d (tree, node, new_node, key, value, &
# 278 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_sp2d (new_node)
# 280 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_left_insertion_i8_sp2d (tree, node, new_node, key, value, &
# 282 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos)
               END IF
               !
            ELSE
               upgrade_key = node%keys(split_pos)
               upgrade_value = node%values(split_pos)
               IF (PRESENT(subtree)) THEN
                  CALL btree_right_insertion_i8_sp2d (tree, node, new_node, key, value, &
# 290 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_sp2d (new_node)
# 292 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_right_insertion_i8_sp2d (tree, node, new_node, key, value, &
# 294 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos)
               END IF
               !
            END IF
            !
            new_node%parent => node%parent
            !
            IF (.NOT. leaf) THEN
               CALL btree_adopt_subtrees_i8_sp2d (new_node)
# 303 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            !
            CALL btree_add_into_i8_sp2d (tree, node%parent, upgrade_key, upgrade_value, &
# 306 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                              subtree=new_node)
            !
         END IF
      END SUBROUTINE btree_add_into_i8_sp2d
# 310 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_simple_insertion_i8_sp2d (node, key, value, before, subtree)
# 312 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), INTENT(INOUT) :: node
# 313 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 314 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(IN) :: value
# 315 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before
         TYPE(btree_node_i8_sp2d), POINTER, OPTIONAL :: subtree
# 317 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         ! Shift keys
         node%keys(before + 1:node%filled + 1) = node%keys(before:node%filled)
         node%keys(before) = key
         ! Shift values
         node%values(before + 1:node%filled + 1) = node%values(before:node%filled)
         node%values(before) = value
         ! Shift subtree pointers, but only if node is not a leaf ; assume
         ! leaf <=> present(subtree)
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 2) = &
               node%subtrees(before + 1:node%filled + 1)
            node%subtrees(before + 1)%node => subtree
         END IF
         node%filled = node%filled + 1
      END SUBROUTINE btree_simple_insertion_i8_sp2d
# 333 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_left_insertion_i8_sp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 335 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(IN) :: tree
# 336 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), INTENT(INOUT) :: node, new_node
# 337 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 338 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(IN) :: value
# 339 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_sp2d), POINTER, OPTIONAL :: subtree
# 341 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill) - (split_pos - 1)
         new_node%keys(1:new_node%filled) = &
            node%keys(split_pos:tree%b%max_fill)
         new_node%values(1:new_node%filled) = &
            node%values(split_pos:tree%b%max_fill)
         !IF (ASSOCIATED (node%subtrees(1)%node)) THEN
         IF (PRESENT(subtree)) THEN
            IF (before .EQ. split_pos) THEN
               new_node%subtrees(2:new_node%filled + 1) = &
                  node%subtrees(split_pos + 1:tree%b%max_fill + 1)
               new_node%subtrees(1)%node => subtree
            ELSE
               new_node%subtrees(1:new_node%filled + 1) = &
                  node%subtrees(split_pos:tree%b%max_fill + 1)
            END IF
         END IF
         ! Fill node%{keys,values}(1:node%filled), where node%filled
         ! is split_pos-1, but do insert the new value at ge_pos. The
         ! key/value at split_pos is to be inserted into the
         ! parent.
         ! The new tree is added to the right of the new insertion.
         node%keys(before + 1:node%filled) = node%keys(before:node%filled - 1)
         node%keys(before) = key
         node%values(before + 1:node%filled) = node%values(before:node%filled - 1)
         node%values(before) = value
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 1) = &
               node%subtrees(before + 1:node%filled)
            node%subtrees(before + 1)%node => subtree
         ELSE
            NULLIFY (node%subtrees(before + 1)%node)
         END IF
      END SUBROUTINE btree_left_insertion_i8_sp2d
# 375 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_right_insertion_i8_sp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 377 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(IN) :: tree
# 378 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), INTENT(INOUT) :: node, new_node
# 379 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 380 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(IN) :: value
# 381 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_sp2d), POINTER, OPTIONAL :: subtree
# 383 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill + 1) - split_pos
         new_node%keys(1:before - split_pos - 1) = &
            node%keys(split_pos + 1:before - 1)
         new_node%keys(before - split_pos) = key
         new_node%keys(before - split_pos + 1:new_node%filled) = &
            node%keys(before:tree%b%max_fill)
         new_node%values(1:before - split_pos - 1) = &
            node%values(split_pos + 1:before - 1)
         new_node%values(before - split_pos) = value
         new_node%values(before - split_pos + 1:new_node%filled) = &
            node%values(before:tree%b%max_fill)
         IF (PRESENT(subtree)) THEN
            new_node%subtrees(1:before - split_pos) = &
               node%subtrees(split_pos + 1:before)
            new_node%subtrees(before - split_pos + 1)%node => subtree
            new_node%subtrees(before - split_pos + 2:new_node%filled + 1) = &
               node%subtrees(before + 1:tree%b%max_fill + 1)
         END IF
      END SUBROUTINE btree_right_insertion_i8_sp2d
# 403 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_root_i8_sp2d (tree, key, value)
# 405 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(INOUT) :: tree
# 406 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 407 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(IN) :: value
# 408 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: old_root, new_root
# 409 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         CALL btree_new_node_i8_sp2d (tree, new_root)
# 411 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         new_root%filled = 1
         new_root%keys(1) = key
         new_root%values(1) = value
         IF (ASSOCIATED(tree%b%root)) THEN
            old_root => tree%b%root
            old_root%parent => new_root
            new_root%subtrees(1)%node => old_root
            old_root%parent => new_root
         END IF
         tree%b%root => new_root
      END SUBROUTINE btree_new_root_i8_sp2d
# 422 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_node_i8_sp2d (tree, node)
# 424 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(INOUT) :: tree
# 425 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 426 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ALLOCATE (node)
         ALLOCATE (node%keys(tree%b%max_fill))
         ALLOCATE (node%values(tree%b%max_fill))
         ALLOCATE (node%subtrees(tree%b%max_fill + 1))
         DO i = 1, tree%b%max_fill + 1
            NULLIFY (node%subtrees(i)%node)
         END DO
         node%filled = 0
         NULLIFY (node%parent)
         tree%b%lastid = tree%b%lastid + 1
         node%id = tree%b%lastid
      END SUBROUTINE btree_new_node_i8_sp2d
# 440 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_free_node_i8_sp2d (node)
# 442 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 443 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         DEALLOCATE (node%keys)
         DEALLOCATE (node%values)
         DEALLOCATE (node%subtrees)
         DEALLOCATE (node)
      END SUBROUTINE btree_free_node_i8_sp2d
# 449 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_find_i8_sp2d (tree, key, value, exists)
# 451 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(IN) :: tree
# 452 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 453 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_sp2d), INTENT(OUT) :: value
# 454 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         !
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 457 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: position
         !
         CALL btree_find_full_i8_sp2d (tree, key, node, position, short=.TRUE.)
# 460 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) THEN
            exists = position .GT. 0
         END IF
         IF (position .GT. 0) THEN
            value = node%values(position)
         END IF
      END SUBROUTINE btree_find_i8_sp2d
# 467 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_ge_pos_i8_sp2d (keys, key, position, filled)
# 469 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 470 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 471 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GE. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LT. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GE. key .AND. keys(position - 1) .LT. key) THEN
               RETURN
            END IF
            IF (keys(position) .GE. key) right = MIN(position, right - 1)
            IF (keys(position) .LT. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_ge_pos_i8_sp2d
# 495 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gt_pos_i8_sp2d (keys, key, position, filled)
# 497 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 498 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 499 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GT. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gt_pos_i8_sp2d
# 523 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gte_pos_i8_sp2d (keys, key, position, filled, first)
# 525 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 526 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 527 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER, INTENT(IN), OPTIONAL :: first
         INTEGER :: left, right, one
         !
         one = 1
         IF (PRESENT(FIRST)) one = first
         IF (one .LE. filled) THEN
            IF (keys(one) .GT. key) THEN
               position = one
               RETURN
            END IF
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = one + 1
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gte_pos_i8_sp2d
# 556 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_full_i8_sp2d (tree, key, node, position, ge_position, short)
# 561 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(IN) :: tree
# 562 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 563 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 564 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(OUT), OPTIONAL :: ge_position
         LOGICAL, INTENT(IN), OPTIONAL :: short
         INTEGER :: gti ! Used mark searches
         LOGICAL :: stop_short
         !
         stop_short = .FALSE.
         IF (PRESENT(short)) stop_short = short
         NULLIFY (node)
         position = 0
         IF (PRESENT(ge_position)) ge_position = 0
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            CALL btree_node_find_ge_pos_i8_sp2d (node%keys, key, position, node%filled)
# 585 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            IF (stop_short .AND. position .LE. node%filled) THEN
               IF (node%keys(position) .EQ. key) THEN
                  IF (PRESENT(ge_position)) ge_position = position
                  RETURN
               END IF
            END IF
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            !CALL btree_node_find_gt_pos_i8_sp2d (node%keys, key, gti, node%filled, position)
# 601 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            CALL btree_node_find_gte_pos_i8_sp2d (node%keys, key, gti, node%filled, position)
# 602 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               IF (PRESENT(ge_position)) ge_position = gti
               position = 0
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_full_i8_sp2d
# 611 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_leaf_i8_sp2d (tree, key, node, gti)
# 616 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_sp2d), INTENT(IN) :: tree
# 617 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 618 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_sp2d), POINTER :: node
# 619 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: gti
         !
         NULLIFY (node)
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            !CALL btree_node_find_ge_pos_i8_sp2d (node%keys, key, position, node%filled)
# 632 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            !
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            CALL btree_node_find_gt_pos_i8_sp2d (node%keys, key, gti, node%filled)
# 643 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !CALL btree_node_find_gt2_pos_i8_sp2d (node%keys, key, i, node%filled)
# 644 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !IF (i .NE. gti) WRITE(*,*)'XXXX difference',i,gti
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_leaf_i8_sp2d
# 652 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 93 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      SUBROUTINE btree_new_i8_dp2d (tree, order)
# 94 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(OUT) :: tree
# 95 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: order
         INTEGER :: maxs, mins
         !
         IF (PRESENT(order)) THEN
            maxs = order - 1
         ELSE
            maxs = 15
         END IF
         mins = ISHFT(maxs, -1)
         IF (mins*2 .GT. maxs) maxs = 2*maxs
         IF (mins .LT. 1) mins = 1
         IF (maxs .LT. 3) maxs = 3
         tree%b%min_fill = mins
         tree%b%max_fill = maxs
         tree%b%refcount = 1
         tree%b%n = 0
         NULLIFY (tree%b%root)
         tree%b%lastid = 0
      END SUBROUTINE btree_new_i8_dp2d
# 114 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      FUNCTION btree_get_entries_i8_dp2d (tree) RESULT(num_entries)
# 116 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(INOUT) :: tree
# 117 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: num_entries
         num_entries = tree%b%n
      END FUNCTION btree_get_entries_i8_dp2d
# 120 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is a non-leaf node
      SUBROUTINE btree_adopt_subtrees_i8_dp2d (node)
# 123 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 124 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ! Assume that node is not a leaf!
         DO i = 1, node%filled + 1
            !IF (ASSOCIATED (node%subtrees(i)%node)) THEN
            !IF (.NOT. ASSOCIATED (node%subtrees(i)%node%parent,&
            ! node)) THEN
            node%subtrees(i)%node%parent => node
            !ENDIF
            !ENDIF
         END DO
      END SUBROUTINE btree_adopt_subtrees_i8_dp2d
# 136 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_delete_i8_dp2d (tree, keys, values)
# 138 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(INOUT) :: tree
# 139 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(OUT), OPTIONAL :: keys
# 140 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), DIMENSION(:), INTENT(OUT), OPTIONAL :: values
# 141 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: pos
         !
         IF (ASSOCIATED(tree%b%root)) THEN
            pos = 0
            IF (PRESENT(keys) .AND. PRESENT(values)) THEN
               pos = 1
               CALL btree_delete_node_i8_dp2d (tree%b%root, pos, keys, values)
# 148 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_delete_node_i8_dp2d (tree%b%root)
# 150 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
         END IF
         NULLIFY (tree%b%root)
      END SUBROUTINE btree_delete_i8_dp2d
# 154 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_delete_node_i8_dp2d (node, pos, keys, values)
# 156 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 157 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(INOUT), OPTIONAL :: pos
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(INOUT), OPTIONAL :: keys
# 159 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), DIMENSION(:), INTENT(INOUT), OPTIONAL :: values
# 160 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         INTEGER :: i
         !
         IF (node%filled .GT. 0 .AND. ASSOCIATED(node%subtrees(1)%node)) THEN
            DO i = 1, node%filled + 1
               IF (PRESENT(pos)) THEN
                  CALL btree_delete_node_i8_dp2d (node%subtrees(i)%node, pos, keys, values)
# 167 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_delete_node_i8_dp2d (node%subtrees(i)%node)
# 169 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               END IF
               IF (PRESENT(pos) .AND. i .LE. node%filled) THEN
                  keys(pos) = node%keys(i)
                  values(pos) = node%values(i)
                  pos = pos + 1
               END IF
            END DO
         ELSEIF (PRESENT(pos) .AND. node%filled .GT. 0) THEN
            keys(pos:pos + node%filled - 1) = node%keys(1:node%filled)
            values(pos:pos + node%filled - 1) = node%values(1:node%filled)
            pos = pos + node%filled
         END IF
         CALL btree_free_node_i8_dp2d (node)
# 182 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END SUBROUTINE btree_delete_node_i8_dp2d
# 183 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! Find the key
      ! IF node still has space, insert & update the node
      ! else
      ! 1. select median
      ! 2. split keys into two nodes (one is new)
      ! 3. insert separation key put into parent, and repeat upwards

      SUBROUTINE btree_add_i8_dp2d (tree, key, value, exists, existing_value, replace)
# 192 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(INOUT) :: tree
# 193 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 194 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(IN) :: value
# 195 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         TYPE(btree_data_dp2d), INTENT(OUT), OPTIONAL :: existing_value
# 197 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(IN), OPTIONAL :: replace
         !
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 200 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, position
         !
         IF (PRESENT(exists)) THEN
            CALL btree_find_full_i8_dp2d (tree, key, node, position, ge_pos, short=.TRUE.)
# 204 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (position .GT. 0) THEN
               exists = .TRUE.
               existing_value = node%values(position)
               IF (PRESENT(replace)) THEN
                  IF (replace) THEN
                     node%values(position) = value
                  END IF
               END IF
               RETURN
            ELSE
               exists = .FALSE.
            END IF
         ELSE
            CALL btree_find_leaf_i8_dp2d (tree, key, node, ge_pos)
# 218 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         CALL btree_add_into_i8_dp2d (tree, node, key, value, before=ge_pos)
# 220 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) existing_value = value
         tree%b%n = tree%b%n + 1
      END SUBROUTINE btree_add_i8_dp2d
# 223 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_add_into_i8_dp2d (tree, node, key, value, before, subtree)
# 225 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(INOUT) :: tree
# 226 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 227 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 228 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(IN) :: value
# 229 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: before
         TYPE(btree_node_i8_dp2d), POINTER, OPTIONAL :: subtree
# 231 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         TYPE(btree_node_i8_dp2d), POINTER :: new_node
# 233 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt) :: upgrade_key
# 234 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, split_pos
         TYPE(btree_data_dp2d) :: upgrade_value
# 236 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL :: leaf
         !
         ! Root is special
         IF (.NOT. ASSOCIATED(node)) THEN
            CALL btree_new_root_i8_dp2d (tree, key, value)
# 241 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (PRESENT(subtree)) THEN
               tree%b%root%subtrees(2)%node => subtree
               subtree%parent => tree%b%root
            END IF
            RETURN
         END IF
         ! Where the insertion takes place.
         IF (PRESENT(before)) THEN
            ge_pos = before
         ELSE
            CALL btree_node_find_gt_pos_i8_dp2d (node%keys, key, ge_pos, node%filled)
# 252 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         ! Addition is easy if the node has enough space.
         leaf = .NOT. ASSOCIATED(node%subtrees(1)%node)
         IF (node%filled .LT. tree%b%max_fill) THEN
            IF (PRESENT(subtree)) THEN
               CALL btree_simple_insertion_i8_dp2d (node, key, value, ge_pos, subtree)
# 258 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_simple_insertion_i8_dp2d (node, key, value, ge_pos)
# 260 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            RETURN
         ELSE
            split_pos = ISHFT(tree%b%max_fill + 1, -1)
            ! I assert that split_pos <= SIZE(node%keys)
            CALL btree_new_node_i8_dp2d (tree, new_node)
# 266 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! The key to be added falls in the left node
            node%filled = split_pos - 1
            IF (ge_pos .LE. split_pos) THEN
               IF (ge_pos .EQ. split_pos) THEN
                  upgrade_key = key
                  upgrade_value = value
               ELSE
                  upgrade_key = node%keys(split_pos - 1)
                  upgrade_value = node%values(split_pos - 1)
               END IF
               IF (PRESENT(subtree)) THEN
                  CALL btree_left_insertion_i8_dp2d (tree, node, new_node, key, value, &
# 278 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_dp2d (new_node)
# 280 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_left_insertion_i8_dp2d (tree, node, new_node, key, value, &
# 282 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos)
               END IF
               !
            ELSE
               upgrade_key = node%keys(split_pos)
               upgrade_value = node%values(split_pos)
               IF (PRESENT(subtree)) THEN
                  CALL btree_right_insertion_i8_dp2d (tree, node, new_node, key, value, &
# 290 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_dp2d (new_node)
# 292 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_right_insertion_i8_dp2d (tree, node, new_node, key, value, &
# 294 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos)
               END IF
               !
            END IF
            !
            new_node%parent => node%parent
            !
            IF (.NOT. leaf) THEN
               CALL btree_adopt_subtrees_i8_dp2d (new_node)
# 303 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            !
            CALL btree_add_into_i8_dp2d (tree, node%parent, upgrade_key, upgrade_value, &
# 306 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                              subtree=new_node)
            !
         END IF
      END SUBROUTINE btree_add_into_i8_dp2d
# 310 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_simple_insertion_i8_dp2d (node, key, value, before, subtree)
# 312 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), INTENT(INOUT) :: node
# 313 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 314 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(IN) :: value
# 315 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before
         TYPE(btree_node_i8_dp2d), POINTER, OPTIONAL :: subtree
# 317 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         ! Shift keys
         node%keys(before + 1:node%filled + 1) = node%keys(before:node%filled)
         node%keys(before) = key
         ! Shift values
         node%values(before + 1:node%filled + 1) = node%values(before:node%filled)
         node%values(before) = value
         ! Shift subtree pointers, but only if node is not a leaf ; assume
         ! leaf <=> present(subtree)
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 2) = &
               node%subtrees(before + 1:node%filled + 1)
            node%subtrees(before + 1)%node => subtree
         END IF
         node%filled = node%filled + 1
      END SUBROUTINE btree_simple_insertion_i8_dp2d
# 333 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_left_insertion_i8_dp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 335 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(IN) :: tree
# 336 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), INTENT(INOUT) :: node, new_node
# 337 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 338 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(IN) :: value
# 339 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_dp2d), POINTER, OPTIONAL :: subtree
# 341 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill) - (split_pos - 1)
         new_node%keys(1:new_node%filled) = &
            node%keys(split_pos:tree%b%max_fill)
         new_node%values(1:new_node%filled) = &
            node%values(split_pos:tree%b%max_fill)
         !IF (ASSOCIATED (node%subtrees(1)%node)) THEN
         IF (PRESENT(subtree)) THEN
            IF (before .EQ. split_pos) THEN
               new_node%subtrees(2:new_node%filled + 1) = &
                  node%subtrees(split_pos + 1:tree%b%max_fill + 1)
               new_node%subtrees(1)%node => subtree
            ELSE
               new_node%subtrees(1:new_node%filled + 1) = &
                  node%subtrees(split_pos:tree%b%max_fill + 1)
            END IF
         END IF
         ! Fill node%{keys,values}(1:node%filled), where node%filled
         ! is split_pos-1, but do insert the new value at ge_pos. The
         ! key/value at split_pos is to be inserted into the
         ! parent.
         ! The new tree is added to the right of the new insertion.
         node%keys(before + 1:node%filled) = node%keys(before:node%filled - 1)
         node%keys(before) = key
         node%values(before + 1:node%filled) = node%values(before:node%filled - 1)
         node%values(before) = value
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 1) = &
               node%subtrees(before + 1:node%filled)
            node%subtrees(before + 1)%node => subtree
         ELSE
            NULLIFY (node%subtrees(before + 1)%node)
         END IF
      END SUBROUTINE btree_left_insertion_i8_dp2d
# 375 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_right_insertion_i8_dp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 377 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(IN) :: tree
# 378 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), INTENT(INOUT) :: node, new_node
# 379 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 380 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(IN) :: value
# 381 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_dp2d), POINTER, OPTIONAL :: subtree
# 383 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill + 1) - split_pos
         new_node%keys(1:before - split_pos - 1) = &
            node%keys(split_pos + 1:before - 1)
         new_node%keys(before - split_pos) = key
         new_node%keys(before - split_pos + 1:new_node%filled) = &
            node%keys(before:tree%b%max_fill)
         new_node%values(1:before - split_pos - 1) = &
            node%values(split_pos + 1:before - 1)
         new_node%values(before - split_pos) = value
         new_node%values(before - split_pos + 1:new_node%filled) = &
            node%values(before:tree%b%max_fill)
         IF (PRESENT(subtree)) THEN
            new_node%subtrees(1:before - split_pos) = &
               node%subtrees(split_pos + 1:before)
            new_node%subtrees(before - split_pos + 1)%node => subtree
            new_node%subtrees(before - split_pos + 2:new_node%filled + 1) = &
               node%subtrees(before + 1:tree%b%max_fill + 1)
         END IF
      END SUBROUTINE btree_right_insertion_i8_dp2d
# 403 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_root_i8_dp2d (tree, key, value)
# 405 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(INOUT) :: tree
# 406 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 407 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(IN) :: value
# 408 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: old_root, new_root
# 409 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         CALL btree_new_node_i8_dp2d (tree, new_root)
# 411 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         new_root%filled = 1
         new_root%keys(1) = key
         new_root%values(1) = value
         IF (ASSOCIATED(tree%b%root)) THEN
            old_root => tree%b%root
            old_root%parent => new_root
            new_root%subtrees(1)%node => old_root
            old_root%parent => new_root
         END IF
         tree%b%root => new_root
      END SUBROUTINE btree_new_root_i8_dp2d
# 422 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_node_i8_dp2d (tree, node)
# 424 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(INOUT) :: tree
# 425 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 426 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ALLOCATE (node)
         ALLOCATE (node%keys(tree%b%max_fill))
         ALLOCATE (node%values(tree%b%max_fill))
         ALLOCATE (node%subtrees(tree%b%max_fill + 1))
         DO i = 1, tree%b%max_fill + 1
            NULLIFY (node%subtrees(i)%node)
         END DO
         node%filled = 0
         NULLIFY (node%parent)
         tree%b%lastid = tree%b%lastid + 1
         node%id = tree%b%lastid
      END SUBROUTINE btree_new_node_i8_dp2d
# 440 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_free_node_i8_dp2d (node)
# 442 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 443 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         DEALLOCATE (node%keys)
         DEALLOCATE (node%values)
         DEALLOCATE (node%subtrees)
         DEALLOCATE (node)
      END SUBROUTINE btree_free_node_i8_dp2d
# 449 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_find_i8_dp2d (tree, key, value, exists)
# 451 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(IN) :: tree
# 452 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 453 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_dp2d), INTENT(OUT) :: value
# 454 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         !
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 457 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: position
         !
         CALL btree_find_full_i8_dp2d (tree, key, node, position, short=.TRUE.)
# 460 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) THEN
            exists = position .GT. 0
         END IF
         IF (position .GT. 0) THEN
            value = node%values(position)
         END IF
      END SUBROUTINE btree_find_i8_dp2d
# 467 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_ge_pos_i8_dp2d (keys, key, position, filled)
# 469 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 470 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 471 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GE. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LT. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GE. key .AND. keys(position - 1) .LT. key) THEN
               RETURN
            END IF
            IF (keys(position) .GE. key) right = MIN(position, right - 1)
            IF (keys(position) .LT. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_ge_pos_i8_dp2d
# 495 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gt_pos_i8_dp2d (keys, key, position, filled)
# 497 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 498 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 499 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GT. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gt_pos_i8_dp2d
# 523 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gte_pos_i8_dp2d (keys, key, position, filled, first)
# 525 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 526 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 527 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER, INTENT(IN), OPTIONAL :: first
         INTEGER :: left, right, one
         !
         one = 1
         IF (PRESENT(FIRST)) one = first
         IF (one .LE. filled) THEN
            IF (keys(one) .GT. key) THEN
               position = one
               RETURN
            END IF
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = one + 1
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gte_pos_i8_dp2d
# 556 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_full_i8_dp2d (tree, key, node, position, ge_position, short)
# 561 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(IN) :: tree
# 562 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 563 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 564 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(OUT), OPTIONAL :: ge_position
         LOGICAL, INTENT(IN), OPTIONAL :: short
         INTEGER :: gti ! Used mark searches
         LOGICAL :: stop_short
         !
         stop_short = .FALSE.
         IF (PRESENT(short)) stop_short = short
         NULLIFY (node)
         position = 0
         IF (PRESENT(ge_position)) ge_position = 0
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            CALL btree_node_find_ge_pos_i8_dp2d (node%keys, key, position, node%filled)
# 585 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            IF (stop_short .AND. position .LE. node%filled) THEN
               IF (node%keys(position) .EQ. key) THEN
                  IF (PRESENT(ge_position)) ge_position = position
                  RETURN
               END IF
            END IF
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            !CALL btree_node_find_gt_pos_i8_dp2d (node%keys, key, gti, node%filled, position)
# 601 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            CALL btree_node_find_gte_pos_i8_dp2d (node%keys, key, gti, node%filled, position)
# 602 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               IF (PRESENT(ge_position)) ge_position = gti
               position = 0
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_full_i8_dp2d
# 611 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_leaf_i8_dp2d (tree, key, node, gti)
# 616 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_dp2d), INTENT(IN) :: tree
# 617 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 618 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_dp2d), POINTER :: node
# 619 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: gti
         !
         NULLIFY (node)
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            !CALL btree_node_find_ge_pos_i8_dp2d (node%keys, key, position, node%filled)
# 632 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            !
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            CALL btree_node_find_gt_pos_i8_dp2d (node%keys, key, gti, node%filled)
# 643 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !CALL btree_node_find_gt2_pos_i8_dp2d (node%keys, key, i, node%filled)
# 644 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !IF (i .NE. gti) WRITE(*,*)'XXXX difference',i,gti
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_leaf_i8_dp2d
# 652 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 93 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      SUBROUTINE btree_new_i8_cp2d (tree, order)
# 94 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(OUT) :: tree
# 95 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: order
         INTEGER :: maxs, mins
         !
         IF (PRESENT(order)) THEN
            maxs = order - 1
         ELSE
            maxs = 15
         END IF
         mins = ISHFT(maxs, -1)
         IF (mins*2 .GT. maxs) maxs = 2*maxs
         IF (mins .LT. 1) mins = 1
         IF (maxs .LT. 3) maxs = 3
         tree%b%min_fill = mins
         tree%b%max_fill = maxs
         tree%b%refcount = 1
         tree%b%n = 0
         NULLIFY (tree%b%root)
         tree%b%lastid = 0
      END SUBROUTINE btree_new_i8_cp2d
# 114 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      FUNCTION btree_get_entries_i8_cp2d (tree) RESULT(num_entries)
# 116 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(INOUT) :: tree
# 117 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: num_entries
         num_entries = tree%b%n
      END FUNCTION btree_get_entries_i8_cp2d
# 120 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is a non-leaf node
      SUBROUTINE btree_adopt_subtrees_i8_cp2d (node)
# 123 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 124 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ! Assume that node is not a leaf!
         DO i = 1, node%filled + 1
            !IF (ASSOCIATED (node%subtrees(i)%node)) THEN
            !IF (.NOT. ASSOCIATED (node%subtrees(i)%node%parent,&
            ! node)) THEN
            node%subtrees(i)%node%parent => node
            !ENDIF
            !ENDIF
         END DO
      END SUBROUTINE btree_adopt_subtrees_i8_cp2d
# 136 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_delete_i8_cp2d (tree, keys, values)
# 138 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(INOUT) :: tree
# 139 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(OUT), OPTIONAL :: keys
# 140 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), DIMENSION(:), INTENT(OUT), OPTIONAL :: values
# 141 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: pos
         !
         IF (ASSOCIATED(tree%b%root)) THEN
            pos = 0
            IF (PRESENT(keys) .AND. PRESENT(values)) THEN
               pos = 1
               CALL btree_delete_node_i8_cp2d (tree%b%root, pos, keys, values)
# 148 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_delete_node_i8_cp2d (tree%b%root)
# 150 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
         END IF
         NULLIFY (tree%b%root)
      END SUBROUTINE btree_delete_i8_cp2d
# 154 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_delete_node_i8_cp2d (node, pos, keys, values)
# 156 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 157 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(INOUT), OPTIONAL :: pos
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(INOUT), OPTIONAL :: keys
# 159 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), DIMENSION(:), INTENT(INOUT), OPTIONAL :: values
# 160 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         INTEGER :: i
         !
         IF (node%filled .GT. 0 .AND. ASSOCIATED(node%subtrees(1)%node)) THEN
            DO i = 1, node%filled + 1
               IF (PRESENT(pos)) THEN
                  CALL btree_delete_node_i8_cp2d (node%subtrees(i)%node, pos, keys, values)
# 167 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_delete_node_i8_cp2d (node%subtrees(i)%node)
# 169 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               END IF
               IF (PRESENT(pos) .AND. i .LE. node%filled) THEN
                  keys(pos) = node%keys(i)
                  values(pos) = node%values(i)
                  pos = pos + 1
               END IF
            END DO
         ELSEIF (PRESENT(pos) .AND. node%filled .GT. 0) THEN
            keys(pos:pos + node%filled - 1) = node%keys(1:node%filled)
            values(pos:pos + node%filled - 1) = node%values(1:node%filled)
            pos = pos + node%filled
         END IF
         CALL btree_free_node_i8_cp2d (node)
# 182 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END SUBROUTINE btree_delete_node_i8_cp2d
# 183 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! Find the key
      ! IF node still has space, insert & update the node
      ! else
      ! 1. select median
      ! 2. split keys into two nodes (one is new)
      ! 3. insert separation key put into parent, and repeat upwards

      SUBROUTINE btree_add_i8_cp2d (tree, key, value, exists, existing_value, replace)
# 192 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(INOUT) :: tree
# 193 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 194 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(IN) :: value
# 195 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         TYPE(btree_data_cp2d), INTENT(OUT), OPTIONAL :: existing_value
# 197 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(IN), OPTIONAL :: replace
         !
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 200 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, position
         !
         IF (PRESENT(exists)) THEN
            CALL btree_find_full_i8_cp2d (tree, key, node, position, ge_pos, short=.TRUE.)
# 204 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (position .GT. 0) THEN
               exists = .TRUE.
               existing_value = node%values(position)
               IF (PRESENT(replace)) THEN
                  IF (replace) THEN
                     node%values(position) = value
                  END IF
               END IF
               RETURN
            ELSE
               exists = .FALSE.
            END IF
         ELSE
            CALL btree_find_leaf_i8_cp2d (tree, key, node, ge_pos)
# 218 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         CALL btree_add_into_i8_cp2d (tree, node, key, value, before=ge_pos)
# 220 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) existing_value = value
         tree%b%n = tree%b%n + 1
      END SUBROUTINE btree_add_i8_cp2d
# 223 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_add_into_i8_cp2d (tree, node, key, value, before, subtree)
# 225 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(INOUT) :: tree
# 226 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 227 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 228 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(IN) :: value
# 229 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: before
         TYPE(btree_node_i8_cp2d), POINTER, OPTIONAL :: subtree
# 231 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         TYPE(btree_node_i8_cp2d), POINTER :: new_node
# 233 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt) :: upgrade_key
# 234 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, split_pos
         TYPE(btree_data_cp2d) :: upgrade_value
# 236 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL :: leaf
         !
         ! Root is special
         IF (.NOT. ASSOCIATED(node)) THEN
            CALL btree_new_root_i8_cp2d (tree, key, value)
# 241 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (PRESENT(subtree)) THEN
               tree%b%root%subtrees(2)%node => subtree
               subtree%parent => tree%b%root
            END IF
            RETURN
         END IF
         ! Where the insertion takes place.
         IF (PRESENT(before)) THEN
            ge_pos = before
         ELSE
            CALL btree_node_find_gt_pos_i8_cp2d (node%keys, key, ge_pos, node%filled)
# 252 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         ! Addition is easy if the node has enough space.
         leaf = .NOT. ASSOCIATED(node%subtrees(1)%node)
         IF (node%filled .LT. tree%b%max_fill) THEN
            IF (PRESENT(subtree)) THEN
               CALL btree_simple_insertion_i8_cp2d (node, key, value, ge_pos, subtree)
# 258 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_simple_insertion_i8_cp2d (node, key, value, ge_pos)
# 260 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            RETURN
         ELSE
            split_pos = ISHFT(tree%b%max_fill + 1, -1)
            ! I assert that split_pos <= SIZE(node%keys)
            CALL btree_new_node_i8_cp2d (tree, new_node)
# 266 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! The key to be added falls in the left node
            node%filled = split_pos - 1
            IF (ge_pos .LE. split_pos) THEN
               IF (ge_pos .EQ. split_pos) THEN
                  upgrade_key = key
                  upgrade_value = value
               ELSE
                  upgrade_key = node%keys(split_pos - 1)
                  upgrade_value = node%values(split_pos - 1)
               END IF
               IF (PRESENT(subtree)) THEN
                  CALL btree_left_insertion_i8_cp2d (tree, node, new_node, key, value, &
# 278 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_cp2d (new_node)
# 280 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_left_insertion_i8_cp2d (tree, node, new_node, key, value, &
# 282 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos)
               END IF
               !
            ELSE
               upgrade_key = node%keys(split_pos)
               upgrade_value = node%values(split_pos)
               IF (PRESENT(subtree)) THEN
                  CALL btree_right_insertion_i8_cp2d (tree, node, new_node, key, value, &
# 290 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_cp2d (new_node)
# 292 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_right_insertion_i8_cp2d (tree, node, new_node, key, value, &
# 294 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos)
               END IF
               !
            END IF
            !
            new_node%parent => node%parent
            !
            IF (.NOT. leaf) THEN
               CALL btree_adopt_subtrees_i8_cp2d (new_node)
# 303 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            !
            CALL btree_add_into_i8_cp2d (tree, node%parent, upgrade_key, upgrade_value, &
# 306 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                              subtree=new_node)
            !
         END IF
      END SUBROUTINE btree_add_into_i8_cp2d
# 310 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_simple_insertion_i8_cp2d (node, key, value, before, subtree)
# 312 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), INTENT(INOUT) :: node
# 313 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 314 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(IN) :: value
# 315 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before
         TYPE(btree_node_i8_cp2d), POINTER, OPTIONAL :: subtree
# 317 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         ! Shift keys
         node%keys(before + 1:node%filled + 1) = node%keys(before:node%filled)
         node%keys(before) = key
         ! Shift values
         node%values(before + 1:node%filled + 1) = node%values(before:node%filled)
         node%values(before) = value
         ! Shift subtree pointers, but only if node is not a leaf ; assume
         ! leaf <=> present(subtree)
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 2) = &
               node%subtrees(before + 1:node%filled + 1)
            node%subtrees(before + 1)%node => subtree
         END IF
         node%filled = node%filled + 1
      END SUBROUTINE btree_simple_insertion_i8_cp2d
# 333 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_left_insertion_i8_cp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 335 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(IN) :: tree
# 336 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), INTENT(INOUT) :: node, new_node
# 337 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 338 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(IN) :: value
# 339 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_cp2d), POINTER, OPTIONAL :: subtree
# 341 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill) - (split_pos - 1)
         new_node%keys(1:new_node%filled) = &
            node%keys(split_pos:tree%b%max_fill)
         new_node%values(1:new_node%filled) = &
            node%values(split_pos:tree%b%max_fill)
         !IF (ASSOCIATED (node%subtrees(1)%node)) THEN
         IF (PRESENT(subtree)) THEN
            IF (before .EQ. split_pos) THEN
               new_node%subtrees(2:new_node%filled + 1) = &
                  node%subtrees(split_pos + 1:tree%b%max_fill + 1)
               new_node%subtrees(1)%node => subtree
            ELSE
               new_node%subtrees(1:new_node%filled + 1) = &
                  node%subtrees(split_pos:tree%b%max_fill + 1)
            END IF
         END IF
         ! Fill node%{keys,values}(1:node%filled), where node%filled
         ! is split_pos-1, but do insert the new value at ge_pos. The
         ! key/value at split_pos is to be inserted into the
         ! parent.
         ! The new tree is added to the right of the new insertion.
         node%keys(before + 1:node%filled) = node%keys(before:node%filled - 1)
         node%keys(before) = key
         node%values(before + 1:node%filled) = node%values(before:node%filled - 1)
         node%values(before) = value
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 1) = &
               node%subtrees(before + 1:node%filled)
            node%subtrees(before + 1)%node => subtree
         ELSE
            NULLIFY (node%subtrees(before + 1)%node)
         END IF
      END SUBROUTINE btree_left_insertion_i8_cp2d
# 375 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_right_insertion_i8_cp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 377 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(IN) :: tree
# 378 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), INTENT(INOUT) :: node, new_node
# 379 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 380 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(IN) :: value
# 381 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_cp2d), POINTER, OPTIONAL :: subtree
# 383 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill + 1) - split_pos
         new_node%keys(1:before - split_pos - 1) = &
            node%keys(split_pos + 1:before - 1)
         new_node%keys(before - split_pos) = key
         new_node%keys(before - split_pos + 1:new_node%filled) = &
            node%keys(before:tree%b%max_fill)
         new_node%values(1:before - split_pos - 1) = &
            node%values(split_pos + 1:before - 1)
         new_node%values(before - split_pos) = value
         new_node%values(before - split_pos + 1:new_node%filled) = &
            node%values(before:tree%b%max_fill)
         IF (PRESENT(subtree)) THEN
            new_node%subtrees(1:before - split_pos) = &
               node%subtrees(split_pos + 1:before)
            new_node%subtrees(before - split_pos + 1)%node => subtree
            new_node%subtrees(before - split_pos + 2:new_node%filled + 1) = &
               node%subtrees(before + 1:tree%b%max_fill + 1)
         END IF
      END SUBROUTINE btree_right_insertion_i8_cp2d
# 403 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_root_i8_cp2d (tree, key, value)
# 405 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(INOUT) :: tree
# 406 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 407 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(IN) :: value
# 408 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: old_root, new_root
# 409 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         CALL btree_new_node_i8_cp2d (tree, new_root)
# 411 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         new_root%filled = 1
         new_root%keys(1) = key
         new_root%values(1) = value
         IF (ASSOCIATED(tree%b%root)) THEN
            old_root => tree%b%root
            old_root%parent => new_root
            new_root%subtrees(1)%node => old_root
            old_root%parent => new_root
         END IF
         tree%b%root => new_root
      END SUBROUTINE btree_new_root_i8_cp2d
# 422 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_node_i8_cp2d (tree, node)
# 424 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(INOUT) :: tree
# 425 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 426 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ALLOCATE (node)
         ALLOCATE (node%keys(tree%b%max_fill))
         ALLOCATE (node%values(tree%b%max_fill))
         ALLOCATE (node%subtrees(tree%b%max_fill + 1))
         DO i = 1, tree%b%max_fill + 1
            NULLIFY (node%subtrees(i)%node)
         END DO
         node%filled = 0
         NULLIFY (node%parent)
         tree%b%lastid = tree%b%lastid + 1
         node%id = tree%b%lastid
      END SUBROUTINE btree_new_node_i8_cp2d
# 440 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_free_node_i8_cp2d (node)
# 442 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 443 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         DEALLOCATE (node%keys)
         DEALLOCATE (node%values)
         DEALLOCATE (node%subtrees)
         DEALLOCATE (node)
      END SUBROUTINE btree_free_node_i8_cp2d
# 449 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_find_i8_cp2d (tree, key, value, exists)
# 451 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(IN) :: tree
# 452 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 453 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_cp2d), INTENT(OUT) :: value
# 454 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         !
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 457 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: position
         !
         CALL btree_find_full_i8_cp2d (tree, key, node, position, short=.TRUE.)
# 460 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) THEN
            exists = position .GT. 0
         END IF
         IF (position .GT. 0) THEN
            value = node%values(position)
         END IF
      END SUBROUTINE btree_find_i8_cp2d
# 467 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_ge_pos_i8_cp2d (keys, key, position, filled)
# 469 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 470 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 471 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GE. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LT. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GE. key .AND. keys(position - 1) .LT. key) THEN
               RETURN
            END IF
            IF (keys(position) .GE. key) right = MIN(position, right - 1)
            IF (keys(position) .LT. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_ge_pos_i8_cp2d
# 495 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gt_pos_i8_cp2d (keys, key, position, filled)
# 497 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 498 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 499 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GT. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gt_pos_i8_cp2d
# 523 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gte_pos_i8_cp2d (keys, key, position, filled, first)
# 525 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 526 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 527 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER, INTENT(IN), OPTIONAL :: first
         INTEGER :: left, right, one
         !
         one = 1
         IF (PRESENT(FIRST)) one = first
         IF (one .LE. filled) THEN
            IF (keys(one) .GT. key) THEN
               position = one
               RETURN
            END IF
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = one + 1
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gte_pos_i8_cp2d
# 556 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_full_i8_cp2d (tree, key, node, position, ge_position, short)
# 561 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(IN) :: tree
# 562 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 563 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 564 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(OUT), OPTIONAL :: ge_position
         LOGICAL, INTENT(IN), OPTIONAL :: short
         INTEGER :: gti ! Used mark searches
         LOGICAL :: stop_short
         !
         stop_short = .FALSE.
         IF (PRESENT(short)) stop_short = short
         NULLIFY (node)
         position = 0
         IF (PRESENT(ge_position)) ge_position = 0
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            CALL btree_node_find_ge_pos_i8_cp2d (node%keys, key, position, node%filled)
# 585 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            IF (stop_short .AND. position .LE. node%filled) THEN
               IF (node%keys(position) .EQ. key) THEN
                  IF (PRESENT(ge_position)) ge_position = position
                  RETURN
               END IF
            END IF
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            !CALL btree_node_find_gt_pos_i8_cp2d (node%keys, key, gti, node%filled, position)
# 601 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            CALL btree_node_find_gte_pos_i8_cp2d (node%keys, key, gti, node%filled, position)
# 602 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               IF (PRESENT(ge_position)) ge_position = gti
               position = 0
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_full_i8_cp2d
# 611 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_leaf_i8_cp2d (tree, key, node, gti)
# 616 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_cp2d), INTENT(IN) :: tree
# 617 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 618 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_cp2d), POINTER :: node
# 619 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: gti
         !
         NULLIFY (node)
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            !CALL btree_node_find_ge_pos_i8_cp2d (node%keys, key, position, node%filled)
# 632 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            !
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            CALL btree_node_find_gt_pos_i8_cp2d (node%keys, key, gti, node%filled)
# 643 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !CALL btree_node_find_gt2_pos_i8_cp2d (node%keys, key, i, node%filled)
# 644 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !IF (i .NE. gti) WRITE(*,*)'XXXX difference',i,gti
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_leaf_i8_cp2d
# 652 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 93 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      SUBROUTINE btree_new_i8_zp2d (tree, order)
# 94 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(OUT) :: tree
# 95 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: order
         INTEGER :: maxs, mins
         !
         IF (PRESENT(order)) THEN
            maxs = order - 1
         ELSE
            maxs = 15
         END IF
         mins = ISHFT(maxs, -1)
         IF (mins*2 .GT. maxs) maxs = 2*maxs
         IF (mins .LT. 1) mins = 1
         IF (maxs .LT. 3) maxs = 3
         tree%b%min_fill = mins
         tree%b%max_fill = maxs
         tree%b%refcount = 1
         tree%b%n = 0
         NULLIFY (tree%b%root)
         tree%b%lastid = 0
      END SUBROUTINE btree_new_i8_zp2d
# 114 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      FUNCTION btree_get_entries_i8_zp2d (tree) RESULT(num_entries)
# 116 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(INOUT) :: tree
# 117 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: num_entries
         num_entries = tree%b%n
      END FUNCTION btree_get_entries_i8_zp2d
# 120 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is a non-leaf node
      SUBROUTINE btree_adopt_subtrees_i8_zp2d (node)
# 123 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 124 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ! Assume that node is not a leaf!
         DO i = 1, node%filled + 1
            !IF (ASSOCIATED (node%subtrees(i)%node)) THEN
            !IF (.NOT. ASSOCIATED (node%subtrees(i)%node%parent,&
            ! node)) THEN
            node%subtrees(i)%node%parent => node
            !ENDIF
            !ENDIF
         END DO
      END SUBROUTINE btree_adopt_subtrees_i8_zp2d
# 136 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_delete_i8_zp2d (tree, keys, values)
# 138 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(INOUT) :: tree
# 139 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(OUT), OPTIONAL :: keys
# 140 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), DIMENSION(:), INTENT(OUT), OPTIONAL :: values
# 141 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: pos
         !
         IF (ASSOCIATED(tree%b%root)) THEN
            pos = 0
            IF (PRESENT(keys) .AND. PRESENT(values)) THEN
               pos = 1
               CALL btree_delete_node_i8_zp2d (tree%b%root, pos, keys, values)
# 148 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_delete_node_i8_zp2d (tree%b%root)
# 150 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
         END IF
         NULLIFY (tree%b%root)
      END SUBROUTINE btree_delete_i8_zp2d
# 154 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_delete_node_i8_zp2d (node, pos, keys, values)
# 156 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 157 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(INOUT), OPTIONAL :: pos
         INTEGER(KIND=keyt), DIMENSION(:), INTENT(INOUT), OPTIONAL :: keys
# 159 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), DIMENSION(:), INTENT(INOUT), OPTIONAL :: values
# 160 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         INTEGER :: i
         !
         IF (node%filled .GT. 0 .AND. ASSOCIATED(node%subtrees(1)%node)) THEN
            DO i = 1, node%filled + 1
               IF (PRESENT(pos)) THEN
                  CALL btree_delete_node_i8_zp2d (node%subtrees(i)%node, pos, keys, values)
# 167 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_delete_node_i8_zp2d (node%subtrees(i)%node)
# 169 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               END IF
               IF (PRESENT(pos) .AND. i .LE. node%filled) THEN
                  keys(pos) = node%keys(i)
                  values(pos) = node%values(i)
                  pos = pos + 1
               END IF
            END DO
         ELSEIF (PRESENT(pos) .AND. node%filled .GT. 0) THEN
            keys(pos:pos + node%filled - 1) = node%keys(1:node%filled)
            values(pos:pos + node%filled - 1) = node%values(1:node%filled)
            pos = pos + node%filled
         END IF
         CALL btree_free_node_i8_zp2d (node)
# 182 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
      END SUBROUTINE btree_delete_node_i8_zp2d
# 183 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! Find the key
      ! IF node still has space, insert & update the node
      ! else
      ! 1. select median
      ! 2. split keys into two nodes (one is new)
      ! 3. insert separation key put into parent, and repeat upwards

      SUBROUTINE btree_add_i8_zp2d (tree, key, value, exists, existing_value, replace)
# 192 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(INOUT) :: tree
# 193 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 194 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(IN) :: value
# 195 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         TYPE(btree_data_zp2d), INTENT(OUT), OPTIONAL :: existing_value
# 197 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(IN), OPTIONAL :: replace
         !
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 200 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, position
         !
         IF (PRESENT(exists)) THEN
            CALL btree_find_full_i8_zp2d (tree, key, node, position, ge_pos, short=.TRUE.)
# 204 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (position .GT. 0) THEN
               exists = .TRUE.
               existing_value = node%values(position)
               IF (PRESENT(replace)) THEN
                  IF (replace) THEN
                     node%values(position) = value
                  END IF
               END IF
               RETURN
            ELSE
               exists = .FALSE.
            END IF
         ELSE
            CALL btree_find_leaf_i8_zp2d (tree, key, node, ge_pos)
# 218 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         CALL btree_add_into_i8_zp2d (tree, node, key, value, before=ge_pos)
# 220 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) existing_value = value
         tree%b%n = tree%b%n + 1
      END SUBROUTINE btree_add_i8_zp2d
# 223 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      RECURSIVE SUBROUTINE btree_add_into_i8_zp2d (tree, node, key, value, before, subtree)
# 225 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(INOUT) :: tree
# 226 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 227 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 228 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(IN) :: value
# 229 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN), OPTIONAL :: before
         TYPE(btree_node_i8_zp2d), POINTER, OPTIONAL :: subtree
# 231 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         TYPE(btree_node_i8_zp2d), POINTER :: new_node
# 233 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt) :: upgrade_key
# 234 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: ge_pos, split_pos
         TYPE(btree_data_zp2d) :: upgrade_value
# 236 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL :: leaf
         !
         ! Root is special
         IF (.NOT. ASSOCIATED(node)) THEN
            CALL btree_new_root_i8_zp2d (tree, key, value)
# 241 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (PRESENT(subtree)) THEN
               tree%b%root%subtrees(2)%node => subtree
               subtree%parent => tree%b%root
            END IF
            RETURN
         END IF
         ! Where the insertion takes place.
         IF (PRESENT(before)) THEN
            ge_pos = before
         ELSE
            CALL btree_node_find_gt_pos_i8_zp2d (node%keys, key, ge_pos, node%filled)
# 252 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         END IF
         ! Addition is easy if the node has enough space.
         leaf = .NOT. ASSOCIATED(node%subtrees(1)%node)
         IF (node%filled .LT. tree%b%max_fill) THEN
            IF (PRESENT(subtree)) THEN
               CALL btree_simple_insertion_i8_zp2d (node, key, value, ge_pos, subtree)
# 258 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ELSE
               CALL btree_simple_insertion_i8_zp2d (node, key, value, ge_pos)
# 260 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            RETURN
         ELSE
            split_pos = ISHFT(tree%b%max_fill + 1, -1)
            ! I assert that split_pos <= SIZE(node%keys)
            CALL btree_new_node_i8_zp2d (tree, new_node)
# 266 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! The key to be added falls in the left node
            node%filled = split_pos - 1
            IF (ge_pos .LE. split_pos) THEN
               IF (ge_pos .EQ. split_pos) THEN
                  upgrade_key = key
                  upgrade_value = value
               ELSE
                  upgrade_key = node%keys(split_pos - 1)
                  upgrade_value = node%values(split_pos - 1)
               END IF
               IF (PRESENT(subtree)) THEN
                  CALL btree_left_insertion_i8_zp2d (tree, node, new_node, key, value, &
# 278 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_zp2d (new_node)
# 280 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_left_insertion_i8_zp2d (tree, node, new_node, key, value, &
# 282 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                          ge_pos, split_pos)
               END IF
               !
            ELSE
               upgrade_key = node%keys(split_pos)
               upgrade_value = node%values(split_pos)
               IF (PRESENT(subtree)) THEN
                  CALL btree_right_insertion_i8_zp2d (tree, node, new_node, key, value, &
# 290 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos, subtree)
                  !CALL btree_adopt_subtrees_i8_zp2d (new_node)
# 292 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
               ELSE
                  CALL btree_right_insertion_i8_zp2d (tree, node, new_node, key, value, &
# 294 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                                           ge_pos, split_pos)
               END IF
               !
            END IF
            !
            new_node%parent => node%parent
            !
            IF (.NOT. leaf) THEN
               CALL btree_adopt_subtrees_i8_zp2d (new_node)
# 303 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            END IF
            !
            CALL btree_add_into_i8_zp2d (tree, node%parent, upgrade_key, upgrade_value, &
# 306 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
                                              subtree=new_node)
            !
         END IF
      END SUBROUTINE btree_add_into_i8_zp2d
# 310 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_simple_insertion_i8_zp2d (node, key, value, before, subtree)
# 312 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), INTENT(INOUT) :: node
# 313 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 314 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(IN) :: value
# 315 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before
         TYPE(btree_node_i8_zp2d), POINTER, OPTIONAL :: subtree
# 317 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         ! Shift keys
         node%keys(before + 1:node%filled + 1) = node%keys(before:node%filled)
         node%keys(before) = key
         ! Shift values
         node%values(before + 1:node%filled + 1) = node%values(before:node%filled)
         node%values(before) = value
         ! Shift subtree pointers, but only if node is not a leaf ; assume
         ! leaf <=> present(subtree)
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 2) = &
               node%subtrees(before + 1:node%filled + 1)
            node%subtrees(before + 1)%node => subtree
         END IF
         node%filled = node%filled + 1
      END SUBROUTINE btree_simple_insertion_i8_zp2d
# 333 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_left_insertion_i8_zp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 335 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(IN) :: tree
# 336 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), INTENT(INOUT) :: node, new_node
# 337 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 338 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(IN) :: value
# 339 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_zp2d), POINTER, OPTIONAL :: subtree
# 341 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill) - (split_pos - 1)
         new_node%keys(1:new_node%filled) = &
            node%keys(split_pos:tree%b%max_fill)
         new_node%values(1:new_node%filled) = &
            node%values(split_pos:tree%b%max_fill)
         !IF (ASSOCIATED (node%subtrees(1)%node)) THEN
         IF (PRESENT(subtree)) THEN
            IF (before .EQ. split_pos) THEN
               new_node%subtrees(2:new_node%filled + 1) = &
                  node%subtrees(split_pos + 1:tree%b%max_fill + 1)
               new_node%subtrees(1)%node => subtree
            ELSE
               new_node%subtrees(1:new_node%filled + 1) = &
                  node%subtrees(split_pos:tree%b%max_fill + 1)
            END IF
         END IF
         ! Fill node%{keys,values}(1:node%filled), where node%filled
         ! is split_pos-1, but do insert the new value at ge_pos. The
         ! key/value at split_pos is to be inserted into the
         ! parent.
         ! The new tree is added to the right of the new insertion.
         node%keys(before + 1:node%filled) = node%keys(before:node%filled - 1)
         node%keys(before) = key
         node%values(before + 1:node%filled) = node%values(before:node%filled - 1)
         node%values(before) = value
         IF (PRESENT(subtree)) THEN
            node%subtrees(before + 2:node%filled + 1) = &
               node%subtrees(before + 1:node%filled)
            node%subtrees(before + 1)%node => subtree
         ELSE
            NULLIFY (node%subtrees(before + 1)%node)
         END IF
      END SUBROUTINE btree_left_insertion_i8_zp2d
# 375 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_right_insertion_i8_zp2d (tree, node, new_node, key, value, before, split_pos, subtree)
# 377 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(IN) :: tree
# 378 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), INTENT(INOUT) :: node, new_node
# 379 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 380 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(IN) :: value
# 381 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(IN) :: before, split_pos
         TYPE(btree_node_i8_zp2d), POINTER, OPTIONAL :: subtree
# 383 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         new_node%filled = (tree%b%max_fill + 1) - split_pos
         new_node%keys(1:before - split_pos - 1) = &
            node%keys(split_pos + 1:before - 1)
         new_node%keys(before - split_pos) = key
         new_node%keys(before - split_pos + 1:new_node%filled) = &
            node%keys(before:tree%b%max_fill)
         new_node%values(1:before - split_pos - 1) = &
            node%values(split_pos + 1:before - 1)
         new_node%values(before - split_pos) = value
         new_node%values(before - split_pos + 1:new_node%filled) = &
            node%values(before:tree%b%max_fill)
         IF (PRESENT(subtree)) THEN
            new_node%subtrees(1:before - split_pos) = &
               node%subtrees(split_pos + 1:before)
            new_node%subtrees(before - split_pos + 1)%node => subtree
            new_node%subtrees(before - split_pos + 2:new_node%filled + 1) = &
               node%subtrees(before + 1:tree%b%max_fill + 1)
         END IF
      END SUBROUTINE btree_right_insertion_i8_zp2d
# 403 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_root_i8_zp2d (tree, key, value)
# 405 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(INOUT) :: tree
# 406 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 407 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(IN) :: value
# 408 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: old_root, new_root
# 409 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         CALL btree_new_node_i8_zp2d (tree, new_root)
# 411 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         new_root%filled = 1
         new_root%keys(1) = key
         new_root%values(1) = value
         IF (ASSOCIATED(tree%b%root)) THEN
            old_root => tree%b%root
            old_root%parent => new_root
            new_root%subtrees(1)%node => old_root
            old_root%parent => new_root
         END IF
         tree%b%root => new_root
      END SUBROUTINE btree_new_root_i8_zp2d
# 422 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_new_node_i8_zp2d (tree, node)
# 424 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(INOUT) :: tree
# 425 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 426 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: i
         !
         ALLOCATE (node)
         ALLOCATE (node%keys(tree%b%max_fill))
         ALLOCATE (node%values(tree%b%max_fill))
         ALLOCATE (node%subtrees(tree%b%max_fill + 1))
         DO i = 1, tree%b%max_fill + 1
            NULLIFY (node%subtrees(i)%node)
         END DO
         node%filled = 0
         NULLIFY (node%parent)
         tree%b%lastid = tree%b%lastid + 1
         node%id = tree%b%lastid
      END SUBROUTINE btree_new_node_i8_zp2d
# 440 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_free_node_i8_zp2d (node)
# 442 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 443 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         !
         DEALLOCATE (node%keys)
         DEALLOCATE (node%values)
         DEALLOCATE (node%subtrees)
         DEALLOCATE (node)
      END SUBROUTINE btree_free_node_i8_zp2d
# 449 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_find_i8_zp2d (tree, key, value, exists)
# 451 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(IN) :: tree
# 452 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 453 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_data_zp2d), INTENT(OUT) :: value
# 454 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         LOGICAL, INTENT(OUT), OPTIONAL :: exists
         !
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 457 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER :: position
         !
         CALL btree_find_full_i8_zp2d (tree, key, node, position, short=.TRUE.)
# 460 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         IF (PRESENT(exists)) THEN
            exists = position .GT. 0
         END IF
         IF (position .GT. 0) THEN
            value = node%values(position)
         END IF
      END SUBROUTINE btree_find_i8_zp2d
# 467 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_ge_pos_i8_zp2d (keys, key, position, filled)
# 469 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 470 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 471 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GE. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LT. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GE. key .AND. keys(position - 1) .LT. key) THEN
               RETURN
            END IF
            IF (keys(position) .GE. key) right = MIN(position, right - 1)
            IF (keys(position) .LT. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_ge_pos_i8_zp2d
# 495 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gt_pos_i8_zp2d (keys, key, position, filled)
# 497 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 498 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 499 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER :: left, right
         !
         IF (keys(1) .GT. key) THEN
            position = 1
            RETURN
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = 2
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gt_pos_i8_zp2d
# 523 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      SUBROUTINE btree_node_find_gte_pos_i8_zp2d (keys, key, position, filled, first)
# 525 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), DIMENSION(:) :: keys
# 526 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 527 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(IN) :: filled
         INTEGER, INTENT(IN), OPTIONAL :: first
         INTEGER :: left, right, one
         !
         one = 1
         IF (PRESENT(FIRST)) one = first
         IF (one .LE. filled) THEN
            IF (keys(one) .GT. key) THEN
               position = one
               RETURN
            END IF
         END IF
         IF (keys(filled) .LE. key) THEN
            position = filled + 1
            RETURN
         END IF
         left = one + 1
         right = filled
         position = MAX(ISHFT(left + right, -1), left)
         DO WHILE (left .LE. right)
            IF (keys(position) .GT. key .AND. keys(position - 1) .LE. key) THEN
               RETURN
            END IF
            IF (keys(position) .GT. key) right = MIN(position, right - 1)
            IF (keys(position) .LE. key) left = MAX(position, left + 1)
            position = MAX(ISHFT(left + right, -1), left)
         END DO
      END SUBROUTINE btree_node_find_gte_pos_i8_zp2d
# 556 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_full_i8_zp2d (tree, key, node, position, ge_position, short)
# 561 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(IN) :: tree
# 562 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 563 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 564 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: position
         INTEGER, INTENT(OUT), OPTIONAL :: ge_position
         LOGICAL, INTENT(IN), OPTIONAL :: short
         INTEGER :: gti ! Used mark searches
         LOGICAL :: stop_short
         !
         stop_short = .FALSE.
         IF (PRESENT(short)) stop_short = short
         NULLIFY (node)
         position = 0
         IF (PRESENT(ge_position)) ge_position = 0
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            CALL btree_node_find_ge_pos_i8_zp2d (node%keys, key, position, node%filled)
# 585 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            IF (stop_short .AND. position .LE. node%filled) THEN
               IF (node%keys(position) .EQ. key) THEN
                  IF (PRESENT(ge_position)) ge_position = position
                  RETURN
               END IF
            END IF
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            !CALL btree_node_find_gt_pos_i8_zp2d (node%keys, key, gti, node%filled, position)
# 601 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            CALL btree_node_find_gte_pos_i8_zp2d (node%keys, key, gti, node%filled, position)
# 602 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               IF (PRESENT(ge_position)) ge_position = gti
               position = 0
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_full_i8_zp2d
# 611 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

      ! node is unassociated and position=0 if not found
      ! Precondition: The key is tree or its subtree.

      SUBROUTINE btree_find_leaf_i8_zp2d (tree, key, node, gti)
# 616 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_i8_zp2d), INTENT(IN) :: tree
# 617 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER(KIND=keyt), INTENT(IN) :: key
# 618 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         TYPE(btree_node_i8_zp2d), POINTER :: node
# 619 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
         INTEGER, INTENT(OUT) :: gti
         !
         NULLIFY (node)
         !IF (tree%b%n .EQ. 0) RETURN
         IF (.NOT. ASSOCIATED(tree%b%root)) RETURN
         gti = 1
         ! Try to find the key in the given node. If it's found, then
         ! return the node.
         node => tree%b%root
         descent: DO WHILE (.TRUE.)
            ! Try to find the first element equal to or greater than the
            ! one we're searching for.
            !CALL btree_node_find_ge_pos_i8_zp2d (node%keys, key, position, node%filled)
# 632 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            ! One of three things is now true about position: it's now
            ! greater than the number of keys (if all keys are smaller), or
            ! it points to the key that is equal to or greater than the one
            ! we are searching for. If it is found and we are just
            ! searching for one equal element (i.e., user search), we can
            ! return.
            !
            ! If the key is not found, then either return the GE position
            ! if we're in a leaf (case 2 here), otherwise descend into the
            ! subtrees.
            CALL btree_node_find_gt_pos_i8_zp2d (node%keys, key, gti, node%filled)
# 643 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !CALL btree_node_find_gt2_pos_i8_zp2d (node%keys, key, i, node%filled)
# 644 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
            !IF (i .NE. gti) WRITE(*,*)'XXXX difference',i,gti
            IF (ASSOCIATED(node%subtrees(1)%node)) THEN
               node => node%subtrees(gti)%node
            ELSE
               RETURN
            END IF
         END DO descent
      END SUBROUTINE btree_find_leaf_i8_zp2d
# 652 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"
# 653 "/__w/dbcsr/dbcsr/src/utils/dbcsr_btree.F"

END MODULE dbcsr_btree
