!
! generate a benchmark for the following cases
!
! 1) tiny_gemm
! 2) matmul
! 3) dgemm
! 4) multrec 1
! 5) multrec 2
! 6) multrec 3
! 7) multrec 4
! 8) multvec
! 9) libxsmm
!
PROGRAM small_gen
   USE mults
   USE multrec_gen
   IMPLICIT NONE

   CHARACTER(LEN=1024) :: arg, filename, label
   INTEGER :: M, N, K, transpose_flavor, data_type, SIMD_size
   INTEGER :: ibest_square = 3, best_square = 4
   INTEGER :: isquare
   LOGICAL :: do_libxsmm

   CALL GET_COMMAND_ARGUMENT(1, arg)
   READ (arg, *) M
   CALL GET_COMMAND_ARGUMENT(2, arg)
   READ (arg, *) N
   CALL GET_COMMAND_ARGUMENT(3, arg)
   READ (arg, *) K
   CALL GET_COMMAND_ARGUMENT(4, arg)
   READ (arg, *) transpose_flavor
   CALL GET_COMMAND_ARGUMENT(5, arg)
   READ (arg, *) data_type
   CALL GET_COMMAND_ARGUMENT(6, arg)
   READ (arg, *) SIMD_size
   CALL GET_COMMAND_ARGUMENT(7, filename)

   IF (COMMAND_ARGUMENT_COUNT() .gt. 7) THEN
      do_libxsmm = .TRUE.
   ELSE
      do_libxsmm = .FALSE.
   END IF

   ! generation of the tiny version
   write (label, '(A,I0)') "_", 1
   CALL mult_versions(M, N, K, 1, label, transpose_flavor, data_type, SIMD_size, filename)

   ! generation of the matmul version
   write (label, '(A,I0)') "_", 2
   CALL mult_versions(M, N, K, 2, label, transpose_flavor, data_type, SIMD_size, filename)

   ! generation of the dgemm version
   write (label, '(A,I0)') "_", 3
   CALL mult_versions(M, N, K, 3, label, transpose_flavor, data_type, SIMD_size, filename)

   ! generation of the multrec versions (4)
   DO isquare = ibest_square + 1, ibest_square + best_square
      write (label, '(A,I0)') "_", isquare
      CALL mult_versions(M, N, K, isquare, label, transpose_flavor, data_type, SIMD_size, filename)
   ENDDO

   ! generation of the vector version,
   ! only in the case of SIMD_size=32(i.e. AVX/AVX2) and SIMD_size=64(i.e. KNC/AVX512)
   ibest_square = ibest_square + 1
   IF ((SIMD_size == 32 .OR. SIMD_size == 64) .AND. transpose_flavor == 1 .AND. data_type <= 2) THEN
      write (label, '(A,I0)') "_", ibest_square + best_square
      CALL mult_versions(M, N, K, ibest_square + best_square, label, &
                         transpose_flavor, data_type, SIMD_size, filename, stack_size_label="")
   ENDIF

   ! generation of the libxsmm version interface
   ibest_square = ibest_square + 1
   IF (do_libxsmm .AND. transpose_flavor == 1 .AND. data_type <= 2) THEN
      write (label, '(A,I0)') "_", ibest_square + best_square
      CALL mult_versions(M, N, K, ibest_square + best_square, label, &
                         transpose_flavor, data_type, SIMD_size, filename, stack_size_label="")
   END IF

   write (6, '(A)') "#ifdef __INTEL_OFFLOAD"
   write (6, '(A)') "!dir$ attributes offload:mic :: run_kernels"
   write (6, '(A)') "#endif"
   write (6, '(A,I0,A,I0,A,I0,A)') "SUBROUTINE small_find_", M, "_", N, "_", K, "(unit)"
   write (6, '(A)') "  IMPLICIT NONE"
   write (6, '(A)') "  INTEGER :: unit ! Output unit"
   write (6, '(A,I0,A,I0,A,I0,A,I0)') "  INTEGER, PARAMETER :: M=", M, ",N=", N, ",K=", K
   write (6, '(A)') "  CHARACTER(len=64) :: filename"
   CALL write_matrix_defs(M, N, K, transpose_flavor, data_type, .FALSE., .FALSE., padding=.TRUE.)
   write (6, '(A)') "  INTERFACE"
   write (6, '(A)') "    SUBROUTINE X("//TRIM(trparam())//")"
   CALL write_matrix_defs(data_type=data_type, write_intent=.TRUE., write_target=.FALSE.)
   write (6, '(A)') "    END SUBROUTINE"
   write (6, '(A)') "  END INTERFACE"
   DO isquare = 1, ibest_square + best_square
      IF (isquare == 8 .AND. ((SIMD_size /= 32 .AND. SIMD_size /= 64) .OR. transpose_flavor /= 1 .OR. data_type > 2)) CYCLE
      IF (isquare == 9 .AND. ((.NOT. do_libxsmm) .OR. transpose_flavor /= 1 .OR. data_type > 2)) CYCLE
      write (6, '(A,I0,A,I0,A,I0,A,I0)') "PROCEDURE(X) :: smm_"//trstr(transpose_flavor, data_type)//"_", &
         M, "_", N, "_", K, "_", isquare
   ENDDO
   write (6, '(A,I0,A,I0)') "  INTEGER, PARAMETER :: Nmin=5,Nk=1,Nloop=", ibest_square + best_square
   write (6, '(A)') "  TYPE t_kernels"
   write (6, '(A)') "    PROCEDURE(X), POINTER, NOPASS :: ptr"
   write (6, '(A)') "  END TYPE t_kernels"
   write (6, '(A)') "  TYPE(t_kernels) :: kernels(Nk,Nloop)"
   write (6, '(A)') "  INTEGER :: mnk(3,Nk) ! mu, nu, ku"
   DO isquare = 1, ibest_square + best_square
      IF ((isquare == 8 .AND. ((SIMD_size /= 32 .AND. SIMD_size /= 64) .OR. transpose_flavor /= 1 .OR. data_type > 2)) .OR. &
          (isquare == 9 .AND. ((.NOT. do_libxsmm) .OR. transpose_flavor /= 1 .OR. data_type > 2))) THEN
         write (6, '(A,I0,A)') "  kernels(Nk,", isquare, ")%ptr => Null()"
      ELSE
       write (6, '(A,I0,A,I0,A,I0,A,I0,A,I0)') &
            "  kernels(Nk,", isquare, ")%ptr => smm_"//trstr(transpose_flavor, data_type)//"_", &
            M, "_", N, "_", K, "_", isquare
      ENDIF
   ENDDO
   write (6, '(A,I0,A,I0,A,I0,A)') "  filename='small_find_", M, "_", N, "_", K, ".out'"
   write (6, '(A)') "  C = 0 ; A = 0 ; B = 0"
   write (6, '(A)') "  mnk=0"
   write (6, '(A)') "  CALL run_kernels(filename,unit,M,N,K,A,B,C,Nmin,Nk,Nloop,kernels,mnk)"
   write (6, '(A,I0,A,I0,A,I0)') "END SUBROUTINE small_find_", M, "_", N, "_", K

END PROGRAM small_gen
