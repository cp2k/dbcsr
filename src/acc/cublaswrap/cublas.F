MODULE cublas
   USE ISO_C_BINDING, ONLY: C_INT, &
      C_PTR, &
      C_DOUBLE

   IMPLICIT NONE


#if defined(__ACC_CUBLAS)
   PUBLIC :: acc_dgemm_cu
#endif

    INTERFACE

#if defined(__ACC_CUBLAS)
      FUNCTION acc_dgemm_cu(handle_ptr, stack_params, stack_size, mnk, transa, transb,&
                            a_data, b_data, c_data, alpha, beta, stream_ptr) &
         RESULT(istat) &
         BIND(C, name="acc_dgemm")
         IMPORT
         type(C_PTR),intent(in),value             :: handle_ptr
         TYPE(C_PTR), intent(in), value           :: stack_params
         INTEGER(KIND=C_INT), INTENT(IN), VALUE   :: stack_size, mnk, transa, transb
         TYPE(C_PTR), INTENT(IN), VALUE           :: a_data, b_data, c_data
         REAL(KIND=C_DOUBLE), intent(in),value    :: alpha, beta
         TYPE(C_PTR), intent(in), VALUE           :: stream_ptr
         INTEGER(KIND=C_INT)                      :: istat
      END FUNCTION acc_dgemm_cu



! cublas innterface ------------

      integer(C_INT) function cudaMalloc(ptr, bytes) BIND(C, NAME='cudaMalloc')
        use ISO_C_BINDING
        type(C_PTR) :: ptr
        integer(C_SIZE_T), value :: bytes
      end function
 
    integer(C_INT) function cudaMemcpy(dst, src, count, kind) BIND(C, NAME='cudaMemcpy')
        use ISO_C_BINDING
        type(C_PTR), value :: dst
        type(C_PTR), value :: src
        integer(C_SIZE_T), value :: count
        integer(C_INT), value :: kind
    end function
 
    integer(C_INT) function cublasCreate(handle_ptr) BIND(C, NAME='f_cublasCreate')
        use ISO_C_BINDING
        type(C_PTR) :: handle_ptr
    end function
 
    subroutine  cublasDestroy(handle_ptr) BIND(C, NAME='f_cublasDestroy')
        use ISO_C_BINDING
        type(C_PTR), value :: handle_ptr
    end subroutine
 
    subroutine cudaFree(ptr) BIND(C, NAME='cudaFree')
        use ISO_C_BINDING
        type(C_PTR), value :: ptr
    end subroutine
 
    integer(C_INT) function cublasSetMatrix(rows, cols, elemSize, a_ptr, &
                                           lda, b_ptr, ldb) &
                                           BIND(C, NAME='cublasSetMatrix')
        use ISO_C_BINDING
        integer(C_INT), value :: rows
        integer(C_INT), value :: cols
        integer(C_INT), value :: elemSize
        type(C_PTR),    value :: a_ptr
        integer(C_INT), value :: lda
        type(C_PTR),    value :: b_ptr
        integer(C_INT), value :: ldb
    end function
 
    integer(C_INT) function cublasGetMatrix(rows, cols, elemSize, a_ptr, &
                                            lda, b_ptr, ldb ) &
                                            BIND(C, NAME='cublasGetMatrix')
        use ISO_C_BINDING
        integer(C_INT), value :: rows
        integer(C_INT), value :: cols
        integer(C_INT), value :: elemSize
        type(C_PTR),    value :: a_ptr
        integer(C_INT), value :: lda
        type(C_PTR),    value :: b_ptr
        integer(C_INT), value :: ldb
 
    end function
 
    integer(C_INT) function cublasDgemm(handle, transa, transb, m, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc) &
                                        BIND(C, NAME='f_cublasDgemm')
        use ISO_C_BINDING
        type(C_PTR), value    :: handle
        integer(C_INT), value :: transa
        integer(C_INT), value :: transb
        integer(C_INT), value :: m
        integer(C_INT), value :: n
        integer(C_INT), value :: k
        real(C_DOUBLE)        :: alpha
        type(C_PTR), value    :: A
        integer(C_INT), value :: lda
        type(C_PTR), value    :: B
        integer(C_INT), value :: ldb
        real(C_DOUBLE)        :: beta
        type(C_PTR), value    :: C
        integer(C_INT), value :: ldc
    end function
 
    integer(C_INT) function cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, &
                                        A, lda, B, ldb, beta, C, ldc, batch_count) &
                                        BIND(C, NAME='f_cublasDgemmBatched')
        use ISO_C_BINDING
        type(C_PTR), value    :: handle
        integer(C_INT), value :: transa
        integer(C_INT), value :: transb
        integer(C_INT), value :: m
        integer(C_INT), value :: n
        integer(C_INT), value :: k
        real(C_DOUBLE)        :: alpha
        type(C_PTR), value    :: A
        integer(C_INT), value :: lda
        type(C_PTR), value    :: B
        integer(C_INT), value :: ldb
        real(C_DOUBLE)        :: beta
        type(C_PTR), value    :: C
        integer(C_INT), value :: ldc
        integer(C_INT), value :: batch_count
    end function
 
    integer(C_INT) function cudaStreamCreate(stream_ptr) BIND(C, NAME='f_cudaStreamCreate')
        use ISO_C_BINDING
        type(C_PTR) :: stream_ptr
    end function
 
    integer(C_INT) function cublasSetStream(handle, stream) BIND(C, NAME='f_cublasSetStream')
        use ISO_C_BINDING
        type(C_PTR), value :: handle
        type(C_PTR), value :: stream
    end function
 
    integer(C_INT) function cudaStreamDestroy(stream) BIND(C, NAME='f_cudaStreamDestroy')
        use ISO_C_BINDING
        type(C_PTR), value :: stream
    end function
 
    subroutine cudaDeviceSynchronize() BIND(C, NAME='cudaDeviceSynchronize')
    end subroutine


#endif

   END INTERFACE



END MODULE cublas
