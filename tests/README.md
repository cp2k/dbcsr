!--------------------------------------------------------------------------------------------------!
! Copyright (C) by the DBCSR developers group - All rights reserved                                !
! This file is part of the DBCSR library.                                                          !
!                                                                                                  !
! For information on the license, see the LICENSE file.                                            !
! For further information please visit https://dbcsr.cp2k.org                                      !
! SPDX-License-Identifier: GPL-2.0+                                                                !
!--------------------------------------------------------------------------------------------------!

# DBCSR Testing and Performance

## Correctness tests

- [dbcsr_tas_unittest](dbcsr_tas_unittest.F) : unit test for tall-and-skinny matrices
- [dbcsr_tensor_test](dbcsr_tensor_test.cpp) : test the tensor contraction (13|2)x(54|21)=(3|45) 31 and other functions
- [dbcsr_tensor_unittest](dbcsr_tensor_unittest.F) : unit test for tensor functionalities
- [dbcsr_test_csr_conversions](dbcsr_test_csr_conversions.F) : test DBCSR to CSR conversion with random matrices
- [dbcsr_unittest_1](dbcsr_unittest1.F) : test matrix operations: add, multiply and multiply-ghost
- [dbcsr_unittest_2](dbcsr_unittest2.F) : test matrix-multiply with large blocks (block size=100) and rectangular matrices (block size=5)

### GPU-backend correctness tests:

- [dbcsr_unittest_3](dbcsr_unittest3.F) : test matrix-multiply with various block sizes that are run by the libsmm_acc GPU backend if DBCSR is compiled with GPU support
- [libsmm_acc_unittest_multiply](libsmm_acc_unittest_multiply.cpp.template) : tests all libsmm_acc transpose kernels
- [libsmm_acc_unittest_transpose](libsmm_acc_unittest_transpose.cpp) : tests all libsmm_acc batch-multiplication kernels

## Performance tests

DBCSR performance tests:

- [dbcsr_performance_driver](dbcsr_performance_driver.F) : performance tester for matrix operations. The input matrices can be described in an input file in order to test different     configurations. See below.

### GPU backend performance tests:

- [libsmm_acc_timer_multiply](libsmm_acc_timer_multiply.cpp.template) : time all libsmm_acc batch-multiplication kernels

---

See the [tests' documentation](../docs/guide/2-user-guide/2-tests/index.md).
