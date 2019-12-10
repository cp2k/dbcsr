# libsmm_acc/kernels

## Directory Organization

* [`autotuning_properties.json`](autotuning_properties.json) Properties of the autotuning procedure, read from [DBCSR source code](../libsmm_acc_benchmark.cpp)

* [`smm_acc_common.h`](smm_acc_common.h) Functionalities common to kernel CUDA/HIP codes

* [`smm_acc_dnt_base.py`](smm_acc_dnt_base.py) Kernel base class

  * `smm_acc_dnt_ALGORITHM.py` Kernel class

  * `smm_acc_dnt_ALGORITHM.h` Kernel CUDA/HIP code

* [`smm_acc_predict.py`](smm_acc_predict.py) Class and helper functions for parameter prediction procedure

* [`smm_acc_transpose.h`](smm_acc_transpose.h) Transposition CUDA/HIP code

* [`gpu_properties.json`](gpu_properties.json) GPU card properties
