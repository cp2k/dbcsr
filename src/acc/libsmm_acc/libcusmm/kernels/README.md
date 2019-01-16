# libcusmm/kernels

## Directory Organization

* [`autotuning_properties.json`](autotuning_properties.json) Properties of the autotuning procedure, read from [DBCSR source code](../libcusmm_benchmark.cu)

* [`cusmm_common.h`](cusmm_common.h) Functionalities common to kernel CUDA codes

* [`cusmm_dnt_base.py`](cusmm_dnt_base.py) Kernel base class

  * `cusmm_dnt_ALGORITHM.py` Kernel class

  * `cusmm_dnt_ALGORITHM.h` Kernel CUDA code

* [`cusmm_predict.py`](cusmm_predict.py) Class and helper functions for parameter prediction procedure

* [`cusmm_transpose.h`](cusmm_transpose.h) Transposition CUDA code

* [`gpu_properties.json`](gpu_properties.json) GPU card properties
