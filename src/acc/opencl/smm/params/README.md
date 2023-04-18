# Tuned Parameters

The OpenCL based implementation of LIBSMM supports default kernel-parameters, i.e., kernels can be successfuly generated for every requested multiplication/matrix shape (M, N, K) within the definition of a "Small Matrix Multiplication" (maximum M, N, and K).

Tuned parameters targeting different devices can co-exist and can be embeded into the same executable, i.e., the executable does not depend on a particular build-path or location of parameter-files.

Parameters are selected by matching against a device-ID with fallback to the "best-matching" parameters. The device-ID can be based on a vendor-specific function to identify a certain device or is generated from device's name as exposed by the OpenCL API.

Parameters can be loaded from a CSV-file at runtime (`OPENCL_LIBSMM_SMM_PARAMS` environment variable) and thereby disable matching devices, i.e., parameters loaded this way will take precedence.
