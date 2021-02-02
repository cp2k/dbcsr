/*------------------------------------------------------------------------------------------------*
 * Copyright (C) by the DBCSR developers group - All rights reserved                              *
 * This file is part of the DBCSR library.                                                        *
 *                                                                                                *
 * For information on the license, see the LICENSE file.                                          *
 * For further information please visit https://dbcsr.cp2k.org                                    *
 * SPDX-License-Identifier: GPL-2.0+                                                              *
 *------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#include "acc_opencl.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <limits.h>
#include <ctype.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(_WIN32)
# include <windows.h>
#else
# include <glob.h>
#endif
#if defined(__DBCSR_ACC)
# include "../acc_libsmm.h"
#endif

#if !defined(ACC_OPENCL_EXTLINE)
# define ACC_OPENCL_EXTLINE
#endif
#if !defined(ACC_OPENCL_DELIMS)
# define ACC_OPENCL_DELIMS " \t;,:"
#endif


#if defined(__cplusplus)
extern "C" {
#endif

acc_opencl_options_t acc_opencl_options;
int acc_opencl_ndevices;
cl_device_id acc_opencl_devices[ACC_OPENCL_DEVICES_MAXCOUNT];
cl_context acc_opencl_context;

#if !defined(NDEBUG)
void acc_opencl_notify(const char* /*errinfo*/, const void* /*private_info*/, size_t /*cb*/, void* /*user_data*/);
void acc_opencl_notify(const char* errinfo, const void* private_info, size_t cb, void* user_data)
{
  ACC_OPENCL_UNUSED(private_info); ACC_OPENCL_UNUSED(cb); ACC_OPENCL_UNUSED(user_data);
  fprintf(stderr, "ERROR ACC/OpenCL: %s\n", errinfo);
}
#endif


const char* acc_opencl_stristr(const char* a, const char* b)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    do {
      if (tolower(*a) != tolower(*b)) {
        ++a;
      }
      else {
        const char* c = b;
        result = a;
        while ('\0' != *++a && '\0' != *++c) {
          if (tolower(*a) != tolower(*c)) {
            result = NULL;
            break;
          }
        }
        if ('\0' != c[0] && '\0' != c[1]) {
          result = NULL;
        }
        else break;
      }
    } while ('\0' != *a);
  }
  return result;
}


/* comparator used with qsort; stabilized by tail condition (a < b ? -1 : 1) */
int acc_opencl_order_devices(const void* /*dev_a*/, const void* /*dev_b*/);
int acc_opencl_order_devices(const void* dev_a, const void* dev_b)
{
  const cl_device_id *const a = (const cl_device_id*)dev_a;
  const cl_device_id *const b = (const cl_device_id*)dev_b;
  cl_device_type type_a, type_b;
  assert(NULL != a && NULL != b && a != b);
  ACC_OPENCL_EXPECT(EXIT_SUCCESS, clGetDeviceInfo(*a,
    CL_DEVICE_TYPE, sizeof(cl_device_type), &type_a, NULL));
  ACC_OPENCL_EXPECT(EXIT_SUCCESS, clGetDeviceInfo(*b,
    CL_DEVICE_TYPE, sizeof(cl_device_type), &type_b, NULL));
  if (CL_DEVICE_TYPE_DEFAULT & type_a) return -1;
  else if (CL_DEVICE_TYPE_DEFAULT & type_b) return 1;
  else {
    if (CL_DEVICE_TYPE_GPU & type_a) {
      if (CL_DEVICE_TYPE_GPU & type_b) {
        size_t size_a, size_b;
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, acc_opencl_info_devmem(*a, NULL, &size_a));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, acc_opencl_info_devmem(*b, NULL, &size_b));
        return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
      }
      else return -1;
    }
    else if (CL_DEVICE_TYPE_GPU & type_b) return 1;
    else {
      if (CL_DEVICE_TYPE_ACCELERATOR & type_a) {
        if (CL_DEVICE_TYPE_ACCELERATOR & type_b) {
          size_t size_a, size_b;
          ACC_OPENCL_EXPECT(EXIT_SUCCESS, acc_opencl_info_devmem(*a, NULL, &size_a));
          ACC_OPENCL_EXPECT(EXIT_SUCCESS, acc_opencl_info_devmem(*b, NULL, &size_b));
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        else return -1;
      }
      else if (CL_DEVICE_TYPE_ACCELERATOR & type_b) return 1;
      else {
        size_t size_a, size_b;
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, acc_opencl_info_devmem(*a, NULL, &size_a));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, acc_opencl_info_devmem(*b, NULL, &size_b));
        return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
      }
    }
  }
}


int acc_init(void)
{
#if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = (0 == omp_in_parallel() ? EXIT_SUCCESS : EXIT_FAILURE);
#else
  int result = EXIT_SUCCESS;
#endif
  if (0 == acc_opencl_ndevices) { /* avoid to initialize multiple times */
    const char *const disable = getenv("ACC_OPENCL_DISABLE");
    if (NULL == disable || '0' == *disable) {
      cl_platform_id platforms[ACC_OPENCL_DEVICES_MAXCOUNT];
      char buffer[ACC_OPENCL_BUFFERSIZE];
      const char *const env_device_vendor = getenv("ACC_OPENCL_VENDOR");
      const char *const env_device_type = getenv("ACC_OPENCL_DEVTYPE");
      const char *const env_device_id = getenv("ACC_OPENCL_DEVICE");
      int device_id = (NULL == env_device_id ? 0 : atoi(env_device_id));
      cl_uint nplatforms = 0, ndevices = 0, i;
      cl_device_type type = CL_DEVICE_TYPE_ALL;
      ACC_OPENCL_CHECK(clGetPlatformIDs(0, NULL, &nplatforms),
        "query number of platforms", result);
      ACC_OPENCL_CHECK(clGetPlatformIDs(
        nplatforms <= ACC_OPENCL_DEVICES_MAXCOUNT ? nplatforms : ACC_OPENCL_DEVICES_MAXCOUNT,
        platforms, 0), "retrieve platform ids", result);
      if (NULL != env_device_type && '\0' != *env_device_type) {
        if (NULL != acc_opencl_stristr(env_device_type, "gpu")) type = CL_DEVICE_TYPE_GPU;
        else if (NULL != acc_opencl_stristr(env_device_type, "cpu")) type = CL_DEVICE_TYPE_CPU;
        else type = CL_DEVICE_TYPE_ACCELERATOR;
      }
      acc_opencl_ndevices = 0;
      for (i = 0; i < nplatforms; ++i) {
        if (EXIT_SUCCESS == result
          && CL_SUCCESS == clGetDeviceIDs(platforms[i], type, 0, NULL, &ndevices))
        {
          const int n = (acc_opencl_ndevices + ndevices) < ACC_OPENCL_DEVICES_MAXCOUNT
            ? (int)ndevices : (ACC_OPENCL_DEVICES_MAXCOUNT - acc_opencl_ndevices);
          if (CL_SUCCESS == clGetDeviceIDs(platforms[i], type,
            n, acc_opencl_devices + acc_opencl_ndevices, NULL))
          {
            acc_opencl_ndevices += n;
          }
          else {
            ACC_OPENCL_ERROR("retrieve device ids", result);
          }
        }
      }
      assert(NULL == acc_opencl_context);
      if (device_id < acc_opencl_ndevices) {
        if (NULL != env_device_vendor && '\0' != *env_device_vendor) {
          for (i = 0; i < (cl_uint)acc_opencl_ndevices;) {
            buffer[0] = '\0';
            if (CL_SUCCESS == clGetDeviceInfo(acc_opencl_devices[i],
              CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL))
            {
              if (NULL == acc_opencl_stristr(buffer, env_device_vendor)) {
                --acc_opencl_ndevices;
                if (i < (cl_uint)acc_opencl_ndevices) { /* keep relative order of IDs */
                  memmove(acc_opencl_devices + i, acc_opencl_devices + i + 1,
                    sizeof(cl_device_id) * (acc_opencl_ndevices - i));
                }
              }
              else ++i;
            }
            else {
              ACC_OPENCL_ERROR("retrieve device vendor", result);
              break;
            }
          }
        }
      }
      if (device_id < acc_opencl_ndevices) {
        if (EXIT_SUCCESS == result && 1 < acc_opencl_ndevices) {
          /* reorder devices according to acc_opencl_order_devices */
          qsort(acc_opencl_devices, acc_opencl_ndevices,
            sizeof(cl_device_id), acc_opencl_order_devices);
          /* preselect default device */
          if (NULL == env_device_id || '\0' == *env_device_id) {
            for (i = 0; i < (cl_uint)acc_opencl_ndevices; ++i) {
              ACC_OPENCL_CHECK(clGetDeviceInfo(acc_opencl_devices[i],
                CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL),
                "retrieve device type", result);
              if (CL_DEVICE_TYPE_DEFAULT & type) {
                device_id = (int)i;
                break;
              }
            }
          }
        }
        if (EXIT_SUCCESS == result) {
          cl_device_id active_device;
          result = acc_opencl_set_active_device(device_id, &active_device);
#if defined(_OPENMP) && defined(ACC_OPENCL_THREADLOCAL_CONTEXT)
          if (EXIT_SUCCESS == result) {
            const cl_context context = acc_opencl_context;
#           pragma omp parallel
            if (context != acc_opencl_context) {
              if (CL_SUCCESS == clRetainContext(context)) {
                acc_opencl_context = context;
              }
              else {
                ACC_OPENCL_ERROR("retain context", result);
                acc_opencl_context = NULL;
              }
            }
          }
#endif
#if defined(ACC_OPENCL_MEM_ASYNC)
          if (EXIT_SUCCESS == result) {
            const char *const env = getenv("ACC_OPENCL_ASYNC_MEMOPS");
            if (NULL == env) {
              const int confirmation = acc_opencl_device_vendor(active_device, "nvidia");
              acc_opencl_options.async_memops = (EXIT_SUCCESS != confirmation);
            }
            else acc_opencl_options.async_memops = (0 != atoi(env));
          }
          else
#endif
          acc_opencl_options.async_memops = CL_FALSE;
#if defined(ACC_OPENCL_SVM)
          if (EXIT_SUCCESS == result) {
            const char *const env = getenv("ACC_OPENCL_SVM");
            int level_major = 0;
            acc_opencl_options.svm_interop = (NULL == env || 0 != atoi(env)) &&
              (EXIT_SUCCESS == acc_opencl_device_level(active_device,
                &level_major, NULL/*level_minor*/) && 2 <= level_major);
          }
          else
#endif
          acc_opencl_options.svm_interop = CL_FALSE;
        }
      }
      else { /* mark as initialized */
        acc_opencl_ndevices = -1;
      }
    }
    else { /* mark as initialized */
      acc_opencl_ndevices = -1;
    }
#if defined(__DBCSR_ACC)
    /* DBCSR shall call acc_init as well as libsmm_acc_init (since both interfaces are used).
     * Also, libsmm_acc_init may privately call acc_init (as it depends on the ACC interface).
     * The implementation of acc_init should hence be safe against "over initialization".
     * However, DBCSR only calls acc_init (and expects an implicit libsmm_acc_init).
     */
    if (EXIT_SUCCESS == result) {
      result = libsmm_acc_init();
    }
#endif
  }
  ACC_OPENCL_RETURN(result);
}


int acc_finalize(void)
{
#if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = (0 == omp_in_parallel() ? EXIT_SUCCESS : EXIT_FAILURE);
#else
  int result = EXIT_SUCCESS;
#endif
  if (NULL != acc_opencl_context) {
    const cl_context context = acc_opencl_context;
    assert(0 < acc_opencl_ndevices);
#if defined(_OPENMP) && defined(ACC_OPENCL_THREADLOCAL_CONTEXT)
#   pragma omp parallel
    if (context != acc_opencl_context) {
      ACC_OPENCL_CHECK(clReleaseContext(acc_opencl_context),
        "release context", result);
      acc_opencl_context = NULL;
    }
#endif
    ACC_OPENCL_CHECK(clReleaseContext(context),
      "release context", result);
    acc_opencl_context = NULL;
#if defined(__DBCSR_ACC)
    /* DBCSR may call acc_init() as well as libsmm_acc_init() since both interface are used.
     * libsmm_acc_init may privately call acc_init (as it depends on the ACC interface).
     * The implementation of acc_init() should be safe against "over initialization".
     * However, DBCSR only calls acc_init() and expects an implicit libsmm_acc_init().
     */
    if (EXIT_SUCCESS == result) {
      result = libsmm_acc_finalize();
    }
#endif
  }
  ACC_OPENCL_RETURN(result);
}


void acc_clear_errors(void)
{
}


int acc_get_ndevices(int* ndevices)
{
  int result;

#if defined(__DBCSR_ACC)
  /* DBCSR calls acc_get_ndevices before calling acc_init(). */
  result = acc_init();
  if (EXIT_SUCCESS == result)
#endif
  {
    if (NULL != ndevices && 0 != acc_opencl_ndevices) {
      *ndevices = (0 < acc_opencl_ndevices ? acc_opencl_ndevices : 0);
      result = EXIT_SUCCESS;
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  ACC_OPENCL_RETURN(result);
}


int acc_opencl_device(void* stream, cl_device_id* device)
{
  int result = EXIT_SUCCESS;
  assert(NULL != device);
  if (NULL != stream) {
    ACC_OPENCL_CHECK(clGetCommandQueueInfo(*ACC_OPENCL_STREAM(stream), CL_QUEUE_DEVICE,
      sizeof(cl_device_id), device, NULL), "retrieve device from queue", result);
  }
  else if (NULL != acc_opencl_context) {
#if !defined(NDEBUG)
    size_t n = sizeof(cl_device_id);
    ACC_OPENCL_CHECK(clGetContextInfo(acc_opencl_context, CL_CONTEXT_DEVICES,
      sizeof(cl_device_id), device, &n), "retrieve id of active device", result);
#else
    ACC_OPENCL_CHECK(clGetContextInfo(acc_opencl_context, CL_CONTEXT_DEVICES,
      sizeof(cl_device_id), device, NULL), "retrieve id of active device", result);
#endif
    assert(EXIT_SUCCESS != result || sizeof(cl_device_id) == n/*single-device context*/);
  }
  else {
    *device = NULL;
  }
  ACC_OPENCL_RETURN(result);
}


int acc_opencl_device_vendor(cl_device_id device, const char* vendor)
{
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != vendor);
  buffer[0] = '\0';
  ACC_OPENCL_CHECK(clGetDeviceInfo(device,
    CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL),
    "retrieve device vendor", result);
  if (EXIT_SUCCESS == result) {
    return (NULL != acc_opencl_stristr(buffer, vendor)
      ? EXIT_SUCCESS
      : EXIT_FAILURE);
  }
  else ACC_OPENCL_RETURN(result);
}


int acc_opencl_device_level(cl_device_id device, int* level_major, int* level_minor)
{
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && (NULL != level_major || NULL != level_minor));
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_VERSION,
    ACC_OPENCL_BUFFERSIZE, buffer, NULL),
    "retrieve device level", result);
  if (EXIT_SUCCESS == result) {
    unsigned int level[2];
    /* input: "OpenCL <level_major>.<level_minor> ..." */
    if (2 == sscanf(buffer, "%*s %u.%u", level, level+1)) {
      if (NULL != level_major) *level_major = (int)level[0];
      if (NULL != level_minor) *level_minor = (int)level[1];
    }
    else {
      result = EXIT_SUCCESS;
    }
  }
  ACC_OPENCL_RETURN(result);
}


int acc_opencl_device_ext(cl_device_id device, const char *const extnames[], int num_exts)
{
  int result = ((NULL != extnames && 0 < num_exts) ? EXIT_SUCCESS : EXIT_FAILURE);
  char extensions[ACC_OPENCL_BUFFERSIZE], buffer[ACC_OPENCL_BUFFERSIZE];
  assert(NULL != device);
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
    ACC_OPENCL_BUFFERSIZE, extensions, NULL),
    "retrieve device extensions", result);
  if (EXIT_SUCCESS == result) {
    do {
      --num_exts;
      if (NULL == extnames[num_exts]) {
        return EXIT_FAILURE;
      }
      else {
        char *const exts = strncpy(buffer, extnames[num_exts], ACC_OPENCL_BUFFERSIZE - 1);
        const char* ext = strtok(exts, ACC_OPENCL_DELIMS);
        for (; NULL != ext; ext = strtok(NULL, ACC_OPENCL_DELIMS)) {
          if (NULL == strstr(extensions, ext)) {
            return EXIT_FAILURE;
          }
        }
      }
    } while (0 < num_exts);
  }
  ACC_OPENCL_RETURN(result);
}


int acc_opencl_set_active_device(int device_id, cl_device_id* device)
{
  cl_int result = (((0 <= device_id && device_id < acc_opencl_ndevices) ||
    /* allow successful completion if no device was found */
    0 > acc_opencl_ndevices) ? EXIT_SUCCESS : EXIT_FAILURE);
  if (0 < acc_opencl_ndevices) {
    const cl_device_id active_id = acc_opencl_devices[device_id];
    cl_device_id current_id = NULL;
    if (EXIT_SUCCESS == result) result = acc_opencl_device(NULL/*stream*/, &current_id);
    if (EXIT_SUCCESS == result && active_id != current_id) {
      if (NULL != acc_opencl_context) {
        ACC_OPENCL_CHECK(clReleaseContext(acc_opencl_context),
          "release context", result);
      }
      if (EXIT_SUCCESS == result) {
        cl_context_properties properties[] = {
          /* insert other properties in front of below property */
          CL_CONTEXT_INTEROP_USER_SYNC, CL_FALSE, /* TODO */
          0 /* end of properties */
        };
#if defined(NDEBUG)
        void (*const notify)(const char*, const void*, size_t, void*) = NULL;
#else
        void (*const notify)(const char*, const void*, size_t, void*) = acc_opencl_notify;
#endif
        acc_opencl_context = clCreateContext(properties,
          1/*num_devices*/, &active_id, notify, NULL/* user_data*/, &result);
        if (CL_INVALID_VALUE == result) { /* retry */
          const size_t n = sizeof(properties) / sizeof(*properties);
          assert(3 <= n);
          properties[n-3] = 0;
          acc_opencl_context = clCreateContext(0 != properties[0] ? properties : NULL,
            1/*num_devices*/, &active_id, notify, NULL/* user_data*/, &result);
        }
        ACC_OPENCL_CHECK(result, "create context", result);
      }
    }
    if (NULL != device) {
      *device = (EXIT_SUCCESS == result ? active_id : NULL);
    }
  }
  ACC_OPENCL_RETURN(result);
}


int acc_set_active_device(int device_id)
{
  ACC_OPENCL_RETURN(acc_opencl_set_active_device(device_id, NULL/*device*/));
}


int acc_opencl_wgsize(cl_device_id device, cl_kernel kernel,
  int* max_value, int* preferred_multiple)
{
  int result = (NULL != device && (NULL != preferred_multiple
                                || NULL != max_value))
    ? EXIT_SUCCESS : EXIT_FAILURE;
  if (NULL != kernel) { /* kernel-specific */
    if (NULL != max_value) {
      size_t value = 0;
      ACC_OPENCL_CHECK(clGetKernelWorkGroupInfo(kernel, device,
        CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &value, NULL),
        "query maximum WG-size of kernel", result);
      assert(value <= INT_MAX);
      *max_value = (int)value;
    }
    if (NULL != preferred_multiple) {
      size_t value = 0;
      ACC_OPENCL_CHECK(clGetKernelWorkGroupInfo(kernel, device,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(size_t), &value, NULL),
        "query preferred multiple of WG-size of kernel", result);
      assert(value <= INT_MAX);
      *preferred_multiple = (int)value;
    }
  }
  else { /* device-specific */
    if (NULL != max_value) {
      size_t value = 0;
      ACC_OPENCL_CHECK(clGetDeviceInfo(device,
        CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &value, NULL),
        "query maximum WG-size of device", result);
      assert(value <= INT_MAX);
      *max_value = (int)value;
    }
    if (NULL != preferred_multiple) {
#if defined(CL_VERSION_3_0)
      size_t value = 0;
      ACC_OPENCL_CHECK(clGetDeviceInfo(device,
        CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(size_t), &value, NULL),
        "query preferred multiple of WG-size of device", result);
      assert(value <= INT_MAX);
      *preferred_multiple = (int)value;
#else
      result = EXIT_FAILURE;
#endif
    }
  }
  ACC_OPENCL_RETURN(result);
}


int acc_opencl_kernel(const char* source, const char* build_options,
  const char* kernel_name, cl_kernel* kernel)
{
  char buffer[ACC_OPENCL_BUFFERSIZE] = "\0";
  cl_int result;
  assert(NULL != kernel);
  if (NULL != acc_opencl_context) {
    const cl_program program = clCreateProgramWithSource(
      acc_opencl_context, 1/*nlines*/, &source, NULL, &result);
    if (NULL != program) {
      cl_device_id active_id = NULL;
      assert(CL_SUCCESS == result);
      result = acc_opencl_device(NULL/*stream*/, &active_id);
      if (EXIT_SUCCESS == result) {
        result = clBuildProgram(program,
        1/*num_devices*/, &active_id, build_options,
        NULL/*callback*/, NULL/*user_data*/);
        if (CL_SUCCESS == result) {
          *kernel = clCreateKernel(program, kernel_name, &result);
          if (CL_SUCCESS == result) assert(NULL != *kernel);
          else {
            ACC_OPENCL_ERROR("create kernel", result);
          }
        }
        else {
          clGetProgramBuildInfo(program, active_id, CL_PROGRAM_BUILD_LOG,
            ACC_OPENCL_BUFFERSIZE, buffer, NULL); /* ignore retval */
          *kernel = NULL;
        }
      }
      else {
        *kernel = NULL;
      }
    }
    else {
      assert(CL_SUCCESS != result);
      ACC_OPENCL_ERROR("create program", result);
      *kernel = NULL;
    }
  }
  else {
    result = EXIT_FAILURE;
    *kernel = NULL;
  }
  ACC_OPENCL_RETURN_CAUSE(result, buffer);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
