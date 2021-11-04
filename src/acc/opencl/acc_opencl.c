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

#if !defined(ACC_OPENCL_CPPBIN)
# define ACC_OPENCL_CPPBIN "/usr/bin/cpp"
#endif
#if !defined(ACC_OPENCL_SEDBIN)
# define ACC_OPENCL_SEDBIN "/usr/bin/sed"
#endif
#if !defined(ACC_OPENCL_DELIMS)
# define ACC_OPENCL_DELIMS " \t;,:"
#endif


#if defined(__cplusplus)
extern "C" {
#endif

cl_device_id c_dbcsr_acc_opencl_devices[ACC_OPENCL_DEVICES_MAXCOUNT];
c_dbcsr_acc_opencl_config_t c_dbcsr_acc_opencl_config;
cl_context* c_dbcsr_acc_opencl_contexts;

#if !defined(NDEBUG)
void c_dbcsr_acc_opencl_notify(const char /*errinfo*/[], const void* /*private_info*/, size_t /*cb*/, void* /*user_data*/);
void c_dbcsr_acc_opencl_notify(const char errinfo[], const void* private_info, size_t cb, void* user_data)
{
  ACC_OPENCL_UNUSED(private_info); ACC_OPENCL_UNUSED(cb); ACC_OPENCL_UNUSED(user_data);
  fprintf(stderr, "ERROR ACC/OpenCL: %s\n", errinfo);
}
#endif


cl_context c_dbcsr_acc_opencl_context(int* tid)
{
  cl_context result;
#if defined(_OPENMP) && defined(ACC_OPENCL_THREADLOCAL_CONTEXT)
  const int i = omp_get_thread_num();
  assert(0 <= i && i < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL != c_dbcsr_acc_opencl_contexts);
  result = c_dbcsr_acc_opencl_contexts[i];
  if (0 < i && NULL == result) {
    result = c_dbcsr_acc_opencl_contexts[/*master*/0];
    if (NULL != result && CL_SUCCESS != clRetainContext(result)) {
      result = NULL;
    }
  }
  if (NULL != tid) *tid = i;
#else
  assert(NULL != c_dbcsr_acc_opencl_contexts);
  result = c_dbcsr_acc_opencl_contexts[/*master*/0];
  if (NULL != tid) *tid = 0;
#endif
  return result;
}


const char* c_dbcsr_acc_opencl_stristr(const char a[], const char b[])
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


/**
 * Comparator used with qsort; stabilized by tail condition (a < b ? -1 : 1).
 * Brings GPUs with local memory in front, followed by (potentially) integrated GPUs,
 * and further orders by memory capacity.
 */
int c_dbcsr_acc_opencl_order_devices(const void* /*dev_a*/, const void* /*dev_b*/);
int c_dbcsr_acc_opencl_order_devices(const void* dev_a, const void* dev_b)
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
        int unified_a, unified_b;
        size_t size_a, size_b;
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, &unified_a));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, &unified_b));
        if ((0 == unified_a && 0 == unified_b) || (0 != unified_a && 0 != unified_b)) {
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        /* discrete GPU goes in front */
        else if (0 == unified_b) return 1;
        else return -1;
      }
      else return -1;
    }
    else if (CL_DEVICE_TYPE_GPU & type_b) return 1;
    else {
      if (CL_DEVICE_TYPE_CPU & type_a) {
        if (CL_DEVICE_TYPE_CPU & type_b) {
          size_t size_a, size_b;
          ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
          ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        else return -1;
      }
      else if (CL_DEVICE_TYPE_CPU & type_b) return 1;
      else {
        size_t size_a, size_b;
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
        return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
      }
    }
  }
}


int c_dbcsr_acc_init(void)
{
#if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel()
# if /*WORKAROUND*/defined(__DBCSR_ACC)
    || 0/*master*/ == omp_get_thread_num()
# endif
    ) ? EXIT_SUCCESS : EXIT_FAILURE);
#else
  int result = EXIT_SUCCESS;
#endif
  if (NULL == c_dbcsr_acc_opencl_contexts) { /* avoid to initialize multiple times */
    const char *const disable = getenv("ACC_OPENCL_DISABLE");
    if (NULL == disable || '0' == *disable) {
      cl_platform_id platforms[ACC_OPENCL_DEVICES_MAXCOUNT];
      cl_device_id devices[ACC_OPENCL_DEVICES_MAXCOUNT];
      char buffer[ACC_OPENCL_BUFFERSIZE];
      const char *const env_device_vendor = getenv("ACC_OPENCL_VENDOR");
      /* TODO: introduce more advanced syntax (partitioning a device) */
      const char *const env_device_split = getenv("ACC_OPENCL_DEVSPLIT");
      const char *const env_device_type = getenv("ACC_OPENCL_DEVTYPE");
      const char *const env_device_id = getenv("ACC_OPENCL_DEVICE");
      int device_id = (NULL == env_device_id ? 0 : atoi(env_device_id));
      cl_uint nplatforms = 0, i;
      cl_device_type type = CL_DEVICE_TYPE_ALL;
      if (EXIT_SUCCESS == result) {
        ACC_OPENCL_EXPECT(CL_SUCCESS, clGetPlatformIDs(0, NULL, &nplatforms)); /* soft error */
        if (0 < nplatforms) {
          ACC_OPENCL_CHECK(clGetPlatformIDs(
            nplatforms <= ACC_OPENCL_DEVICES_MAXCOUNT ? nplatforms : ACC_OPENCL_DEVICES_MAXCOUNT,
            platforms, 0), "retrieve platform ids", result);
        }
      }
      if (NULL != env_device_type && '\0' != *env_device_type) {
        if (NULL != c_dbcsr_acc_opencl_stristr(env_device_type, "gpu")) type = CL_DEVICE_TYPE_GPU;
        else if (NULL != c_dbcsr_acc_opencl_stristr(env_device_type, "cpu")) type = CL_DEVICE_TYPE_CPU;
        else if (NULL != c_dbcsr_acc_opencl_stristr(env_device_type, "acc")
              || NULL != c_dbcsr_acc_opencl_stristr(env_device_type, "other"))
        {
          type = CL_DEVICE_TYPE_ACCELERATOR;
        }
        else type = CL_DEVICE_TYPE_ALL;
      }
      c_dbcsr_acc_opencl_config.ndevices = 0;
      for (i = 0; i < nplatforms; ++i) {
        cl_uint ndevices;
        if (CL_SUCCESS == clGetDeviceIDs(platforms[i], type, 0, NULL, &ndevices) && 0 < ndevices) {
          ACC_OPENCL_CHECK(clGetDeviceIDs(platforms[i], type, ndevices, devices, NULL),
            "retrieve device ids", result);
          if (EXIT_SUCCESS == result) {
#if defined(CL_VERSION_1_2)
            const cl_device_partition_property properties[] = {
              CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, CL_DEVICE_AFFINITY_DOMAIN_NUMA, /*terminator*/0
            };
#endif
            cl_uint j = 0, n = 0;
            for (; j < ndevices; ++j) {
#if defined(CL_VERSION_1_2)
              if ( (NULL != env_device_split && '0' == *env_device_split)
                || (c_dbcsr_acc_opencl_config.ndevices + 1) == ACC_OPENCL_DEVICES_MAXCOUNT
                || (CL_SUCCESS != clCreateSubDevices(devices[j], properties, 0, NULL, &n)))
#endif
              {
                c_dbcsr_acc_opencl_devices[c_dbcsr_acc_opencl_config.ndevices] = devices[j];
                ++c_dbcsr_acc_opencl_config.ndevices;
              }
#if defined(CL_VERSION_1_2)
              else if (1 < n) { /* create subdevices */
                if (ACC_OPENCL_DEVICES_MAXCOUNT < (c_dbcsr_acc_opencl_config.ndevices + n)) {
                  n = ACC_OPENCL_DEVICES_MAXCOUNT - (cl_uint)c_dbcsr_acc_opencl_config.ndevices;
                }
                if (EXIT_SUCCESS == clCreateSubDevices(devices[j], properties, n,
                  c_dbcsr_acc_opencl_devices + c_dbcsr_acc_opencl_config.ndevices, NULL))
                {
                  ACC_OPENCL_CHECK(clReleaseDevice(devices[j]), "release device", result);
                  c_dbcsr_acc_opencl_config.ndevices += n;
                }
                else break;
              }
              else {
                c_dbcsr_acc_opencl_devices[c_dbcsr_acc_opencl_config.ndevices] = devices[j];
                ++c_dbcsr_acc_opencl_config.ndevices;
              }
#endif
            }
          } /*else break;*/
        }
      }
      assert(NULL == c_dbcsr_acc_opencl_contexts);
      if (device_id < c_dbcsr_acc_opencl_config.ndevices) {
        if (NULL != env_device_vendor && '\0' != *env_device_vendor) {
          for (i = 0; i < (cl_uint)c_dbcsr_acc_opencl_config.ndevices;) {
            if (CL_SUCCESS == clGetDeviceInfo(c_dbcsr_acc_opencl_devices[i],
              CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL))
            {
              if (NULL == c_dbcsr_acc_opencl_stristr(buffer, env_device_vendor)) {
                --c_dbcsr_acc_opencl_config.ndevices;
                if (i < (cl_uint)c_dbcsr_acc_opencl_config.ndevices) { /* keep relative order of IDs */
                  memmove(c_dbcsr_acc_opencl_devices + i, c_dbcsr_acc_opencl_devices + i + 1,
                    sizeof(cl_device_id) * (c_dbcsr_acc_opencl_config.ndevices - i));
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
      if (device_id < c_dbcsr_acc_opencl_config.ndevices) {
        if (EXIT_SUCCESS == result && 1 < c_dbcsr_acc_opencl_config.ndevices) {
          char tmp[ACC_OPENCL_BUFFERSIZE] = "";
          cl_device_type itype;
          /* reorder devices according to c_dbcsr_acc_opencl_order_devices */
          qsort(c_dbcsr_acc_opencl_devices, c_dbcsr_acc_opencl_config.ndevices,
            sizeof(cl_device_id), c_dbcsr_acc_opencl_order_devices);
          /* search backwards to capture leading GPUs (order of devices) */
          i = c_dbcsr_acc_opencl_config.ndevices - 1;
          do {
            ACC_OPENCL_CHECK(clGetDeviceInfo(c_dbcsr_acc_opencl_devices[i],
              CL_DEVICE_TYPE, sizeof(cl_device_type), &itype, NULL),
              "retrieve device type", result);
            if (EXIT_SUCCESS == result) {
              /* preselect default device */
              if ((NULL == env_device_id || '\0' == *env_device_id)
                && (CL_DEVICE_TYPE_DEFAULT & itype))
              {
                device_id = (int)i;
                break;
              }
              /* prune number of devices to capture GPUs only */
              else if (CL_DEVICE_TYPE_ALL == type && NULL == env_device_type
                && CL_DEVICE_TYPE_GPU == itype && device_id <= (int)i)
              {
                result = clGetDeviceInfo(c_dbcsr_acc_opencl_devices[i],
                  CL_DEVICE_NAME, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
                if (CL_SUCCESS == result /* prune for homogeneous set of GPUs */
                  && 0 != strncmp(buffer, tmp, ACC_OPENCL_BUFFERSIZE))
                {
                  c_dbcsr_acc_opencl_config.ndevices = i + 1;
                  strncpy(tmp, buffer, ACC_OPENCL_BUFFERSIZE);
                }
              }
            }
            else break;
          } while (0 < i--);
        }
        if (EXIT_SUCCESS == result) {
          const char *const env_verbose = getenv("ACC_OPENCL_VERBOSE");
          cl_device_id active_device;
          c_dbcsr_acc_opencl_config.verbosity = (NULL == env_verbose ? 0 : atoi(env_verbose));
#if defined(_OPENMP)
          c_dbcsr_acc_opencl_config.nthreads = omp_get_max_threads();
#else
          c_dbcsr_acc_opencl_config.nthreads = 1;
#endif
          assert(NULL == c_dbcsr_acc_opencl_contexts);
          c_dbcsr_acc_opencl_contexts = (cl_context*)calloc( /* thread-specific */
            c_dbcsr_acc_opencl_config.nthreads, sizeof(cl_context));
          if (NULL != c_dbcsr_acc_opencl_contexts) {
            result = c_dbcsr_acc_opencl_set_active_device(device_id, &active_device);
            assert(EXIT_SUCCESS != result || NULL != c_dbcsr_acc_opencl_contexts[/*master*/0]);
          }
          else result = EXIT_FAILURE;
#if defined(ACC_OPENCL_STREAMS_MAXCOUNT)
          if (EXIT_SUCCESS == result) {
            c_dbcsr_acc_opencl_config.streams = (cl_command_queue*)calloc( /* allocate streams */
              ACC_OPENCL_STREAMS_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads, sizeof(cl_command_queue));
            if (NULL == c_dbcsr_acc_opencl_config.streams) result = EXIT_FAILURE;
            assert(0 == c_dbcsr_acc_opencl_config.nstreams);
          }
#endif
          if (EXIT_SUCCESS == result) {
            const int cl_nonv = (EXIT_SUCCESS != c_dbcsr_acc_opencl_device_vendor(active_device, "nvidia"));
            const char *const env_sync = getenv("ACC_OPENCL_ASYNC_MEMOPS");
            const char *const env_dump = getenv("ACC_OPENCL_DUMP");
            c_dbcsr_acc_opencl_config.dump = (NULL == env_dump ? 0 : atoi(env_dump));
            c_dbcsr_acc_opencl_config.async_memops = (NULL == env_sync ? cl_nonv : (0 != atoi(env_sync)));
            c_dbcsr_acc_opencl_config.record_event = (cl_nonv
              ? c_dbcsr_acc_opencl_enqueue_marker /* validation errors -> barrier */
              : c_dbcsr_acc_opencl_enqueue_barrier);
#if defined(ACC_OPENCL_SVM)
            { const char *const env_svm = getenv("ACC_OPENCL_SVM");
              int level_major = 0;
              c_dbcsr_acc_opencl_config.svm_interop = (NULL == env_svm || 0 != atoi(env_svm)) &&
                (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_level(active_device,
                  &level_major, NULL/*level_minor*/, NULL/*cl_std*/) && 2 <= level_major);
            }
#else
            c_dbcsr_acc_opencl_config.svm_interop = CL_FALSE;
#endif
            if (CL_SUCCESS != clGetDeviceInfo(active_device, CL_DEVICE_HOST_UNIFIED_MEMORY,
              sizeof(cl_bool), &c_dbcsr_acc_opencl_config.unified, NULL))
            {
              c_dbcsr_acc_opencl_config.unified = CL_FALSE;
            }
            if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_device, "intel")) {
              if (EXIT_SUCCESS != c_dbcsr_acc_opencl_device_id(active_device, "%[^[][0x%xi]",
                &c_dbcsr_acc_opencl_config.intel_id))
              {
                c_dbcsr_acc_opencl_config.intel_id = -1;
              }
            }
            else c_dbcsr_acc_opencl_config.intel_id = 0;
          }
        }
      }
      else { /* mark as initialized */
        c_dbcsr_acc_opencl_config.ndevices = -1;
      }
    }
    else { /* mark as initialized */
      c_dbcsr_acc_opencl_config.ndevices = -1;
    }
#if defined(__DBCSR_ACC)
    /* DBCSR shall call c_dbcsr_acc_init as well as libsmm_acc_init (since both interfaces are used).
     * Also, libsmm_acc_init may privately call c_dbcsr_acc_init (as it depends on the ACC interface).
     * The implementation of c_dbcsr_acc_init should hence be safe against "over initialization".
     * However, DBCSR only calls c_dbcsr_acc_init (and expects an implicit libsmm_acc_init).
     */
    if (EXIT_SUCCESS == result) {
      result = libsmm_acc_init();
    }
#endif
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_finalize(void)
{
#if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel()
# if /*WORKAROUND*/defined(__DBCSR_ACC)
    || 0/*master*/ == omp_get_thread_num()
# endif
    ) ? EXIT_SUCCESS : EXIT_FAILURE);
#else
  int result = EXIT_SUCCESS;
#endif
  if (NULL != c_dbcsr_acc_opencl_contexts) {
    int i;
    assert(0 < c_dbcsr_acc_opencl_config.ndevices);
    for (i = 0; i < c_dbcsr_acc_opencl_config.nthreads
      && EXIT_SUCCESS == result; ++i)
    {
      const cl_context context = c_dbcsr_acc_opencl_contexts[i];
      if (NULL != context) {
        c_dbcsr_acc_opencl_contexts[i] = NULL;
        result = clReleaseContext(context);
      }
    }
#if defined(__DBCSR_ACC)
    /* DBCSR may call c_dbcsr_acc_init as well as libsmm_acc_init() since both interface are used.
     * libsmm_acc_init may privately call c_dbcsr_acc_init (as it depends on the ACC interface).
     * The implementation of c_dbcsr_acc_init should be safe against "over initialization".
     * However, DBCSR only calls c_dbcsr_acc_init and expects an implicit libsmm_acc_init().
     */
    if (EXIT_SUCCESS == result) {
      result = libsmm_acc_finalize();
    }
#endif
    for (i = 0; i < ACC_OPENCL_DEVICES_MAXCOUNT; ++i) {
      const cl_device_id device_id = c_dbcsr_acc_opencl_devices[i];
      if (NULL != device_id) {
#if defined(CL_VERSION_1_2)
        ACC_OPENCL_CHECK(clReleaseDevice(device_id), "release device", result);
#endif
        /* c_dbcsr_acc_opencl_set_active_device scans for non-NULL devices */
        c_dbcsr_acc_opencl_devices[i] = NULL;
      }
    }
    { /* release buffers */
#if defined(ACC_OPENCL_STREAMS_MAXCOUNT)
      cl_command_queue *const streams = c_dbcsr_acc_opencl_config.streams;
      c_dbcsr_acc_opencl_config.nstreams = 0;
      c_dbcsr_acc_opencl_config.streams = NULL;
      free(streams);
#endif
      free(c_dbcsr_acc_opencl_contexts);
      c_dbcsr_acc_opencl_contexts = NULL;
    }
  }
  ACC_OPENCL_RETURN(result);
}


void c_dbcsr_acc_clear_errors(void)
{
}


int c_dbcsr_acc_get_ndevices(int* ndevices)
{
  int result;
#if defined(__DBCSR_ACC)
  /* DBCSR calls c_dbcsr_acc_get_ndevices before calling c_dbcsr_acc_init. */
  result = c_dbcsr_acc_init();
  if (EXIT_SUCCESS == result)
#endif
  {
    if (NULL != ndevices && 0 != c_dbcsr_acc_opencl_config.ndevices) {
      *ndevices = (0 < c_dbcsr_acc_opencl_config.ndevices ? c_dbcsr_acc_opencl_config.ndevices : 0);
      result = EXIT_SUCCESS;
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device(void* stream, cl_device_id* device)
{
  int result = EXIT_SUCCESS;
  assert(NULL != device);
  if (NULL != stream) {
    ACC_OPENCL_CHECK(clGetCommandQueueInfo(*ACC_OPENCL_STREAM(stream), CL_QUEUE_DEVICE,
      sizeof(cl_device_id), device, NULL), "retrieve device from queue", result);
  }
  else {
    cl_context context;
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
    assert(NULL != c_dbcsr_acc_opencl_contexts);
    context = c_dbcsr_acc_opencl_contexts[tid];
    if (NULL == context)
#else
    assert(NULL != c_dbcsr_acc_opencl_contexts);
#endif
    {
      context = c_dbcsr_acc_opencl_contexts[/*master*/0];
    }
    if (NULL != context) {
      ACC_OPENCL_CHECK(clGetContextInfo(context,
        CL_CONTEXT_DEVICES, sizeof(cl_device_id), device, NULL),
        "retrieve id of active device", result);
    }
    else {
      *device = NULL;
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_vendor(cl_device_id device, const char vendor[])
{
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != vendor);
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR,
    ACC_OPENCL_BUFFERSIZE, buffer, NULL),
    "retrieve device vendor", result);
  if (EXIT_SUCCESS == result) {
    return (NULL != c_dbcsr_acc_opencl_stristr(buffer, vendor)
      ? EXIT_SUCCESS
      : EXIT_FAILURE);
  }
  else ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_name(cl_device_id device, const char match[])
{
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != match);
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME,
    ACC_OPENCL_BUFFERSIZE, buffer, NULL),
    "retrieve device name", result);
  if (EXIT_SUCCESS == result) {
    const char *const p = c_dbcsr_acc_opencl_stristr(buffer, match);
    return (NULL != p ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  else ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_id(cl_device_id device, const char format[], int* id)
{
  char buffer[ACC_OPENCL_BUFFERSIZE], skip[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != format && NULL != id);
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME,
    ACC_OPENCL_BUFFERSIZE, buffer, NULL),
    "retrieve device name", result);
  if (EXIT_SUCCESS == result) {
    const int n = sscanf(buffer, format, skip, id);
    return (2 == n ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  else {
    id = 0; ACC_OPENCL_RETURN(result);
  }
}


int c_dbcsr_acc_opencl_device_level(cl_device_id device,
  int* level_major, int* level_minor, char cl_std[16])
{
  char buffer[ACC_OPENCL_BUFFERSIZE];
  unsigned int cl_std_level[2];
  cl_int result = clGetDeviceInfo(device, CL_DEVICE_VERSION, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
  assert(NULL != device && (NULL != level_major || NULL != level_minor || NULL != cl_std));
  if (CL_SUCCESS == result) {
    if (2 == sscanf(buffer, "OpenCL %u.%u", cl_std_level, cl_std_level + 1)) {
      if (NULL != level_major) *level_major = (int)cl_std_level[0];
      if (NULL != level_minor) *level_minor = (int)cl_std_level[1];
      if (NULL != cl_std) {
        if (2 <= cl_std_level[0]) {
          const int nchar = ACC_OPENCL_SNPRINTF(cl_std, 16, "-cl-std=CL%u.0", cl_std_level[0]);
          if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
        }
        else if (1 <= cl_std_level[0]) {
          if (1 <= cl_std_level[1]) {
            const int nchar = ACC_OPENCL_SNPRINTF(cl_std, 16, "-cl-std=CL%u.%u", cl_std_level[0], cl_std_level[1]);
            if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
          }
          else {
            result = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
            if (CL_SUCCESS == result) {
              if (2 == sscanf(buffer, "OpenCL C %u.%u", cl_std_level, cl_std_level + 1)) {
                const int nchar = ACC_OPENCL_SNPRINTF(cl_std, 16, "-cl-std=CL%u.%u", cl_std_level[0], cl_std_level[1]);
                if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
              }
              else {
                result = EXIT_FAILURE;
                *cl_std = '\0';
              }
            }
            else *cl_std = '\0';
          }
        }
        else *cl_std = '\0';
      }
    }
    else {
      if (NULL != level_major) *level_major = 0;
      if (NULL != level_minor) *level_minor = 0;
      if (NULL != cl_std) *cl_std = '\0';
      result = EXIT_FAILURE;
    }
  }
  else {
    if (NULL != level_major) *level_major = 0;
    if (NULL != level_minor) *level_minor = 0;
    if (NULL != cl_std) *cl_std = '\0';
  }
  return result;
}


int c_dbcsr_acc_opencl_device_ext(cl_device_id device, const char *const extnames[], int num_exts)
{
  int result = ((NULL != extnames && 0 < num_exts) ? EXIT_SUCCESS : EXIT_FAILURE);
  char extensions[ACC_OPENCL_BUFFERSIZE], buffer[ACC_OPENCL_BUFFERSIZE];
  assert(NULL != device);
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS,
    ACC_OPENCL_BUFFERSIZE, extensions, NULL),
    "retrieve device extensions", result);
  if (EXIT_SUCCESS == result) {
    do {
      if (NULL != extnames[--num_exts]) {
        char* ext = strtok(strncpy(buffer, extnames[num_exts], ACC_OPENCL_BUFFERSIZE - 1), ACC_OPENCL_DELIMS);
        for (; NULL != ext; ext = strtok('\0' != *ext ? (ext + strlen(ext) + 1) : ext, ACC_OPENCL_DELIMS)) {
          if (NULL == strstr(extensions, ext)) {
            return EXIT_FAILURE;
          }
        }
      }
    } while (0 < num_exts);
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_set_active_device(int device_id, cl_device_id* device)
{
  cl_int result;
  if (0 < c_dbcsr_acc_opencl_config.ndevices && 0 <= device_id && device_id < ACC_OPENCL_DEVICES_MAXCOUNT) {
    const cl_device_id active_id = c_dbcsr_acc_opencl_devices[device_id];
    cl_device_id current_id = NULL;
    result = NULL != active_id
      ? c_dbcsr_acc_opencl_device(NULL/*stream*/, &current_id)
      : EXIT_FAILURE;
    if (EXIT_SUCCESS == result && active_id != current_id) {
      cl_platform_id platform = NULL;
      int tid;
      const cl_context context = c_dbcsr_acc_opencl_context(&tid);
      assert(0 <= tid && tid < c_dbcsr_acc_opencl_config.nthreads);
      ACC_OPENCL_CHECK(clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM,
        sizeof(cl_platform_id), &platform, NULL),
        "query device platform", result);
      if (NULL != context) {
        c_dbcsr_acc_opencl_contexts[tid] = NULL;
        ACC_OPENCL_CHECK(clReleaseContext(context),
          "release context", result);
      }
      if (EXIT_SUCCESS == result) {
#if defined(NDEBUG)
        void (*const notify)(const char*, const void*, size_t, void*) = NULL;
#else
        void (*const notify)(const char*, const void*, size_t, void*) = c_dbcsr_acc_opencl_notify;
#endif
        cl_context_properties properties[] = {
          CL_CONTEXT_PLATFORM, 0/*placeholder*/,
          0 /* end of properties */
        };
        properties[1] = (long)platform;
        c_dbcsr_acc_opencl_contexts[tid] = clCreateContext(properties,
          1/*num_devices*/, &active_id, notify, NULL/* user_data*/, &result);
        if (CL_INVALID_VALUE == result) { /* retry */
          c_dbcsr_acc_opencl_contexts[tid] = clCreateContext(NULL/*properties*/,
            1/*num_devices*/, &active_id, notify, NULL/* user_data*/, &result);
        }
        if (EXIT_SUCCESS == result) {
          if (0 != c_dbcsr_acc_opencl_config.verbosity) {
            char buffer[ACC_OPENCL_BUFFERSIZE];
            if (CL_SUCCESS == clGetDeviceInfo(active_id, CL_DEVICE_NAME,
              ACC_OPENCL_BUFFERSIZE, buffer, NULL))
            {
              fprintf(stderr, "INFO ACC/OpenCL: ndevices=%i device%i=\"%s\"\n",
                c_dbcsr_acc_opencl_config.ndevices, device_id, buffer);
            }
          }
        }
        else {
          if (CL_INVALID_DEVICE == result) {
            if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia")) {
              fprintf(stderr,
                "WARNING ACC/OpenCL: if MPI-ranks target the same device in exclusive mode,\n"
                "                    SMI must be used to enable sharing the device.\n");
            }
          }
          ACC_OPENCL_ERROR("create context", result);
        }
      }
    }
    if (NULL != device) {
      *device = (EXIT_SUCCESS == result ? active_id : NULL);
    }
  }
  else if (0 > c_dbcsr_acc_opencl_config.ndevices) {
    /* allow successful completion if no device was found */
    result = EXIT_SUCCESS;
  }
  else {
    result = EXIT_FAILURE;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_set_active_device(int device_id)
{
  ACC_OPENCL_RETURN(c_dbcsr_acc_opencl_set_active_device(device_id, NULL/*device*/));
}


int c_dbcsr_acc_device_synchronize(void)
{
  int result = EXIT_SUCCESS;
#if defined(ACC_OPENCL_STREAMS_MAXCOUNT)
  cl_device_id active_id = NULL;
  result = c_dbcsr_acc_opencl_device(NULL/*stream*/, &active_id);
  if (EXIT_SUCCESS == result && NULL != active_id) {
    int i = 0, nstreams;
# if defined(_OPENMP)
#   if (201107/*v3.1*/ <= _OPENMP)
#   pragma omp atomic read
#   else
#   pragma omp critical(c_dbcsr_acc_opencl_nstreams)
#   endif
# endif
    nstreams = c_dbcsr_acc_opencl_config.nstreams;
    for (; i < nstreams && EXIT_SUCCESS == result; ++i) {
      const cl_command_queue stream = c_dbcsr_acc_opencl_config.streams[i];
      if (NULL != stream) {
        cl_device_id device;
        result = c_dbcsr_acc_opencl_device(stream, &device);
        if (EXIT_SUCCESS == result) {
          if (device == active_id) { /* synchronize */
            assert(stream == c_dbcsr_acc_opencl_config.streams[i]);
            result = c_dbcsr_acc_stream_sync(stream);
            assert(stream == c_dbcsr_acc_opencl_config.streams[i]);
          }
        }
      }
    }
  }
#else
  result = EXIT_FAILURE;
#endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_wgsize(cl_device_id device, cl_kernel kernel,
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
        CL_DEVICE_MAX_WORK_GROUP_SIZE,
        sizeof(size_t), &value, NULL),
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
      *preferred_multiple = 1;
#endif
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_kernel(const char source[], const char kernel_name[],
  const char build_params[], const char build_options[],
  const char try_build_options[], int* try_ok,
  const char *const extnames[], int num_exts,
  cl_kernel* kernel)
{
  char buffer[ACC_OPENCL_BUFFERSIZE] = "", cl_std[16];
  cl_device_id active_id = NULL;
  const cl_context context = c_dbcsr_acc_opencl_context(NULL);
  cl_int result = (NULL != context
    ? c_dbcsr_acc_opencl_device(NULL/*stream*/, &active_id)
    : EXIT_FAILURE);
  int level_major, level_minor, ok = EXIT_SUCCESS;
  assert(NULL != source && NULL != kernel);
  assert(NULL != kernel_name && '\0' != *kernel_name);
  if (EXIT_SUCCESS == result) {
    result = c_dbcsr_acc_opencl_device_level(active_id, &level_major, &level_minor, cl_std);
  }
  if (EXIT_SUCCESS == result) {
    const char* ext_source = source;
    size_t size_src = strlen(source);
    cl_program program = NULL;
    if (NULL != extnames) {
      int n = num_exts, nflat = 0;
      size_t size_ext = 0;
      for (; 0 < n; --n) if (NULL != extnames[n-1]) {
        char* ext = strtok(strncpy(buffer, extnames[n-1], ACC_OPENCL_BUFFERSIZE - 1), ACC_OPENCL_DELIMS);
        for (; NULL != ext; ext = strtok('\0' != *ext ? (ext + strlen(ext) + 1) : ext, ACC_OPENCL_DELIMS), ++nflat) {
          size_ext += strlen(ext);
        }
      }
      if (0 < size_ext && 0 < nflat) {
        const char *const enable_ext = "#pragma OPENCL EXTENSION %s : enable\n";
        const size_t size_src_ext = size_src + size_ext + nflat * (strlen(enable_ext) - 2/*%s*/);
        char *const ext_source_buffer = (char*)malloc(size_src_ext + 1/*terminator*/);
        if (NULL != ext_source_buffer) {
          for (n = 0; 0 < num_exts; --num_exts) if (NULL != extnames[num_exts-1]) {
            char* ext = strtok(strncpy(buffer, extnames[num_exts-1], ACC_OPENCL_BUFFERSIZE - 1), ACC_OPENCL_DELIMS);
            for (; NULL != ext; ext = strtok('\0' != *ext ? (ext + strlen(ext) + 1) : ext, ACC_OPENCL_DELIMS)) {
#if !defined(NDEBUG)
              if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(active_id, (const char**)&ext, 1))
#endif
              { /* NDEBUG: assume given extension is supported (confirmed upfront) */
                n += ACC_OPENCL_SNPRINTF(ext_source_buffer + n, size_src_ext + 1/*terminator*/ - n,
                  enable_ext, ext);
              }
#if !defined(NDEBUG)
              else fprintf(stderr, "WARNING ACC/OpenCL: extension \"%s\" is not supported.\n", ext);
#endif
            }
          }
          memcpy(ext_source_buffer + n, source, size_src);
          size_src += n; /* according to given/permitted extensions */
          assert(size_src <= size_src_ext);
          ext_source_buffer[size_src] = '\0';
          ext_source = ext_source_buffer;
        }
      }
      buffer[0] = '\0'; /* reset to empty */
    }
    /* consider preprocessing kernel for analysis (cpp); failure does not matter (result) */
    if (0 != c_dbcsr_acc_opencl_config.dump) {
      char name_src[ACC_OPENCL_KERNELNAME_MAXSIZE*2];
      int nchar = ACC_OPENCL_SNPRINTF(name_src, sizeof(name_src), "/tmp/.%s.cl", kernel_name);
      if (0 < nchar && (int)sizeof(name_src) > nchar) {
        FILE *const file_cpp = fopen(ACC_OPENCL_CPPBIN, "rb");
        FILE *const file_sed = fopen(ACC_OPENCL_SEDBIN, "rb");
        if (NULL != file_sed) fclose(file_sed); /* existence-check */
        if (NULL != file_cpp) {
          FILE *const file_src = fopen(name_src, "w");
          fclose(file_cpp); /* existence-check */
          if (NULL != file_src) {
            if (size_src == fwrite(ext_source, 1, size_src, file_src) && EXIT_SUCCESS == fclose(file_src)) {
              nchar = ACC_OPENCL_SNPRINTF(buffer, sizeof(buffer), ACC_OPENCL_CPPBIN
                " -P -C -nostdinc -D__OPENCL_VERSION__=%u %s %s %s %s > %s.cl", 100 * level_major + 10 * level_minor,
                EXIT_SUCCESS != c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia") ? "" : "-D__NV_CL_C_VERSION",
                NULL != build_params ? build_params : "", name_src,
                NULL != file_sed ? "| " ACC_OPENCL_SEDBIN " '/^[[:space:]]*$/d'" : "",
                kernel_name);
              if (0 < nchar && (int)sizeof(buffer) > nchar) {
                if (EXIT_SUCCESS == system(buffer)) {
                  nchar = ACC_OPENCL_SNPRINTF(buffer, sizeof(buffer), "%s.cl", kernel_name);
                  if (0 < nchar && (int)sizeof(buffer) > nchar) {
                    FILE *const file = fopen(buffer, "r");
                    if (NULL != file) {
                      const long int size = (EXIT_SUCCESS == fseek(
                        file, 0/*offset*/, SEEK_END) ? ftell(file) : 0);
                      char *const src = (char*)(EXIT_SUCCESS == fseek(
                        file, 0/*offset*/, SEEK_SET) ? malloc(size + 1/*terminator*/) : NULL);
                      if (NULL != src) {
                        if ((size_t)size == fread(src, 1/*sizeof(char)*/, size/*count*/, file)) {
                          if (source != ext_source) free((void*)ext_source);
                          src[size] = '\0';
                          ext_source = src;
                        }
                        else free(src);
                      }
                      fclose(file);
                    }
                  }
                }
              }
              buffer[0] = '\0'; /* reset to empty */
            }
            remove(name_src);
          }
        }
      }
    }
    program = clCreateProgramWithSource(context, 1/*nlines*/, &ext_source, NULL, &result);
    if (NULL != program) {
      int nchar = ACC_OPENCL_SNPRINTF(buffer, sizeof(buffer), "%s %s %s %s",
        cl_std, NULL != build_options ? build_options : "",
        NULL != try_build_options ? try_build_options : "",
        NULL != build_params ? build_params : "");
      assert(CL_SUCCESS == result);
      result = ((0 < nchar && (int)sizeof(buffer) > nchar)
        ? clBuildProgram(program, 1/*num_devices*/, &active_id,
            buffer, NULL/*callback*/, NULL/*user_data*/)
        : EXIT_FAILURE);
      if (CL_SUCCESS != result && NULL != try_build_options && '\0' != *try_build_options) {
        nchar = ACC_OPENCL_SNPRINTF(buffer, sizeof(buffer), "%s %s %s", cl_std,
          NULL != build_options ? build_options : "",
          NULL != build_params ? build_params : "");
        ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
        /* recreate program after building it failed (unclean state) */
        program = clCreateProgramWithSource(context, 1/*nlines*/, &ext_source, NULL, &result);
        if (NULL != program && 0 < nchar && (int)sizeof(buffer) > nchar) {
          result = clBuildProgram(program, 1/*num_devices*/, &active_id,
            buffer, NULL/*callback*/, NULL/*user_data*/);
        }
        ok = EXIT_FAILURE;
      }
      if (source != ext_source) free((void*)ext_source);
      buffer[0] = '\0'; /* reset to empty */
      if (CL_SUCCESS == result) {
        *kernel = clCreateKernel(program, kernel_name, &result);
        if (CL_SUCCESS == result) {
          assert(NULL != *kernel);
          if (2 <= c_dbcsr_acc_opencl_config.dump || 0 > c_dbcsr_acc_opencl_config.dump) {
            unsigned char* binary = NULL;
            size_t size;
            binary = (unsigned char*)(CL_SUCCESS == clGetProgramInfo(program,
                CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL)
              ? malloc(size) : NULL);
            if (NULL != binary) {
              result = clGetProgramInfo(program, CL_PROGRAM_BINARIES,
                sizeof(unsigned char*), &binary, NULL);
              if (CL_SUCCESS == result) {
                FILE* file;
                nchar = ACC_OPENCL_SNPRINTF(buffer, sizeof(buffer), "%s.dump", kernel_name);
                file = (0 < nchar && (int)sizeof(buffer) > nchar) ? fopen(buffer, "wb") : NULL;
                buffer[0] = '\0'; /* reset to empty */
                if (NULL != file) {
                  if (size != fwrite(binary, 1, size, file)) {
                    ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
                    ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseKernel(*kernel));
                    result = EXIT_FAILURE;
                  }
                  fclose(file);
                }
                else {
                  ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
                  ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseKernel(*kernel));
                  result = EXIT_FAILURE;
                }
              }
              else {
                ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
                ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseKernel(*kernel));
                ACC_OPENCL_ERROR("query program binary", result);
              }
              free(binary);
            }
            else {
              ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
              ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseKernel(*kernel));
              result = EXIT_FAILURE;
            }
          }
        }
        else {
          ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
          ACC_OPENCL_ERROR("create kernel", result);
        }
      }
      else {
        ACC_OPENCL_EXPECT(CL_SUCCESS, clGetProgramBuildInfo(
          program, active_id, CL_PROGRAM_BUILD_LOG,
          ACC_OPENCL_BUFFERSIZE, buffer, NULL));
        ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
      }
    }
    else {
      if (source != ext_source) free((void*)ext_source);
      assert(CL_SUCCESS != result);
      ACC_OPENCL_ERROR("create program", result);
    }
  }
#if !defined(NDEBUG)
  if (EXIT_SUCCESS != result) *kernel = NULL;
#endif
  if (NULL != try_ok) *try_ok = result | ok;
  ACC_OPENCL_RETURN_CAUSE(result, buffer);
}

#if defined(__cplusplus)
}
#endif

#endif /*__OPENCL*/
