/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: GPL-2.0+                                                              */
/*------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#  include "acc_opencl.h"
#  include <libxsmm_sync.h>
#  include <stdlib.h>
#  include <string.h>
#  include <assert.h>
#  include <limits.h>
#  include <ctype.h>
#  if defined(_OPENMP)
#    include <omp.h>
#  endif
#  if defined(_WIN32)
#    include <windows.h>
#    include <process.h>
#  else
LIBXSMM_EXTERN int mkstemp(char*) LIBXSMM_NOTHROW;
LIBXSMM_EXTERN int putenv(char*) LIBXSMM_THROW;
#    include <unistd.h>
#    include <errno.h>
#    include <glob.h>
#  endif
#  if defined(__DBCSR_ACC)
#    include "../acc_libsmm.h"
#  endif

#  include <sys/stat.h>
#  if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#    define S_ISDIR(A) ((S_IFMT & (A)) == S_IFDIR)
#  endif

#  if !defined(ACC_OPENCL_CACHEDIR) && 1
#    define ACC_OPENCL_CACHEDIR ".cl_cache"
#  endif
#  if !defined(ACC_OPENCL_CPPBIN) && 1
#    define ACC_OPENCL_CPPBIN "/usr/bin/cpp"
#  endif
#  if !defined(ACC_OPENCL_SEDBIN) && 1
#    define ACC_OPENCL_SEDBIN "/usr/bin/sed"
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

/* global configuration discovered during initialization */
c_dbcsr_acc_opencl_config_t c_dbcsr_acc_opencl_config;


#  if !defined(NDEBUG)
void c_dbcsr_acc_opencl_notify(const char /*errinfo*/[], const void* /*private_info*/, size_t /*cb*/, void* /*user_data*/);
void c_dbcsr_acc_opencl_notify(const char errinfo[], const void* private_info, size_t cb, void* user_data) {
  LIBXSMM_UNUSED(private_info);
  LIBXSMM_UNUSED(cb);
  LIBXSMM_UNUSED(user_data);
  fprintf(stderr, "ERROR ACC/OpenCL: %s\n", errinfo);
}
#  endif


cl_context c_dbcsr_acc_opencl_context(void) {
  cl_context result;
  const int tid = ACC_OPENCL_OMP_TID();
  assert(0 <= tid && tid < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL != c_dbcsr_acc_opencl_config.contexts);
  result = c_dbcsr_acc_opencl_config.contexts[tid];
  if (NULL == result) { /* fallback */
    int i = 0; /* prefer master's context */
    for (; i < c_dbcsr_acc_opencl_config.nthreads; ++i) {
      if (tid != i) { /* adopt another context */
        result = c_dbcsr_acc_opencl_config.contexts[i];
        if (NULL != result && CL_SUCCESS == clRetainContext(result)) {
          break;
        }
        else {
          result = NULL;
        }
      }
    }
  }
  return result;
}


cl_context c_dbcsr_acc_opencl_device_context(cl_device_id device, const int* thread_id) {
  const int i0 = (NULL != thread_id ? *thread_id : /*master*/ 0);
  cl_context result = NULL;
  int i = 0;
  for (; i < c_dbcsr_acc_opencl_config.nthreads; ++i) {
    const int j = i + i0, tid = (j < c_dbcsr_acc_opencl_config.nthreads ? j : (j - c_dbcsr_acc_opencl_config.nthreads));
    result = c_dbcsr_acc_opencl_config.contexts[tid];
    if (NULL != result) {
      cl_device_id device_id = NULL;
      if (CL_SUCCESS == clGetContextInfo(result, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device_id, NULL) &&
          device == device_id) {
        break;
      }
      else {
        result = NULL;
      }
    }
  }
  return result;
}


/**
 * Comparator used with qsort; stabilized by tail condition (a < b ? -1 : 1).
 * Brings GPUs with local memory in front, followed by (potentially) integrated GPUs,
 * and further orders by memory capacity.
 */
int c_dbcsr_acc_opencl_order_devices(const void* /*dev_a*/, const void* /*dev_b*/);
int c_dbcsr_acc_opencl_order_devices(const void* dev_a, const void* dev_b) {
  const cl_device_id* const a = (const cl_device_id*)dev_a;
  const cl_device_id* const b = (const cl_device_id*)dev_b;
  cl_device_type type_a = 0, type_b = 0;
  assert(NULL != a && NULL != b && a != b);
  ACC_OPENCL_EXPECT(EXIT_SUCCESS, clGetDeviceInfo(*a, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_a, NULL));
  ACC_OPENCL_EXPECT(EXIT_SUCCESS, clGetDeviceInfo(*b, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_b, NULL));
  if (CL_DEVICE_TYPE_DEFAULT & type_a) {
    return -1;
  }
  else if (CL_DEVICE_TYPE_DEFAULT & type_b) {
    return 1;
  }
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
        else if (0 == unified_b) {
          return 1;
        }
        else {
          return -1;
        }
      }
      else {
        return -1;
      }
    }
    else if (CL_DEVICE_TYPE_GPU & type_b) {
      return 1;
    }
    else {
      if (CL_DEVICE_TYPE_CPU & type_a) {
        if (CL_DEVICE_TYPE_CPU & type_b) {
          size_t size_a, size_b;
          ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
          ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        else {
          return -1;
        }
      }
      else if (CL_DEVICE_TYPE_CPU & type_b) {
        return 1;
      }
      else {
        size_t size_a = 0, size_b = 0;
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
        return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
      }
    }
  }
}


int c_dbcsr_acc_opencl_order_streams(const void* /*a*/, const void* /*b*/);
int c_dbcsr_acc_opencl_order_streams(const void* a, const void* b) { /* NULL-pointers are sorted to the upper end */
  return a < b ? -1 : (a > b ? 1 : 0);
}


int c_dbcsr_acc_init(void) {
#  if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*master*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
#  else
  int result = EXIT_SUCCESS;
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != result)
  ACC_OPENCL_DEBUG_FPRINTF(stderr, "ERROR ACC/OpenCL: c_dbcsr_acc_init called in OpenMP parallel region!\n");
  if (0 == c_dbcsr_acc_opencl_config.ndevices) { /* avoid to initialize multiple times */
    cl_platform_id platforms[ACC_OPENCL_DEVICES_MAXCOUNT] = {NULL};
    cl_device_id devices[ACC_OPENCL_DEVICES_MAXCOUNT];
    char buffer[ACC_OPENCL_BUFFERSIZE];
    const char *const disable = getenv("ACC_OPENCL_DISABLE"), *const env_verbose = getenv("ACC_OPENCL_VERBOSE");
    const char *const env_share = getenv("ACC_OPENCL_SHARE"), *const env_async = getenv("ACC_OPENCL_ASYNC");
    const char *const env_priority = getenv("ACC_OPENCL_PRIORITY"), *const env_flush = getenv("ACC_OPENCL_FLUSH");
    const char *const env_devmatch = getenv("ACC_OPENCL_DEVMATCH"), *const env_devtype = getenv("ACC_OPENCL_DEVTYPE");
    const char *const env_dump_acc = getenv("ACC_OPENCL_DUMP"), *const env_dump_igc = getenv("IGC_ShaderDumpEnable");
    const char* const env_dump = (NULL != env_dump_acc ? env_dump_acc : env_dump_igc);
    const char* const env_device = getenv("ACC_OPENCL_DEVICE");
    int device_id = (NULL == env_device ? 0 : atoi(env_device));
    cl_uint nplatforms = 0, i;
    cl_device_type type = CL_DEVICE_TYPE_ALL;
#  if defined(_OPENMP)
    const int max_threads = omp_get_max_threads(), num_threads = omp_get_num_threads();
    c_dbcsr_acc_opencl_config.nthreads = (num_threads < max_threads ? max_threads : num_threads);
#  else
    c_dbcsr_acc_opencl_config.nthreads = 1;
#  endif
    c_dbcsr_acc_opencl_config.dump = (NULL == env_dump ? /*default*/ 0 : atoi(env_dump));
    c_dbcsr_acc_opencl_config.verbosity = (NULL == env_verbose ? 0 : atoi(env_verbose));
    c_dbcsr_acc_opencl_config.priority = (NULL == env_priority ? /*default*/ 3 : atoi(env_priority));
    c_dbcsr_acc_opencl_config.share = (NULL == env_share ? /*default*/ 0 : atoi(env_share));
    c_dbcsr_acc_opencl_config.async = (NULL == env_async ? /*default*/ 0 : (0 != atoi(env_async)));
    c_dbcsr_acc_opencl_config.flush = (NULL == env_flush ? /*default*/ 0 : atoi(env_flush));
    c_dbcsr_acc_opencl_config.devmatch = (NULL == env_devmatch
#  if defined(ACC_OPENCL_DEVMATCH)
                                            ? CL_TRUE
                                            : (0 != atoi(env_devmatch) ? CL_TRUE : CL_FALSE));
#  else
                                            ? CL_FALSE
                                            : (0 != atoi(env_devmatch) ? CL_TRUE : CL_FALSE));
#  endif
    if ((NULL == disable || '0' == *disable) && EXIT_SUCCESS == result) {
#  if defined(ACC_OPENCL_CACHEDIR)
      const char* const env_cache = getenv("ACC_OPENCL_CACHE");
      const int cache = (NULL == env_cache
#    if defined(__DBCSR_ACC)
                           ? CL_TRUE
                           : (0 != atoi(env_cache) ? CL_TRUE : CL_FALSE));
#    else
                           ? CL_FALSE
                           : (0 != atoi(env_cache) ? CL_TRUE : CL_FALSE));
#    endif
      struct stat cachedir;
      if (0 != cache || (stat(ACC_OPENCL_CACHEDIR, &cachedir) == 0 && S_ISDIR(cachedir.st_mode))) {
#    if !defined(_WIN32)
        char cl_cache_dir[] = "cl_cache_dir=" ACC_OPENCL_CACHEDIR;
#      if defined(S_IRWXU) && defined(S_IRGRP) && defined(S_IXGRP) && defined(S_IROTH) && defined(S_IXOTH)
        const int mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
#      else
        const int mode = 0xFFFFFFFF;
#      endif
        if (0 == mkdir(ACC_OPENCL_CACHEDIR, mode) || EEXIST == errno) { /* putenv before entering OpenCL */
          ACC_OPENCL_EXPECT(0, putenv(cl_cache_dir)); /* soft-error */
        }
#    endif
      }
#  endif
      if (CL_SUCCESS == clGetPlatformIDs(0, NULL, &nplatforms) && 0 < nplatforms) {
        ACC_OPENCL_CHECK(
          clGetPlatformIDs(nplatforms <= ACC_OPENCL_DEVICES_MAXCOUNT ? nplatforms : ACC_OPENCL_DEVICES_MAXCOUNT, platforms, 0),
          "retrieve platform ids", result);
      }
      if (EXIT_SUCCESS == result) {
        if (NULL != env_devtype && '\0' != *env_devtype) {
          if (NULL != libxsmm_stristr(env_devtype, "gpu")) {
            type = CL_DEVICE_TYPE_GPU;
          }
          else if (NULL != libxsmm_stristr(env_devtype, "cpu")) {
            type = CL_DEVICE_TYPE_CPU;
          }
          else if (NULL != libxsmm_stristr(env_devtype, "acc") || NULL != libxsmm_stristr(env_devtype, "other")) {
            type = CL_DEVICE_TYPE_ACCELERATOR;
          }
          else {
            type = CL_DEVICE_TYPE_ALL;
          }
        }
        c_dbcsr_acc_opencl_config.ndevices = 0;
        for (i = 0; i < nplatforms; ++i) {
          cl_uint ndevices;
          if (CL_SUCCESS == clGetDeviceIDs(platforms[i], type, 0, NULL, &ndevices) && 0 < ndevices) {
            ACC_OPENCL_CHECK(clGetDeviceIDs(platforms[i], type, ndevices, devices, NULL), "retrieve device ids", result);
            if (EXIT_SUCCESS == result) {
              /* TODO: introduce more advanced syntax (partitioning a device) */
              const char* const env_devsplit = getenv("ACC_OPENCL_DEVSPLIT");
#  if defined(CL_VERSION_1_2)
              const cl_device_partition_property properties[] = {
                CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, CL_DEVICE_AFFINITY_DOMAIN_NUMA, /*terminator*/ 0};
#  endif
              cl_uint j = 0, n = 0;
              for (; j < ndevices; ++j) {
#  if defined(CL_VERSION_1_2)
                if ((NULL != env_devsplit && '0' == *env_devsplit) ||
                    (c_dbcsr_acc_opencl_config.ndevices + 1) == ACC_OPENCL_DEVICES_MAXCOUNT ||
                    (CL_SUCCESS != clCreateSubDevices(devices[j], properties, 0, NULL, &n)))
#  endif
                {
                  c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.ndevices] = devices[j];
                  ++c_dbcsr_acc_opencl_config.ndevices;
                }
#  if defined(CL_VERSION_1_2)
                else if (1 < n) { /* create subdevices */
                  if (ACC_OPENCL_DEVICES_MAXCOUNT < (c_dbcsr_acc_opencl_config.ndevices + n)) {
                    n = ACC_OPENCL_DEVICES_MAXCOUNT - (cl_uint)c_dbcsr_acc_opencl_config.ndevices;
                  }
                  if (EXIT_SUCCESS == clCreateSubDevices(devices[j], properties, n,
                                        c_dbcsr_acc_opencl_config.devices + c_dbcsr_acc_opencl_config.ndevices, NULL)) {
                    ACC_OPENCL_CHECK(clReleaseDevice(devices[j]), "release device", result);
                    c_dbcsr_acc_opencl_config.ndevices += n;
                  }
                  else {
                    break;
                  }
                }
                else {
                  c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.ndevices] = devices[j];
                  ++c_dbcsr_acc_opencl_config.ndevices;
                }
#  endif
              }
            } /*else break;*/
          }
        }
      }
      if (EXIT_SUCCESS == result && 0 < c_dbcsr_acc_opencl_config.ndevices) {
        const char* const env_vendor = getenv("ACC_OPENCL_VENDOR");
        /* filter device by vendor (if requested) */
        if (NULL != env_vendor && '\0' != *env_vendor) {
          for (i = 0; i < (cl_uint)c_dbcsr_acc_opencl_config.ndevices;) {
            if (CL_SUCCESS ==
                clGetDeviceInfo(c_dbcsr_acc_opencl_config.devices[i], CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL)) {
              if (NULL == libxsmm_stristr(buffer, env_vendor)) {
                --c_dbcsr_acc_opencl_config.ndevices;
                if (i < (cl_uint)c_dbcsr_acc_opencl_config.ndevices) { /* keep original order (stable) */
                  memmove(c_dbcsr_acc_opencl_config.devices + i, c_dbcsr_acc_opencl_config.devices + i + 1,
                    sizeof(cl_device_id) * (c_dbcsr_acc_opencl_config.ndevices - i));
                }
              }
              else {
                ++i;
              }
            }
            else {
              break; /* error: retrieving device vendor */
            }
          }
        }
        /* reorder devices according to c_dbcsr_acc_opencl_order_devices */
        if (EXIT_SUCCESS == result && 1 < c_dbcsr_acc_opencl_config.ndevices) {
          qsort(c_dbcsr_acc_opencl_config.devices, c_dbcsr_acc_opencl_config.ndevices, sizeof(cl_device_id),
            c_dbcsr_acc_opencl_order_devices);
        }
      }
      if (EXIT_SUCCESS == result && 0 < c_dbcsr_acc_opencl_config.ndevices) {
        /* preselect any default device or prune to homogeneous set of GPUs */
        if (NULL == env_device || '\0' == *env_device) {
          const cl_uint ndevices = (cl_uint)c_dbcsr_acc_opencl_config.ndevices;
          char tmp[ACC_OPENCL_BUFFERSIZE] = "";
          for (i = 0; i < ndevices; ++i) {
            cl_device_type itype;
            result = clGetDeviceInfo(c_dbcsr_acc_opencl_config.devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &itype, NULL);
            if (CL_SUCCESS == result) {
              if (0 != (CL_DEVICE_TYPE_DEFAULT & itype)) {
                device_id = (int)i;
                break;
              }
              else if (CL_DEVICE_TYPE_ALL == type && NULL == env_devtype && CL_DEVICE_TYPE_GPU == itype && device_id <= (int)i) {
                result = clGetDeviceInfo(c_dbcsr_acc_opencl_config.devices[i], CL_DEVICE_NAME, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
                if (CL_SUCCESS == result /* prune for homogeneous set of GPUs */
                    && ('\0' == *tmp || 0 == strncmp(buffer, tmp, ACC_OPENCL_BUFFERSIZE))) {
                  c_dbcsr_acc_opencl_config.ndevices = i + 1;
                  strncpy(tmp, buffer, ACC_OPENCL_BUFFERSIZE);
                }
                else {
                  break; /* error: retrieving device name */
                }
              }
            }
            else {
              break; /* error: retrieving device type */
            }
          }
        }
        else { /* prune number of devices to only expose requested ID */
          if (1 < c_dbcsr_acc_opencl_config.ndevices) {
            if (0 < device_id) {
              c_dbcsr_acc_opencl_config.devices[0] = c_dbcsr_acc_opencl_config.devices[device_id];
            }
            c_dbcsr_acc_opencl_config.ndevices = 1;
          }
          device_id = 0;
        }
      }
      if (device_id < c_dbcsr_acc_opencl_config.ndevices) {
        if (EXIT_SUCCESS == result) {
          assert(NULL == c_dbcsr_acc_opencl_config.contexts && 0 < c_dbcsr_acc_opencl_config.ndevices);
          assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_DEVICES_MAXCOUNT);
          c_dbcsr_acc_opencl_config.contexts = (cl_context*)calloc(/* thread-specific */
            c_dbcsr_acc_opencl_config.nthreads, sizeof(cl_context));
          if (NULL != c_dbcsr_acc_opencl_config.contexts) {
            result = c_dbcsr_acc_opencl_set_active_device(/*master*/ 0, device_id);
            assert(EXIT_SUCCESS == result || NULL == c_dbcsr_acc_opencl_config.contexts[/*master*/ 0]);
          }
          else {
            result = EXIT_FAILURE;
          }
          c_dbcsr_acc_opencl_config.handle = 0;
          c_dbcsr_acc_opencl_config.handles = NULL;
          c_dbcsr_acc_opencl_config.storage = NULL;
#  if LIBXSMM_VERSION4(1, 17, 0, 2188) <= LIBXSMM_VERSION_NUMBER && defined(ACC_OPENCL_HANDLES_MAXCOUNT) && \
    (0 < ACC_OPENCL_HANDLES_MAXCOUNT)
          if (EXIT_SUCCESS == result) {
            c_dbcsr_acc_opencl_config.handle = ACC_OPENCL_HANDLES_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads;
            c_dbcsr_acc_opencl_config.handles = (void**)malloc(sizeof(void*) * c_dbcsr_acc_opencl_config.handle);
            c_dbcsr_acc_opencl_config.storage = malloc(sizeof(void*) * c_dbcsr_acc_opencl_config.handle);
            if (NULL != c_dbcsr_acc_opencl_config.handles && NULL != c_dbcsr_acc_opencl_config.storage) {
              libxsmm_pmalloc_init(sizeof(void*), &c_dbcsr_acc_opencl_config.handle, c_dbcsr_acc_opencl_config.handles,
                c_dbcsr_acc_opencl_config.storage);
            }
            else {
              free(c_dbcsr_acc_opencl_config.handles);
              free(c_dbcsr_acc_opencl_config.storage);
              c_dbcsr_acc_opencl_config.handles = NULL;
              c_dbcsr_acc_opencl_config.storage = NULL;
              c_dbcsr_acc_opencl_config.handle = 0;
              result = EXIT_FAILURE;
            }
          }
#  endif
          if (EXIT_SUCCESS == result) {
            const int nelements = ACC_OPENCL_STREAMS_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads;
            c_dbcsr_acc_opencl_config.streams = (void**)calloc(nelements, sizeof(void*)); /* allocate streams */
            if (NULL != c_dbcsr_acc_opencl_config.streams) { /* allocate counters */
              c_dbcsr_acc_opencl_config.stats = (cl_command_queue*)calloc(nelements, sizeof(cl_command_queue));
            }
            else {
              result = EXIT_FAILURE;
            }
          }
        }
        ACC_OPENCL_DEBUG_FPRINTF(stderr, "INFO ACC/OpenCL: started pid=%u nthreads=%i ndevices=%i (device=%i, context=%p).\n",
          libxsmm_get_pid(), c_dbcsr_acc_opencl_config.nthreads, c_dbcsr_acc_opencl_config.ndevices, device_id,
          NULL != c_dbcsr_acc_opencl_config.contexts ? ((const void*)c_dbcsr_acc_opencl_config.contexts[/*master*/ 0]) : NULL);
      }
      else { /* mark as initialized */
        c_dbcsr_acc_opencl_config.ndevices = -1;
      }
    }
    else { /* mark as initialized (ACC_OPENCL_DISABLE) */
      c_dbcsr_acc_opencl_config.ndevices = -1;
    }
    ACC_OPENCL_DEBUG_IF(0 >= c_dbcsr_acc_opencl_config.ndevices)
    ACC_OPENCL_DEBUG_FPRINTF(stderr, "INFO ACC/OpenCL: pid=%u found no devices.\n", libxsmm_get_pid());
#  if defined(__DBCSR_ACC)
    /* DBCSR shall call c_dbcsr_acc_init as well as libsmm_acc_init (since both interfaces are used).
     * Also, libsmm_acc_init may privately call c_dbcsr_acc_init (as it depends on the ACC interface).
     * The implementation of c_dbcsr_acc_init should hence be safe against "over initialization".
     * However, DBCSR only calls c_dbcsr_acc_init (and expects an implicit libsmm_acc_init).
     */
    if (EXIT_SUCCESS == result) result = libsmm_acc_init();
#  endif
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_finalize(void) {
#  if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*master*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
#  else
  int result = EXIT_SUCCESS;
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS != result)
  ACC_OPENCL_DEBUG_FPRINTF(stderr, "ERROR ACC/OpenCL: c_dbcsr_acc_finalize called in OpenMP parallel region!\n");
  if (0 != c_dbcsr_acc_opencl_config.ndevices) {
    int i;
    ACC_OPENCL_DEBUG_FPRINTF(stderr, "INFO ACC/OpenCL: stopped pid=%u.\n", libxsmm_get_pid());
    assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_DEVICES_MAXCOUNT);
#  if defined(__DBCSR_ACC)
    /* DBCSR may call c_dbcsr_acc_init as well as libsmm_acc_init() since both interface are used.
     * libsmm_acc_init may privately call c_dbcsr_acc_init (as it depends on the ACC interface).
     * The implementation of c_dbcsr_acc_init should be safe against "over initialization".
     * However, DBCSR only calls c_dbcsr_acc_init and expects an implicit libsmm_acc_init().
     */
    if (EXIT_SUCCESS == result) result = libsmm_acc_finalize();
#  endif
    if (0 != c_dbcsr_acc_opencl_config.verbosity) {
      cl_device_id device;
      int d;
      fprintf(stderr, "INFO ACC/OpenCL: pid=%u nthreads=%i", libxsmm_get_pid(), c_dbcsr_acc_opencl_config.nthreads);
      if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device(0, &device) &&
          EXIT_SUCCESS == c_dbcsr_acc_opencl_device_id(device, NULL /*devid*/, &d)) {
        fprintf(stderr, " device=%i", d);
      }
      if (NULL != c_dbcsr_acc_opencl_config.stats) {
        const int nelements = ACC_OPENCL_STREAMS_MAXCOUNT * c_dbcsr_acc_opencl_config.nthreads;
        cl_command_queue s = NULL;
        int nstreams, j;
        fprintf(stderr, " streams={");
        for (i = 0; i < nelements; i += ACC_OPENCL_STREAMS_MAXCOUNT) {
          for (j = 0, nstreams = 0; j < ACC_OPENCL_STREAMS_MAXCOUNT; ++j) {
            if (NULL != c_dbcsr_acc_opencl_config.stats[i + j]) ++nstreams;
          }
          if (0 != nstreams || 0 == i) fprintf(stderr, 0 < i ? " %i" : "%i", nstreams);
        }
        qsort(c_dbcsr_acc_opencl_config.stats, nelements, sizeof(cl_command_queue),
          c_dbcsr_acc_opencl_order_streams); /* NULL -> upper end */
        for (i = 0, nstreams = 0; i < nelements; ++i) {
          const cl_command_queue q = c_dbcsr_acc_opencl_config.stats[i];
          if (NULL != q && s != q) {
            s = q;
            ++nstreams;
          }
        }
        fprintf(stderr, "} nstreams=%i", nstreams);
      }
      fprintf(stderr, "\n");
    }
    if (NULL != c_dbcsr_acc_opencl_config.contexts) {
      for (i = 0; i < c_dbcsr_acc_opencl_config.nthreads; ++i) {
        const cl_context context = c_dbcsr_acc_opencl_config.contexts[i];
        if (NULL != context) {
          c_dbcsr_acc_opencl_config.contexts[i] = NULL;
          if (EXIT_SUCCESS == clReleaseContext(context)) {
            ACC_OPENCL_DEBUG_FPRINTF(stderr, "INFO ACC/OpenCL: released context %p (tid=%i).\n", (const void*)context, i);
          }
          ACC_OPENCL_DEBUG_ELSE ACC_OPENCL_DEBUG_FPRINTF(
            stderr, "WARNING ACC/OpenCL: releasing context %p (tid=%i) failed!\n", (const void*)context, i);
        }
      }
    }
    for (i = 0; i < ACC_OPENCL_DEVICES_MAXCOUNT; ++i) {
      const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[i];
      if (NULL != device_id) {
#  if defined(CL_VERSION_1_2)
        ACC_OPENCL_CHECK(clReleaseDevice(device_id), "release device", result);
#  endif
        /* c_dbcsr_acc_opencl_create_context scans for non-NULL devices */
        c_dbcsr_acc_opencl_config.devices[i] = NULL;
      }
    }
    /* release/reset buffers */
    free(c_dbcsr_acc_opencl_config.handles);
    free(c_dbcsr_acc_opencl_config.storage);
    free(c_dbcsr_acc_opencl_config.streams);
    free(c_dbcsr_acc_opencl_config.contexts);
    /* clear configuration */
    memset(&c_dbcsr_acc_opencl_config, 0, sizeof(c_dbcsr_acc_opencl_config));
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


void c_dbcsr_acc_clear_errors(void) {}


int c_dbcsr_acc_get_ndevices(int* ndevices) {
  int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(__DBCSR_ACC)
  /* DBCSR calls c_dbcsr_acc_get_ndevices before calling c_dbcsr_acc_init. */
  result = c_dbcsr_acc_init();
  if (EXIT_SUCCESS == result)
#  endif
  {
    if (NULL != ndevices && 0 != c_dbcsr_acc_opencl_config.ndevices) {
      *ndevices = (0 < c_dbcsr_acc_opencl_config.ndevices ? c_dbcsr_acc_opencl_config.ndevices : 0);
      result = EXIT_SUCCESS;
    }
    else {
      result = EXIT_FAILURE;
    }
  }
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device(int thread_id, cl_device_id* device) {
  int result = EXIT_SUCCESS;
  assert(0 <= thread_id && thread_id < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL != device);
  if (NULL != c_dbcsr_acc_opencl_config.contexts) {
    cl_context context = c_dbcsr_acc_opencl_config.contexts[thread_id];
#  if defined(_OPENMP)
    if (NULL == context && 0 < thread_id) { /* fallback to master's context */
      context = c_dbcsr_acc_opencl_config.contexts[/*master*/ 0];
    }
#  endif
    if (NULL != context) {
      result = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), device, NULL);
    }
    else {
      *device = NULL;
    }
  }
  else {
    *device = NULL;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_id(cl_device_id device, int* device_id, int* global_id) {
  int result = EXIT_SUCCESS, i;
  assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_DEVICES_MAXCOUNT);
  assert(NULL != device_id || NULL != global_id);
  for (i = 0; i < c_dbcsr_acc_opencl_config.ndevices; ++i) {
    if (device == c_dbcsr_acc_opencl_config.devices[i]) break;
  }
  if (i < c_dbcsr_acc_opencl_config.ndevices) {
    if (NULL != device_id) *device_id = i;
    if (NULL != global_id) {
      *global_id = i;
      for (++i; i < ACC_OPENCL_DEVICES_MAXCOUNT; ++i) {
        if (NULL != c_dbcsr_acc_opencl_config.devices[i]) {
          if (device == c_dbcsr_acc_opencl_config.devices[i]) {
            *global_id = i;
            break;
          }
        }
        else {
          break;
        }
      }
    }
  }
  else {
    if (NULL != device_id) *device_id = -1;
    if (NULL != global_id) *global_id = -1;
    if (NULL != device) result = EXIT_FAILURE;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_vendor(cl_device_id device, const char vendor[]) {
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != vendor);
  ACC_OPENCL_CHECK(
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL), "retrieve device vendor", result);
  if (EXIT_SUCCESS == result) {
    return (NULL != libxsmm_stristr(buffer, vendor) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  else {
    ACC_OPENCL_RETURN(result);
  }
}


int c_dbcsr_acc_opencl_device_name(cl_device_id device, const char match[]) {
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != match);
  ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_NAME, ACC_OPENCL_BUFFERSIZE, buffer, NULL), "retrieve device name", result);
  if (EXIT_SUCCESS == result) {
    const char* const p = libxsmm_stristr(buffer, match);
    return (NULL != p ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  else {
    ACC_OPENCL_RETURN(result);
  }
}


int c_dbcsr_acc_opencl_devuid(const char devname[], int* uid) {
  int result;
  assert(NULL != devname && NULL != uid);
  result = ('\0' != *devname ? EXIT_SUCCESS : EXIT_FAILURE);
  if (CL_SUCCESS == result) {
    char skip[ACC_OPENCL_BUFFERSIZE];
    union {
      unsigned int u;
      int i;
    } cast;
    if (2 != sscanf(devname, "%[^[][0x%xi]", skip, &cast.u)) {
      cast.u = libxsmm_hash(devname, (unsigned int)strlen(devname), 25071975 /*seed*/);
    }
    *uid = cast.i;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_uid(cl_device_id device, int* uid) {
  char buffer[ACC_OPENCL_BUFFERSIZE];
  cl_int result = clGetDeviceInfo(device, CL_DEVICE_NAME, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
  assert(NULL != uid);
  if (CL_SUCCESS == result) {
    result = c_dbcsr_acc_opencl_devuid(buffer, uid);
  }
#  if defined(OPENCL_LIBSMM_PARAMS_DEVICE)
  else {
    result = c_dbcsr_acc_opencl_devuid(OPENCL_LIBSMM_PARAMS_DEVICE, uid);
  }
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_level(
  cl_device_id device, int* level_major, int* level_minor, char cl_std[16], cl_device_type* type) {
  cl_int result = EXIT_SUCCESS;
  assert(NULL != device && (NULL != level_major || NULL != level_minor || NULL != cl_std || NULL != type));
  if (NULL != level_major || NULL != level_minor || NULL != cl_std) {
    char buffer[ACC_OPENCL_BUFFERSIZE];
    result = clGetDeviceInfo(device, CL_DEVICE_VERSION, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
    if (CL_SUCCESS == result) {
      unsigned int cl_std_level[2];
      if (2 == sscanf(buffer, "OpenCL %u.%u", cl_std_level, cl_std_level + 1)) {
        if (NULL != level_major) *level_major = (int)cl_std_level[0];
        if (NULL != level_minor) *level_minor = (int)cl_std_level[1];
        if (NULL != cl_std) {
          if (2 <= cl_std_level[0]) {
            const int nchar = LIBXSMM_SNPRINTF(cl_std, 16, "-cl-std=CL%u.0", cl_std_level[0]);
            if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
          }
          else if (1 <= cl_std_level[0]) {
            if (1 <= cl_std_level[1]) {
              const int nchar = LIBXSMM_SNPRINTF(cl_std, 16, "-cl-std=CL%u.%u", cl_std_level[0], cl_std_level[1]);
              if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
            }
            else {
              result = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
              if (CL_SUCCESS == result) {
                if (2 == sscanf(buffer, "OpenCL C %u.%u", cl_std_level, cl_std_level + 1)) {
                  const int nchar = LIBXSMM_SNPRINTF(cl_std, 16, "-cl-std=CL%u.%u", cl_std_level[0], cl_std_level[1]);
                  if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
                }
                else {
                  result = EXIT_FAILURE;
                  *cl_std = '\0';
                }
              }
              else {
                *cl_std = '\0';
              }
            }
          }
          else {
            *cl_std = '\0';
          }
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
  }
  if (NULL != type && EXIT_SUCCESS == result) {
    result = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), type, NULL);
  }
  return result;
}


int c_dbcsr_acc_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts) {
  int result = ((NULL != extnames && 0 < num_exts) ? EXIT_SUCCESS : EXIT_FAILURE);
  char extensions[ACC_OPENCL_BUFFERSIZE], buffer[ACC_OPENCL_BUFFERSIZE];
  assert(NULL != device);
  ACC_OPENCL_CHECK(
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ACC_OPENCL_BUFFERSIZE, extensions, NULL), "retrieve device extensions", result);
  if (EXIT_SUCCESS == result) {
    do {
      if (NULL != extnames[--num_exts]) {
        const char* const end = buffer + strlen(extnames[num_exts]);
        char* ext = strtok(strncpy(buffer, extnames[num_exts], ACC_OPENCL_BUFFERSIZE - 1), ACC_OPENCL_DELIMS " \t");
        for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), ACC_OPENCL_DELIMS " \t") : NULL)) {
          if (NULL == strstr(extensions, ext)) {
            return EXIT_FAILURE;
          }
        }
      }
    } while (0 < num_exts);
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_create_context(int thread_id, cl_device_id active_id) {
  cl_platform_id platform = NULL;
  cl_int result;
  assert(0 <= thread_id && thread_id < c_dbcsr_acc_opencl_config.nthreads);
  assert(NULL == c_dbcsr_acc_opencl_config.contexts[thread_id]);
  assert(0 < c_dbcsr_acc_opencl_config.ndevices);
  assert(NULL != active_id);
  result = clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
  assert(CL_SUCCESS != result || NULL != platform);
  if (CL_SUCCESS == result) {
#  if defined(NDEBUG)
    void (*const notify)(const char*, const void*, size_t, void*) = NULL;
#  else
    void (*const notify)(const char*, const void*, size_t, void*) = c_dbcsr_acc_opencl_notify;
#  endif
    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, 0 /*placeholder*/, 0 /* end of properties */
    };
    cl_context context = NULL;
    properties[1] = (long)platform;
    context = clCreateContext(properties, 1 /*num_devices*/, &active_id, notify, NULL /* user_data*/, &result);
    if (CL_SUCCESS != result && CL_INVALID_DEVICE != result) { /* retry */
      context = clCreateContext(NULL /*properties*/, 1 /*num_devices*/, &active_id, notify, NULL /* user_data*/, &result);
    }
    if (CL_SUCCESS == result) {
      assert(NULL != context);
      c_dbcsr_acc_opencl_config.contexts[thread_id] = context;
      if (0 != thread_id) {
        /* apply context to master-thread if master's context is NULL */
        LIBXSMM_ATOMIC_CMPSWP(c_dbcsr_acc_opencl_config.contexts, NULL, context, LIBXSMM_ATOMIC_RELAXED);
        assert(NULL != c_dbcsr_acc_opencl_config.contexts[/*master*/ 0]);
        ACC_OPENCL_DEBUG_IF(NULL == c_dbcsr_acc_opencl_config.contexts[/*master*/ 0])
        ACC_OPENCL_DEBUG_FPRINTF(stderr, "ERROR ACC/OpenCL: applying context to master-thread failed!\n");
      }
      if (0 != c_dbcsr_acc_opencl_config.verbosity) {
        char buffer[ACC_OPENCL_BUFFERSIZE];
        int dev = 0, uid;
        if (CL_SUCCESS == clGetDeviceInfo(active_id, CL_DEVICE_NAME, ACC_OPENCL_BUFFERSIZE, buffer, NULL) &&
            EXIT_SUCCESS == c_dbcsr_acc_opencl_device_id(active_id, NULL /*devid*/, &dev) &&
            EXIT_SUCCESS == c_dbcsr_acc_opencl_device_uid(active_id, &uid)) {
          fprintf(stderr, "INFO ACC/OpenCL: ndevices=%i device%i=\"%s\" uid=%#08x\n", c_dbcsr_acc_opencl_config.ndevices, dev,
            buffer, uid);
        }
      }
    }
    else if (CL_INVALID_DEVICE == result && EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia")) {
      fprintf(stderr, "WARNING ACC/OpenCL: if MPI-ranks target the same device in exclusive mode,\n"
                      "                    SMI must be used to enable sharing the device.\n");
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_set_active_device(int thread_id, int device_id) {
  int result = EXIT_SUCCESS;
  cl_device_id active_id;
  assert(0 <= thread_id && thread_id < c_dbcsr_acc_opencl_config.nthreads);
  assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_DEVICES_MAXCOUNT);
  if (0 <= device_id && device_id < c_dbcsr_acc_opencl_config.ndevices) {
    assert(NULL != c_dbcsr_acc_opencl_config.contexts);
    active_id = c_dbcsr_acc_opencl_config.devices[device_id];
    if (NULL != active_id) {
#  if defined(_OPENMP)
#    pragma omp critical(c_dbcsr_acc_set_active_device)
#  endif
      {
        int inherit_id = 0;
        const cl_context context = c_dbcsr_acc_opencl_device_context(active_id, &inherit_id);
        const cl_context inherit = c_dbcsr_acc_opencl_config.contexts[inherit_id];
        if (NULL != context) {
          if (context != inherit) {
            if (NULL != inherit) {
              c_dbcsr_acc_opencl_config.contexts[inherit_id] = NULL;
              result = clReleaseContext(inherit);
            }
            else if (thread_id != inherit_id) {
              c_dbcsr_acc_opencl_config.contexts[inherit_id] = context;
              result = clRetainContext(context);
            }
          }
        }
        else if (NULL == c_dbcsr_acc_opencl_config.contexts[thread_id]) {
          result = c_dbcsr_acc_opencl_create_context(thread_id, active_id);
          if (EXIT_SUCCESS == result) {
            if (NULL /*context*/ != inherit) {
              c_dbcsr_acc_opencl_config.contexts[inherit_id] = c_dbcsr_acc_opencl_config.contexts[thread_id];
              result = clReleaseContext(inherit);
              ACC_OPENCL_DEBUG_IF(EXIT_SUCCESS == result) {
                ACC_OPENCL_DEBUG_FPRINTF(
                  stderr, "INFO ACC/OpenCL: released context %p (tid=%i).\n", (const void*)inherit, thread_id);
              }
              ACC_OPENCL_DEBUG_ELSE {
                ACC_OPENCL_DEBUG_FPRINTF(
                  stderr, "ERROR ACC/OpenCL: releasing context %p (tid=%i) failed!\n", (const void*)inherit, thread_id);
              }
            }
            ACC_OPENCL_DEBUG_IF(/*master*/ 0 != thread_id)
            ACC_OPENCL_DEBUG_FPRINTF(stderr, "INFO ACC/OpenCL: created device context %p (tid=%i, device=%i).\n",
              (const void*)c_dbcsr_acc_opencl_config.contexts[thread_id], thread_id, device_id);
          }
        }
        if (EXIT_SUCCESS == result) { /* update c_dbcsr_acc_opencl_config.devinfo */
#  if defined(ACC_OPENCL_SVM)
          {
            const char* const env_svm = getenv("ACC_OPENCL_SVM");
            int level_major = 0;
            c_dbcsr_acc_opencl_config.devinfo.svm_interop =
              (NULL == env_svm || 0 != atoi(env_svm)) && (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_level(active_id, &level_major,
                                                                            NULL /*level_minor*/, NULL /*cl_std*/, NULL /*type*/) &&
                                                           2 <= level_major);
          }
#  endif
          if (CL_SUCCESS != clGetDeviceInfo(active_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool),
                              &c_dbcsr_acc_opencl_config.devinfo.unified, NULL)) {
            c_dbcsr_acc_opencl_config.devinfo.unified = CL_FALSE;
          }
          if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "intel")) {
            if (EXIT_SUCCESS != c_dbcsr_acc_opencl_device_uid(active_id, &c_dbcsr_acc_opencl_config.devinfo.intel_id)) {
              c_dbcsr_acc_opencl_config.devinfo.intel_id = -1;
            }
          }
          else {
            c_dbcsr_acc_opencl_config.devinfo.intel_id = 0;
          }
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_set_active_device(int device_id) {
  int result;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
  assert(0 != c_dbcsr_acc_opencl_config.ndevices);
  result = c_dbcsr_acc_opencl_set_active_device(ACC_OPENCL_OMP_TID(), device_id);
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_synchronize(int thread_id) {
  int result = EXIT_SUCCESS;
  if (0 == (4 & c_dbcsr_acc_opencl_config.flush)) {
    void** const streams = c_dbcsr_acc_opencl_config.streams + ACC_OPENCL_STREAMS_MAXCOUNT * thread_id;
    int i = 0;
    assert(0 <= thread_id && thread_id < c_dbcsr_acc_opencl_config.nthreads);
    assert(NULL != c_dbcsr_acc_opencl_config.streams);
    for (; i < ACC_OPENCL_STREAMS_MAXCOUNT; ++i) {
      void* const stream = streams[i];
      if (NULL != stream) {
        result = c_dbcsr_acc_stream_sync(stream);
        if (EXIT_SUCCESS != result) break;
      }
      else {
        break;
      }
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_device_synchronize(void) {
  int result = EXIT_SUCCESS;
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  int routine_handle;
  static const char* const routine_name_ptr = LIBXSMM_FUNCNAME;
  static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - 1;
  c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
#  endif
#  if defined(_OPENMP)
  if (1 < omp_get_num_threads()) {
    result = c_dbcsr_acc_opencl_device_synchronize(omp_get_thread_num());
  }
  else {
    int i;
#    pragma omp parallel for private(i)
    for (i = 0; i < c_dbcsr_acc_opencl_config.nthreads; ++i) {
      ACC_OPENCL_EXPECT(EXIT_SUCCESS, c_dbcsr_acc_opencl_device_synchronize(i));
    }
  }
#  else
  result = c_dbcsr_acc_opencl_device_synchronize(/*master*/ 0);
#  endif
#  if defined(__DBCSR_ACC) && defined(ACC_OPENCL_PROFILE)
  c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_wgsize(cl_device_id device, cl_kernel kernel, size_t* max_value, size_t* preferred_multiple) {
  int result = (NULL != device && (NULL != preferred_multiple || NULL != max_value)) ? EXIT_SUCCESS : EXIT_FAILURE;
  if (NULL != kernel) { /* kernel-specific */
    if (NULL != max_value) {
      ACC_OPENCL_CHECK(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), max_value, NULL),
        "query maximum WG-size of kernel", result);
    }
    if (NULL != preferred_multiple) {
      ACC_OPENCL_CHECK(clGetKernelWorkGroupInfo(
                         kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), preferred_multiple, NULL),
        "query preferred multiple of WG-size of kernel", result);
    }
  }
  else { /* device-specific */
    if (NULL != max_value) {
      ACC_OPENCL_CHECK(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), max_value, NULL),
        "query maximum WG-size of device", result);
    }
    if (NULL != preferred_multiple) {
#  if defined(CL_VERSION_3_0)
      ACC_OPENCL_CHECK(
        clGetDeviceInfo(device, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), preferred_multiple, NULL),
        "query preferred multiple of WG-size of device", result);
#  else
      *preferred_multiple = 1;
#  endif
    }
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_kernel(const char source[], const char kernel_name[], const char build_params[], const char build_options[],
  const char try_build_options[], int* try_ok, const char* const extnames[], int num_exts, cl_kernel* kernel) {
  char buffer[ACC_OPENCL_BUFFERSIZE] = "", cl_std[16];
  char buffer_name[ACC_OPENCL_MAXSTRLEN * 2];
  cl_device_id active_id = NULL;
  cl_int result = c_dbcsr_acc_opencl_device(ACC_OPENCL_OMP_TID(), &active_id);
  int level_major, level_minor, ok = EXIT_SUCCESS;
  assert(NULL != source && NULL != kernel);
  assert(NULL != kernel_name && '\0' != *kernel_name);
  if (EXIT_SUCCESS == result) {
    result = c_dbcsr_acc_opencl_device_level(active_id, &level_major, &level_minor, cl_std, NULL /*type*/);
  }
  if (EXIT_SUCCESS == result) {
    const cl_context context = c_dbcsr_acc_opencl_context();
    const char* ext_source = source;
    size_t size_src = strlen(source);
    cl_program program = NULL;
    if (NULL != extnames) {
      int n = num_exts, nflat = 0;
      size_t size_ext = 0;
      for (; 0 < n; --n) {
        if (NULL != extnames[n - 1]) {
          const char* const end = buffer + strlen(extnames[n - 1]);
          char* ext = strtok(strncpy(buffer, extnames[n - 1], ACC_OPENCL_BUFFERSIZE - 1), ACC_OPENCL_DELIMS " \t");
          for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), ACC_OPENCL_DELIMS " \t") : NULL), ++nflat) {
            size_ext += strlen(ext);
          }
        }
      }
      if (0 < size_ext && 0 < nflat) {
        const char* const enable_ext = "#pragma OPENCL EXTENSION %s : enable\n";
        const size_t size_src_ext = size_src + size_ext + nflat * (strlen(enable_ext) - 2 /*%s*/);
        char* const ext_source_buffer = (char*)libxsmm_aligned_scratch(size_src_ext + 1 /*terminator*/, 0 /*auto-align*/);
        if (NULL != ext_source_buffer) {
          for (n = 0; 0 < num_exts; --num_exts) {
            if (NULL != extnames[num_exts - 1]) {
              const char* const end = buffer_name + strlen(extnames[num_exts - 1]);
              char* ext = strtok(
                strncpy(buffer_name, extnames[num_exts - 1], ACC_OPENCL_MAXSTRLEN * 2 - 1), ACC_OPENCL_DELIMS " \t");
              for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), ACC_OPENCL_DELIMS " \t") : NULL)) {
                const char* line = source;
                for (;;) {
                  if (2 !=
                      sscanf(line, "#pragma OPENCL EXTENSION %[^: ]%*[: ]%[^\n]", buffer, buffer + ACC_OPENCL_BUFFERSIZE / 2)) {
                    line = NULL;
                    break;
                  }
                  else if (0 == strncmp(buffer, ext, ACC_OPENCL_BUFFERSIZE / 2) &&
                           0 == strncmp(buffer + ACC_OPENCL_BUFFERSIZE / 2, "enable", ACC_OPENCL_BUFFERSIZE / 2)) {
                    break;
                  }
                  line = strchr(line, '\n');
                  if (NULL != line) {
                    ++line;
                  }
                  else {
                    break;
                  }
                }
#  if !defined(NDEBUG)
                if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(active_id, (const char**)&ext, 1))
#  endif
                { /* NDEBUG: assume given extension is supported (confirmed upfront) */
                  if (NULL == line) { /* extension is not already part of source */
                    n += LIBXSMM_SNPRINTF(ext_source_buffer + n, size_src_ext + 1 /*terminator*/ - n, enable_ext, ext);
                  }
                }
#  if !defined(NDEBUG)
                else if (0 != strcmp("cl_intel_global_float_atomics", ext)) {
                  fprintf(stderr, "WARNING ACC/OpenCL: extension \"%s\" is not supported.\n", ext);
                }
#  endif
              }
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
#  if defined(ACC_OPENCL_CPPBIN)
    if (0 != c_dbcsr_acc_opencl_config.dump) {
      int nchar = LIBXSMM_SNPRINTF(buffer_name, sizeof(buffer_name), "/tmp/.%s.XXXXXX", kernel_name);
      if (0 < nchar && (int)sizeof(buffer_name) > nchar) {
        FILE* const file_cpp = fopen(ACC_OPENCL_CPPBIN, "rb");
        const char* sed_pattern = "";
#    if defined(ACC_OPENCL_SEDBIN)
        FILE* const file_sed = fopen(ACC_OPENCL_SEDBIN, "rb");
        if (NULL != file_sed) {
          sed_pattern = "| " ACC_OPENCL_SEDBIN " '/^[[:space:]]*\\(\\/\\/.*\\)*$/d'";
          fclose(file_sed); /* existence-check */
        }
#    endif
        if (NULL != file_cpp) {
          const int file_src = mkstemp(buffer_name);
          fclose(file_cpp); /* existence-check */
          if (0 <= file_src) {
            if (size_src == (size_t)write(file_src, ext_source, size_src)) {
              nchar = LIBXSMM_SNPRINTF(buffer, sizeof(buffer),
                ACC_OPENCL_CPPBIN " -P -C -nostdinc -D__OPENCL_VERSION__=%u %s %s %s %s > %s.cl",
                100 * level_major + 10 * level_minor,
                EXIT_SUCCESS != c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia") ? "" : "-D__NV_CL_C_VERSION",
                NULL != build_params ? build_params : "", buffer_name, sed_pattern, kernel_name);
              if (0 < nchar && (int)sizeof(buffer) > nchar) {
                if (EXIT_SUCCESS == system(buffer)) {
                  nchar = LIBXSMM_SNPRINTF(buffer, sizeof(buffer), "%s.cl", kernel_name);
                  if (0 < nchar && (int)sizeof(buffer) > nchar) {
                    FILE* const file = fopen(buffer, "r");
                    if (NULL != file) {
                      const long int size = (EXIT_SUCCESS == fseek(file, 0 /*offset*/, SEEK_END) ? ftell(file) : 0);
                      char* const src = (char*)(EXIT_SUCCESS == fseek(file, 0 /*offset*/, SEEK_SET)
                                                  ? libxsmm_aligned_scratch(size + 1 /*terminator*/, 0 /*auto-align*/)
                                                  : NULL);
                      if (NULL != src) {
                        if ((size_t)size == fread(src, 1 /*sizeof(char)*/, size /*count*/, file)) {
                          if (source != ext_source) libxsmm_free((void*)ext_source);
                          src[size] = '\0';
                          ext_source = src;
                        }
                        else {
                          libxsmm_free(src);
                        }
                      }
                      ACC_OPENCL_EXPECT(EXIT_SUCCESS, fclose(file));
                    }
                  }
                }
              }
              buffer[0] = '\0'; /* reset to empty */
            }
            ACC_OPENCL_EXPECT(EXIT_SUCCESS, unlink(buffer_name));
            ACC_OPENCL_EXPECT(EXIT_SUCCESS, close(file_src));
          }
        }
      }
    }
#  endif
    program = clCreateProgramWithSource(context, 1 /*nlines*/, &ext_source, NULL, &result);
    if (CL_SUCCESS == result) {
      int nchar = LIBXSMM_SNPRINTF(buffer, sizeof(buffer), "%s %s %s %s", cl_std, NULL != build_options ? build_options : "",
        NULL != try_build_options ? try_build_options : "", NULL != build_params ? build_params : "");
      assert(NULL != program);
      result = ((0 < nchar && (int)sizeof(buffer) > nchar)
                  ? clBuildProgram(program, 1 /*num_devices*/, &active_id, buffer, NULL /*callback*/, NULL /*user_data*/)
                  : EXIT_FAILURE);
      if (CL_SUCCESS != result && NULL != try_build_options && '\0' != *try_build_options) {
        nchar = LIBXSMM_SNPRINTF(buffer, sizeof(buffer), "%s %s %s", cl_std, NULL != build_options ? build_options : "",
          NULL != build_params ? build_params : "");
        ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
        /* recreate program after building it failed (unclean state) */
        program = clCreateProgramWithSource(context, 1 /*nlines*/, &ext_source, NULL, &result);
        if (CL_SUCCESS == result && 0 < nchar && (int)sizeof(buffer) > nchar) {
          assert(NULL != program);
          result = clBuildProgram(program, 1 /*num_devices*/, &active_id, buffer, NULL /*callback*/, NULL /*user_data*/);
        }
        ok = EXIT_FAILURE;
      }
      if (source != ext_source) libxsmm_free((void*)ext_source);
      buffer[0] = '\0'; /* reset to empty */
      if (CL_SUCCESS == result) {
        *kernel = clCreateKernel(program, kernel_name, &result);
        if (CL_SUCCESS == result) {
          assert(NULL != *kernel);
          if (2 <= c_dbcsr_acc_opencl_config.dump || 0 > c_dbcsr_acc_opencl_config.dump) {
            unsigned char* binary = NULL;
            size_t size;
            binary = (unsigned char*)(CL_SUCCESS == clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL)
                                        ? libxsmm_aligned_scratch(size, 0 /*auto-align*/)
                                        : NULL);
            if (NULL != binary) {
              result = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);
              if (CL_SUCCESS == result) {
                FILE* file;
                nchar = LIBXSMM_SNPRINTF(buffer, sizeof(buffer), "%s.dump", kernel_name);
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
              else { /* error: querying program binary */
                ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
                ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseKernel(*kernel));
              }
              libxsmm_free(binary);
            }
            else {
              ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
              ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseKernel(*kernel));
              result = EXIT_FAILURE;
            }
          }
        }
        else { /* error: creating kernel */
          ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
        }
      }
      else {
        ACC_OPENCL_EXPECT(
          CL_SUCCESS, clGetProgramBuildInfo(program, active_id, CL_PROGRAM_BUILD_LOG, ACC_OPENCL_BUFFERSIZE, buffer, NULL));
        ACC_OPENCL_EXPECT(CL_SUCCESS, clReleaseProgram(program));
      }
    }
    else if (source != ext_source) { /* error: creating program */
      libxsmm_free((void*)ext_source);
    }
  }
#  if !defined(NDEBUG)
  if (EXIT_SUCCESS != result) *kernel = NULL;
#  endif
  if (NULL != try_ok) *try_ok = result | ok;
  ACC_OPENCL_RETURN_CAUSE(result, buffer);
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
