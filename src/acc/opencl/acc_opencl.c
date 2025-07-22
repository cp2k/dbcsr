/*------------------------------------------------------------------------------------------------*/
/* Copyright (C) by the DBCSR developers group - All rights reserved                              */
/* This file is part of the DBCSR library.                                                        */
/*                                                                                                */
/* For information on the license, see the LICENSE file.                                          */
/* For further information please visit https://dbcsr.cp2k.org                                    */
/* SPDX-License-Identifier: BSD-3-Clause                                                          */
/*------------------------------------------------------------------------------------------------*/
#if defined(__OPENCL)
#  include "acc_opencl.h"
#  include <string.h>
#  include <limits.h>
#  include <ctype.h>
#  include <math.h>
#  if defined(_WIN32)
#    include <windows.h>
#    include <process.h>
#  else
#    include <unistd.h>
#    include <errno.h>
#    include <glob.h>
#  endif
#  if defined(__DBCSR_ACC)
#    include "../acc_libsmm.h"
#  endif
#  include <fcntl.h>
#  include <sys/stat.h>
#  if !defined(S_ISDIR) && defined(S_IFMT) && defined(S_IFDIR)
#    define S_ISDIR(A) ((S_IFMT & (A)) == S_IFDIR)
#  endif
#  if !defined(S_IREAD)
#    define S_IREAD S_IRUSR
#  endif
#  if !defined(S_IWRITE)
#    define S_IWRITE S_IWUSR
#  endif

#  if !defined(ACC_OPENCL_NLOCKS)
#    define ACC_OPENCL_NLOCKS 4
#  endif
#  if !defined(ACC_OPENCL_TEMPDIR) && 1
#    define ACC_OPENCL_TEMPDIR "/tmp"
#  endif
#  if !defined(ACC_OPENCL_CACHE_DID) && 1
#    define ACC_OPENCL_CACHE_DID
#  endif
#  if !defined(ACC_OPENCL_CACHE_DIR) && 0
#    define ACC_OPENCL_CACHE_DIR ".cl_cache"
#  endif
#  if !defined(ACC_OPENCL_CPPBIN) && 1
#    define ACC_OPENCL_CPPBIN "/usr/bin/cpp"
#  endif
#  if !defined(ACC_OPENCL_SEDBIN) && 1
#    define ACC_OPENCL_SEDBIN "/usr/bin/sed"
#  endif
/* disabled: let MPI runtime come up before */
#  if !defined(ACC_OPENCL_PREINIT) && 0
#    define ACC_OPENCL_PREINIT
#  endif
/* attempt to enable command aggregation */
#  if !defined(ACC_OPENCL_CMDAGR) && 1
#    define ACC_OPENCL_CMDAGR
#  endif
#  if !defined(ACC_OPENCL_NCCS) && 1
#    define ACC_OPENCL_NCCS 0
#  endif


#  if defined(__cplusplus)
extern "C" {
#  endif

char c_dbcsr_acc_opencl_locks[ACC_OPENCL_CACHELINE * ACC_OPENCL_NLOCKS];
/* global configuration discovered during initialization */
c_dbcsr_acc_opencl_config_t c_dbcsr_acc_opencl_config;

#  if defined(ACC_OPENCL_CACHE_DID)
int c_dbcsr_acc_opencl_active_id;
#  endif


void c_dbcsr_acc_opencl_notify(const char /*errinfo*/[], const void* /*private_info*/, size_t /*cb*/, void* /*user_data*/);
void c_dbcsr_acc_opencl_notify(const char errinfo[], const void* private_info, size_t cb, void* user_data) {
  LIBXSMM_UNUSED(private_info);
  LIBXSMM_UNUSED(cb);
  LIBXSMM_UNUSED(user_data);
  fprintf(stderr, "ERROR ACC/OpenCL: %s\n", errinfo);
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
  ACC_OPENCL_EXPECT(EXIT_SUCCESS == clGetDeviceInfo(*a, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_a, NULL));
  ACC_OPENCL_EXPECT(EXIT_SUCCESS == clGetDeviceInfo(*b, CL_DEVICE_TYPE, sizeof(cl_device_type), &type_b, NULL));
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
        ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, &unified_a));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, &unified_b));
        if ((0 == unified_a && 0 == unified_b) || (0 != unified_a && 0 != unified_b)) {
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        /* discrete GPU goes in front */
        else if (0 == unified_b) return 1;
        else return -1;
      }
      else return -1;
    }
    else if (CL_DEVICE_TYPE_GPU & type_b) {
      return 1;
    }
    else {
      if (CL_DEVICE_TYPE_CPU & type_a) {
        if (CL_DEVICE_TYPE_CPU & type_b) {
          size_t size_a, size_b;
          ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
          ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
          return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
        }
        else return -1;
      }
      else if (CL_DEVICE_TYPE_CPU & type_b) {
        return 1;
      }
      else {
        size_t size_a = 0, size_b = 0;
        ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devmem(*a, NULL, &size_a, NULL, NULL));
        ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_opencl_info_devmem(*b, NULL, &size_b, NULL, NULL));
        return (size_a < size_b ? 1 : (size_a != size_b ? -1 : (a < b ? -1 : 1)));
      }
    }
  }
}


void c_dbcsr_acc_opencl_configure(void) {
  const char* const env_rank = (NULL != getenv("PMI_RANK") ? getenv("PMI_RANK") : getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
  const char* const env_nranks = getenv("MPI_LOCALNRANKS"); /* TODO */
  const char *const env_devsplit = getenv("ACC_OPENCL_DEVSPLIT"), *const env_nlocks = getenv("ACC_OPENCL_NLOCKS");
  const char *const env_verbose = getenv("ACC_OPENCL_VERBOSE"), *const env_dump_acc = getenv("ACC_OPENCL_DUMP");
  const char *const env_debug = getenv("ACC_OPENCL_DEBUG"), *const env_profile = getenv("ACC_OPENCL_PROFILE");
  const char* const env_dump = (NULL != env_dump_acc ? env_dump_acc : getenv("IGC_ShaderDumpEnable"));
  const char *const env_neo = getenv("NEOReadDebugKeys"), *const env_wa = getenv("ACC_OPENCL_WA");
  static char neo_enable_debug_keys[] = "NEOReadDebugKeys=1";
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  const char* const env_priority = getenv("ACC_OPENCL_PRIORITY");
#  endif
#  if defined(ACC_OPENCL_NCCS)
  const char* const env_nccs = getenv("ACC_OPENCL_NCCS");
  const int nccs = (NULL == env_nccs ? ACC_OPENCL_NCCS : atoi(env_nccs));
#  endif
#  if defined(ACC_OPENCL_XHINTS)
  const char* const env_xhints = (ACC_OPENCL_XHINTS);
  const int xhints_default = 1 + 2 + 4 + 8;
#  else
  const char* const env_xhints = NULL;
  const int xhints_default = 0;
#  endif
#  if defined(ACC_OPENCL_ASYNC)
  const char* const env_async = (ACC_OPENCL_ASYNC);
  const int async_default = 1 + 2;
#  else
  const char* const env_async = NULL;
  const int async_default = 0;
#  endif
  const int nlocks = (NULL == env_nlocks ? 1 /*default*/ : atoi(env_nlocks));
  const int neo = (NULL == env_neo ? 1 : atoi(env_neo));
  int i;
#  if defined(_OPENMP)
  const int max_threads = omp_get_max_threads(), num_threads = omp_get_num_threads();
  memset(&c_dbcsr_acc_opencl_config, 0, sizeof(c_dbcsr_acc_opencl_config));
  c_dbcsr_acc_opencl_config.nthreads = (num_threads < max_threads ? max_threads : num_threads);
#  else
  memset(&c_dbcsr_acc_opencl_config, 0, sizeof(c_dbcsr_acc_opencl_config));
  c_dbcsr_acc_opencl_config.nthreads = 1;
#  endif
  assert(NULL == c_dbcsr_acc_opencl_config.lock_main); /* test condition to avoid initializing multiple times */
  libxsmm_init(); /* before using LIBXSMM's functionality */
  c_dbcsr_acc_opencl_config.nranks = LIBXSMM_MAX(NULL != env_nranks ? atoi(env_nranks) : 1, 1);
  c_dbcsr_acc_opencl_config.nrank = (NULL != env_rank ? atoi(env_rank) : 0) % c_dbcsr_acc_opencl_config.nranks;
  assert(sizeof(ACC_OPENCL_LOCKTYPE) <= ACC_OPENCL_CACHELINE);
  for (i = 0; i < ACC_OPENCL_NLOCKS; ++i) {
    ACC_OPENCL_INIT((ACC_OPENCL_LOCKTYPE*)(c_dbcsr_acc_opencl_locks + ACC_OPENCL_CACHELINE * i));
  }
  c_dbcsr_acc_opencl_config.lock_main = (ACC_OPENCL_LOCKTYPE*)c_dbcsr_acc_opencl_locks;
  c_dbcsr_acc_opencl_config.lock_memory = /* 2nd lock-domain */
    (1 < LIBXSMM_MIN(nlocks, ACC_OPENCL_NLOCKS) ? ((ACC_OPENCL_LOCKTYPE*)(c_dbcsr_acc_opencl_locks + ACC_OPENCL_CACHELINE * 1))
                                                : c_dbcsr_acc_opencl_config.lock_main);
  c_dbcsr_acc_opencl_config.lock_stream = /* 3rd lock-domain */
    (2 < LIBXSMM_MIN(nlocks, ACC_OPENCL_NLOCKS) ? ((ACC_OPENCL_LOCKTYPE*)(c_dbcsr_acc_opencl_locks + ACC_OPENCL_CACHELINE * 2))
                                                : c_dbcsr_acc_opencl_config.lock_main);
  c_dbcsr_acc_opencl_config.lock_event = /* 4th lock-domain */
    (3 < LIBXSMM_MIN(nlocks, ACC_OPENCL_NLOCKS) ? ((ACC_OPENCL_LOCKTYPE*)(c_dbcsr_acc_opencl_locks + ACC_OPENCL_CACHELINE * 3))
                                                : c_dbcsr_acc_opencl_config.lock_main);
  c_dbcsr_acc_opencl_config.verbosity = (NULL == env_verbose ? 0 : atoi(env_verbose));
  c_dbcsr_acc_opencl_config.devsplit = (NULL == env_devsplit ? (/*1 < c_dbcsr_acc_opencl_config.nranks ? -1 :*/ 0)
                                                             : atoi(env_devsplit));
#  if defined(ACC_OPENCL_STREAM_PRIORITIES)
  c_dbcsr_acc_opencl_config.priority = (NULL == env_priority ? /*default*/ 3 : atoi(env_priority));
#  endif
  c_dbcsr_acc_opencl_config.profile = (NULL == env_profile ? /*default*/ 0 : atoi(env_profile));
  c_dbcsr_acc_opencl_config.xhints = (NULL == env_xhints ? xhints_default : atoi(env_xhints));
  c_dbcsr_acc_opencl_config.async = (NULL == env_async ? async_default : atoi(env_async));
  c_dbcsr_acc_opencl_config.dump = (NULL == env_dump ? /*default*/ 0 : atoi(env_dump));
  c_dbcsr_acc_opencl_config.debug = (NULL == env_debug ? c_dbcsr_acc_opencl_config.dump : atoi(env_debug));
  c_dbcsr_acc_opencl_config.wa = neo * (NULL == env_wa ? ((1 != c_dbcsr_acc_opencl_config.devsplit ? 0 : 1) + (2 + 4 + 8))
                                                       : atoi(env_wa));
#  if defined(ACC_OPENCL_CACHE_DIR)
  { /* environment is populated before touching the compute runtime */
    const char *const env_cache = getenv("ACC_OPENCL_CACHE"), *env_cachedir = getenv("NEO_CACHE_DIR");
    int cache = (NULL == env_cache ? 0 : atoi(env_cache));
    struct stat cachedir;
    if (0 == cache) {
      if (stat(ACC_OPENCL_CACHE_DIR, &cachedir) == 0 && S_ISDIR(cachedir.st_mode)) cache = 1;
      else if (stat(ACC_OPENCL_TEMPDIR "/" ACC_OPENCL_CACHE_DIR, &cachedir) == 0 && S_ISDIR(cachedir.st_mode)) cache = 2;
    }
    if (1 == cache) {
      static char neo_cachedir[] = "NEO_CACHE_DIR=" ACC_OPENCL_CACHE_DIR;
      static char ocl_cachedir[] = "cl_cache_dir=" ACC_OPENCL_CACHE_DIR;
      ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(neo_cachedir)); /* putenv before entering OpenCL */
      ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(ocl_cachedir)); /* putenv before entering OpenCL */
      env_cachedir = ACC_OPENCL_CACHE_DIR;
    }
#    if defined(ACC_OPENCL_TEMPDIR)
    else if (NULL == env_cachedir) { /* code-path entered by default */
      if (NULL == env_cache || 0 != cache) { /* customize NEO_CACHE_DIR unless ACC_OPENCL_CACHE=0 */
        static char neo_cachedir[] = "NEO_CACHE_DIR=" ACC_OPENCL_TEMPDIR "/" ACC_OPENCL_CACHE_DIR;
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(neo_cachedir)); /* putenv before entering OpenCL */
        env_cachedir = ACC_OPENCL_TEMPDIR "/" ACC_OPENCL_CACHE_DIR;
      }
      if (0 != cache) { /* legacy-NEO is treated with explicit opt-in */
        static char ocl_cachedir[] = "cl_cache_dir=" ACC_OPENCL_TEMPDIR "/" ACC_OPENCL_CACHE_DIR;
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(ocl_cachedir)); /* putenv before entering OpenCL */
      }
    }
#    endif
    if (NULL != env_cachedir) {
#    if defined(_WIN32)
      LIBXSMM_UNUSED(env_cachedir);
#    else
#      if defined(S_IRWXU) && defined(S_IRGRP) && defined(S_IXGRP) && defined(S_IROTH) && defined(S_IXOTH)
      const int mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
#      else
      const int mode = 0xFFFFFFFF;
#      endif
      ACC_OPENCL_EXPECT(0 == mkdir(env_cachedir, mode) || EEXIST == errno); /* soft-error */
#    endif
    }
  }
#  endif
#  if defined(ACC_OPENCL_NCCS)
  if (0 != nccs && NULL == getenv("ZEX_NUMBER_OF_CCS")) {
    static char zex_nccs[ACC_OPENCL_MAXNDEVS * 8 + 32] = "ZEX_NUMBER_OF_CCS=";
    const int mode = ((1 == nccs || 2 == nccs) ? nccs : 4);
    int j = strlen(zex_nccs);
    for (i = 0; i < ACC_OPENCL_MAXNDEVS; ++i) {
      const char* const istr = (0 < i ? ",%u:%i" : "%u:%i");
      const int n = LIBXSMM_SNPRINTF(zex_nccs + j, sizeof(zex_nccs) - j, istr, i, mode);
      if (0 < n) j += n;
      else {
        j = 0;
        break;
      }
    }
    if (0 < j && 0 == LIBXSMM_PUTENV(zex_nccs) && /* populate before touching the compute runtime */
        (2 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity))
    {
      fprintf(stderr, "INFO ACC/OpenCL: support multiple separate compute command streamers (%i-CCS mode)\n", mode);
    }
  }
#  endif
  if (0 != neo && (NULL != env_neo || 0 == LIBXSMM_PUTENV(neo_enable_debug_keys))) {
    if ((1 + 2 + 4) & c_dbcsr_acc_opencl_config.wa) {
      static char a[] = "ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE", b[] = "EnableRecoverablePageFaults=0";
      static char c[] = "DirectSubmissionOverrideBlitterSupport=0", *const apply[] = {a, b, c};
      if ((1 & c_dbcsr_acc_opencl_config.wa) && NULL == getenv("ZE_FLAT_DEVICE_HIERARCHY")) {
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(apply[0]));
      }
      if ((2 & c_dbcsr_acc_opencl_config.wa) && NULL == getenv("EnableRecoverablePageFaults")) {
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(apply[1]));
      }
      if ((4 & c_dbcsr_acc_opencl_config.wa) && NULL == getenv("DirectSubmissionOverrideBlitterSupport")) {
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(apply[2]));
      }
    }
    if (NULL == getenv("DisableScratchPages")) {
      if (0 == c_dbcsr_acc_opencl_config.debug) {
        static char a[] = "DisableScratchPages=0", *const apply[] = {a};
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(apply[0]));
      }
      else {
        static char a[] = "DisableScratchPages=1", *const apply[] = {a};
        ACC_OPENCL_EXPECT(0 == LIBXSMM_PUTENV(apply[0]));
      }
    }
  }
}


int c_dbcsr_acc_init(void) {
#  if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*main*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
#  else
  int result = EXIT_SUCCESS;
#  endif
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (NULL == c_dbcsr_acc_opencl_config.lock_main) { /* avoid to configure multiple times */
    c_dbcsr_acc_opencl_configure();
  }
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  else
  if (NULL == c_dbcsr_acc_opencl_config.lock_main) { /* avoid to configure multiple times */
    c_dbcsr_acc_opencl_configure();
  }
#  endif
  /* eventually touch OpenCL/compute runtime after configure */
  if (0 == c_dbcsr_acc_opencl_config.ndevices && EXIT_SUCCESS == result) { /* avoid to initialize multiple times */
    char buffer[ACC_OPENCL_BUFFERSIZE];
    cl_platform_id platforms[ACC_OPENCL_MAXNDEVS] = {NULL};
    cl_device_id devices[ACC_OPENCL_MAXNDEVS];
    cl_device_type type = CL_DEVICE_TYPE_ALL;
    cl_uint nplatforms = 0, ndevices = 0, i;
    const char* const env_devmatch = getenv("ACC_OPENCL_DEVMATCH");
    const char* const env_devtype = getenv("ACC_OPENCL_DEVTYPE");
    const char* const env_device = getenv("ACC_OPENCL_DEVICE");
    char* const env_devids = getenv("ACC_OPENCL_DEVIDS");
    int device_id = (NULL == env_device ? 0 : atoi(env_device));
#  if defined(ACC_OPENCL_CACHE_DID)
    assert(0 == c_dbcsr_acc_opencl_active_id);
#  endif
    if (EXIT_SUCCESS != c_dbcsr_acc_opencl_device_uid(NULL /*device*/, env_devmatch, &c_dbcsr_acc_opencl_config.devmatch)) {
      c_dbcsr_acc_opencl_config.devmatch = 1;
    }
    if (EXIT_SUCCESS == clGetPlatformIDs(0, NULL, &nplatforms) && 0 < nplatforms) {
      ACC_OPENCL_CHECK(result, clGetPlatformIDs(nplatforms <= ACC_OPENCL_MAXNDEVS ? nplatforms : ACC_OPENCL_MAXNDEVS, platforms, 0),
        "retrieve platform ids");
    }
    if (EXIT_SUCCESS == result) {
      if (NULL != env_devtype && '\0' != *env_devtype) {
        if (NULL != LIBXSMM_STRISTR(env_devtype, "gpu")) {
          type = CL_DEVICE_TYPE_GPU;
        }
        else if (NULL != LIBXSMM_STRISTR(env_devtype, "cpu")) {
          type = CL_DEVICE_TYPE_CPU;
        }
        else if (NULL != LIBXSMM_STRISTR(env_devtype, "acc") || NULL != LIBXSMM_STRISTR(env_devtype, "other")) {
          type = CL_DEVICE_TYPE_ACCELERATOR;
        }
        else {
          type = CL_DEVICE_TYPE_ALL;
        }
      }
      c_dbcsr_acc_opencl_config.ndevices = 0;
      for (i = 0; i < nplatforms; ++i) {
        if (EXIT_SUCCESS == clGetDeviceIDs(platforms[i], type, 0, NULL, &ndevices) && 0 < ndevices) {
          ACC_OPENCL_CHECK(result, clGetDeviceIDs(platforms[i], type, ndevices, devices, NULL), "retrieve device ids");
          if (EXIT_SUCCESS == result) {
            cl_uint j = 0;
            for (; j < ndevices; ++j) {
#  if defined(CL_VERSION_1_2)
              cl_device_partition_property properties[] = {
                CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN, CL_DEVICE_AFFINITY_DOMAIN_NUMA, /*terminator*/ 0};
              cl_uint nunits = 0, n = 0;
              if ((1 < c_dbcsr_acc_opencl_config.devsplit || 0 > c_dbcsr_acc_opencl_config.devsplit) &&
                  /* Intel CPU (e.g., out of two sockets) yields thread-count of both sockets */
                  EXIT_SUCCESS == clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &nunits, NULL) &&
                  1 < nunits)
              {
                n = LIBXSMM_MIN(1 < c_dbcsr_acc_opencl_config.devsplit ? (cl_uint)c_dbcsr_acc_opencl_config.devsplit : nunits,
                  ACC_OPENCL_MAXNDEVS);
                properties[0] = CL_DEVICE_PARTITION_EQUALLY;
                properties[1] = (nunits + n - 1) / n;
              }
              if (0 == c_dbcsr_acc_opencl_config.devsplit || 1 == c_dbcsr_acc_opencl_config.devsplit ||
                  (c_dbcsr_acc_opencl_config.ndevices + 1) == ACC_OPENCL_MAXNDEVS ||
                  EXIT_SUCCESS != clCreateSubDevices(devices[j], properties, 0, NULL, &n))
#  endif
              {
                c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.ndevices] = devices[j];
                ++c_dbcsr_acc_opencl_config.ndevices;
              }
#  if defined(CL_VERSION_1_2)
              else if (1 < n) { /* create subdevices */
                if (ACC_OPENCL_MAXNDEVS < (c_dbcsr_acc_opencl_config.ndevices + n)) {
                  n = (cl_uint)ACC_OPENCL_MAXNDEVS - c_dbcsr_acc_opencl_config.ndevices;
                }
                if (EXIT_SUCCESS == clCreateSubDevices(devices[j], properties, n,
                                      c_dbcsr_acc_opencl_config.devices + c_dbcsr_acc_opencl_config.ndevices, NULL))
                {
                  ACC_OPENCL_CHECK(result, clReleaseDevice(devices[j]), "release device");
                  c_dbcsr_acc_opencl_config.ndevices += n;
                }
                else break;
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
        for (i = 0; (int)i < c_dbcsr_acc_opencl_config.ndevices;) {
          if (EXIT_SUCCESS ==
              clGetDeviceInfo(c_dbcsr_acc_opencl_config.devices[i], CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL))
          {
            if (NULL == LIBXSMM_STRISTR(buffer, env_vendor)) {
#  if defined(CL_VERSION_1_2)
              ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseDevice(c_dbcsr_acc_opencl_config.devices[i]));
#  endif
              --c_dbcsr_acc_opencl_config.ndevices;
              if ((int)i < c_dbcsr_acc_opencl_config.ndevices) { /* keep original order (stable) */
                memmove(&c_dbcsr_acc_opencl_config.devices[i], &c_dbcsr_acc_opencl_config.devices[i + 1],
                  sizeof(cl_device_id) * (c_dbcsr_acc_opencl_config.ndevices - i));
              }
            }
            else ++i;
          }
          else break; /* error: retrieving device vendor */
        }
      }
      /* reorder devices according to c_dbcsr_acc_opencl_order_devices */
      if (EXIT_SUCCESS == result && 1 < c_dbcsr_acc_opencl_config.ndevices) {
        qsort(c_dbcsr_acc_opencl_config.devices, c_dbcsr_acc_opencl_config.ndevices, sizeof(cl_device_id),
          c_dbcsr_acc_opencl_order_devices);
      }
      /* ACC_OPENCL_DEVIDS is parsed as a list of devices (whitelist) */
      if (EXIT_SUCCESS == result && NULL != env_devids && '\0' != *env_devids) {
        cl_uint devids[ACC_OPENCL_MAXNDEVS], ndevids = 0;
        char* did = strtok(env_devids, ACC_OPENCL_DELIMS " ");
        for (; NULL != did && ndevids < ACC_OPENCL_MAXNDEVS; did = strtok(NULL, ACC_OPENCL_DELIMS " ")) {
          const int id = atoi(did);
          if (0 <= id && id < c_dbcsr_acc_opencl_config.ndevices) devids[ndevids++] = id;
        }
        if (0 < ndevids) {
          ndevices = (cl_uint)c_dbcsr_acc_opencl_config.ndevices;
          for (i = 0; i < ndevices; ++i) {
            cl_uint match = 0, j = 0;
            do
              if (i == devids[j]) {
                match = 1;
                break;
              }
            while (++j < ndevids);
            if (0 == match) {
#  if defined(CL_VERSION_1_2)
              ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseDevice(c_dbcsr_acc_opencl_config.devices[i]));
#  endif
              c_dbcsr_acc_opencl_config.devices[i] = NULL;
            }
          }
          for (i = c_dbcsr_acc_opencl_config.ndevices - 1;; --i) {
            if (NULL == c_dbcsr_acc_opencl_config.devices[i]) { /* keep original order (stable) */
              const cl_uint nmove = c_dbcsr_acc_opencl_config.ndevices - (i + 1);
              if (0 < nmove) {
                memmove(
                  &c_dbcsr_acc_opencl_config.devices[i], &c_dbcsr_acc_opencl_config.devices[i + 1], sizeof(cl_device_id) * nmove);
              }
              --c_dbcsr_acc_opencl_config.ndevices;
            }
            if (0 == i) break;
          }
        }
      }
    }
    if (EXIT_SUCCESS == result && 0 < c_dbcsr_acc_opencl_config.ndevices) {
      /* preselect any default device or prune to homogeneous set of devices */
      if (NULL == env_device || '\0' == *env_device) {
        char tmp[ACC_OPENCL_BUFFERSIZE] = "";
        ndevices = (cl_uint)c_dbcsr_acc_opencl_config.ndevices;
        for (i = 0; i < ndevices; ++i) {
          cl_device_type itype;
          result = clGetDeviceInfo(c_dbcsr_acc_opencl_config.devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &itype, NULL);
          if (EXIT_SUCCESS == result) {
            if (0 != (CL_DEVICE_TYPE_DEFAULT & itype)) {
              if (0 < i) {
                c_dbcsr_acc_opencl_config.devices[0] = c_dbcsr_acc_opencl_config.devices[i];
              }
              c_dbcsr_acc_opencl_config.ndevices = 1;
              device_id = 0;
              break;
            }
            else if (CL_DEVICE_TYPE_ALL == type && NULL == env_devtype /*&& CL_DEVICE_TYPE_GPU == itype*/ && device_id <= (int)i) {
              result = clGetDeviceInfo(c_dbcsr_acc_opencl_config.devices[i], CL_DEVICE_NAME, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
              if (EXIT_SUCCESS == result /* prune for homogeneous set of devices */
                  && ('\0' == *tmp || 0 == strncmp(buffer, tmp, ACC_OPENCL_BUFFERSIZE)))
              {
                c_dbcsr_acc_opencl_config.ndevices = i + 1;
                strncpy(tmp, buffer, ACC_OPENCL_BUFFERSIZE);
              }
              else break; /* error: retrieving device name */
            }
          }
          else break; /* error: retrieving device type */
        }
      }
      else { /* prune number of devices to only expose requested ID */
        if (1 < c_dbcsr_acc_opencl_config.ndevices) {
          if (0 < device_id) {
            c_dbcsr_acc_opencl_config.devices[0] =
              c_dbcsr_acc_opencl_config.devices[device_id % c_dbcsr_acc_opencl_config.ndevices];
          }
          c_dbcsr_acc_opencl_config.ndevices = 1;
        }
        device_id = 0;
      }
    }
    if (device_id < c_dbcsr_acc_opencl_config.ndevices) {
      if (EXIT_SUCCESS == result) {
        const size_t nhandles = ACC_OPENCL_MAXNITEMS * c_dbcsr_acc_opencl_config.nthreads;
        assert(0 < c_dbcsr_acc_opencl_config.ndevices);
        assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_MAXNDEVS);
        assert(NULL == c_dbcsr_acc_opencl_config.memptrs);
        assert(NULL == c_dbcsr_acc_opencl_config.memptr_data);
        assert(0 == c_dbcsr_acc_opencl_config.nmemptrs);
        assert(NULL == c_dbcsr_acc_opencl_config.streams);
        assert(NULL == c_dbcsr_acc_opencl_config.events);
        assert(NULL == c_dbcsr_acc_opencl_config.stream_data);
        assert(NULL == c_dbcsr_acc_opencl_config.event_data);
        assert(0 == c_dbcsr_acc_opencl_config.nstreams);
        assert(0 == c_dbcsr_acc_opencl_config.nevents);
        /* allocate and initialize memptr registry */
        c_dbcsr_acc_opencl_config.nmemptrs = nhandles;
        c_dbcsr_acc_opencl_config.memptrs = (c_dbcsr_acc_opencl_info_memptr_t**)malloc(
          sizeof(c_dbcsr_acc_opencl_info_memptr_t*) * nhandles);
        c_dbcsr_acc_opencl_config.memptr_data = (c_dbcsr_acc_opencl_info_memptr_t*)malloc(
          sizeof(c_dbcsr_acc_opencl_info_memptr_t) * nhandles);
        if (NULL != c_dbcsr_acc_opencl_config.memptrs && NULL != c_dbcsr_acc_opencl_config.memptr_data) {
          c_dbcsr_acc_opencl_pmalloc_init(sizeof(c_dbcsr_acc_opencl_info_memptr_t), &c_dbcsr_acc_opencl_config.nmemptrs,
            (void**)c_dbcsr_acc_opencl_config.memptrs, c_dbcsr_acc_opencl_config.memptr_data);
        }
        else {
          free(c_dbcsr_acc_opencl_config.memptrs);
          free(c_dbcsr_acc_opencl_config.memptr_data);
          c_dbcsr_acc_opencl_config.memptr_data = NULL;
          c_dbcsr_acc_opencl_config.memptrs = NULL;
          c_dbcsr_acc_opencl_config.nmemptrs = 0;
          result = EXIT_FAILURE;
        }
        /* allocate and initialize streams registry */
        c_dbcsr_acc_opencl_config.nstreams = nhandles;
        c_dbcsr_acc_opencl_config.streams = (c_dbcsr_acc_opencl_stream_t**)malloc(sizeof(c_dbcsr_acc_opencl_stream_t*) * nhandles);
        c_dbcsr_acc_opencl_config.stream_data = (c_dbcsr_acc_opencl_stream_t*)malloc(
          sizeof(c_dbcsr_acc_opencl_stream_t) * nhandles);
        if (NULL != c_dbcsr_acc_opencl_config.streams && NULL != c_dbcsr_acc_opencl_config.stream_data) {
          c_dbcsr_acc_opencl_pmalloc_init(sizeof(c_dbcsr_acc_opencl_stream_t), &c_dbcsr_acc_opencl_config.nstreams,
            (void**)c_dbcsr_acc_opencl_config.streams, c_dbcsr_acc_opencl_config.stream_data);
        }
        else {
          free(c_dbcsr_acc_opencl_config.streams);
          free(c_dbcsr_acc_opencl_config.stream_data);
          c_dbcsr_acc_opencl_config.stream_data = NULL;
          c_dbcsr_acc_opencl_config.streams = NULL;
          c_dbcsr_acc_opencl_config.nstreams = 0;
          result = EXIT_FAILURE;
        }
        /* allocate and initialize events registry */
        c_dbcsr_acc_opencl_config.nevents = nhandles;
        c_dbcsr_acc_opencl_config.events = (cl_event**)malloc(sizeof(cl_event*) * nhandles);
        c_dbcsr_acc_opencl_config.event_data = (cl_event*)malloc(sizeof(cl_event) * nhandles);
        if (NULL != c_dbcsr_acc_opencl_config.events && NULL != c_dbcsr_acc_opencl_config.event_data) {
          c_dbcsr_acc_opencl_pmalloc_init(sizeof(cl_event*), &c_dbcsr_acc_opencl_config.nevents,
            (void**)c_dbcsr_acc_opencl_config.events, c_dbcsr_acc_opencl_config.event_data);
        }
        else {
          free(c_dbcsr_acc_opencl_config.events);
          free(c_dbcsr_acc_opencl_config.event_data);
          c_dbcsr_acc_opencl_config.event_data = NULL;
          c_dbcsr_acc_opencl_config.events = NULL;
          c_dbcsr_acc_opencl_config.nevents = 0;
          result = EXIT_FAILURE;
        }
        if (
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
          2 <= c_dbcsr_acc_opencl_config.profile ||
#  else
          1 <= c_dbcsr_acc_opencl_config.profile ||
#  endif
          0 > c_dbcsr_acc_opencl_config.profile)
        {
          const int profile = LIBXSMM_MAX(LIBXSMM_ABS(c_dbcsr_acc_opencl_config.profile), 2);
          const c_dbcsr_acc_opencl_hist_update_fn update[] = {c_dbcsr_acc_opencl_hist_avg, c_dbcsr_acc_opencl_hist_add};
          c_dbcsr_acc_opencl_hist_create(&c_dbcsr_acc_opencl_config.hist_h2d, profile + 1, profile * 4, 2, update);
          c_dbcsr_acc_opencl_hist_create(&c_dbcsr_acc_opencl_config.hist_d2h, profile + 1, profile * 4, 2, update);
          c_dbcsr_acc_opencl_hist_create(&c_dbcsr_acc_opencl_config.hist_d2d, profile + 1, profile * 4, 2, update);
        }
        else {
          assert(NULL == c_dbcsr_acc_opencl_config.hist_h2d);
          assert(NULL == c_dbcsr_acc_opencl_config.hist_d2h);
          assert(NULL == c_dbcsr_acc_opencl_config.hist_d2d);
        }
        if (EXIT_SUCCESS == result) { /* lastly, print active device and list of devices */
#  if defined(ACC_OPENCL_ACTIVATE)
          if (0 <= ACC_OPENCL_ACTIVATE && ACC_OPENCL_ACTIVATE < c_dbcsr_acc_opencl_config.ndevices) {
            result = c_dbcsr_acc_opencl_set_active_device(NULL /*lock*/, ACC_OPENCL_ACTIVATE);
          }
          else {
            if (0 < c_dbcsr_acc_opencl_config.nrank && 1 < c_dbcsr_acc_opencl_config.ndevices) {
              device_id = c_dbcsr_acc_opencl_config.nrank % c_dbcsr_acc_opencl_config.ndevices;
            }
            result = c_dbcsr_acc_opencl_set_active_device(NULL /*lock*/, device_id);
          }
#  else
          c_dbcsr_acc_opencl_config.device_id = device_id;
#  endif
          if ((2 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity) &&
              (0 == c_dbcsr_acc_opencl_config.nrank))
          {
            char platform_name[ACC_OPENCL_BUFFERSIZE];
            for (i = 0; i < (cl_uint)c_dbcsr_acc_opencl_config.ndevices; ++i) {
              if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_name(c_dbcsr_acc_opencl_config.devices[i], buffer,
                                    ACC_OPENCL_BUFFERSIZE, platform_name, ACC_OPENCL_BUFFERSIZE, /*cleanup*/ 0))
              {
                fprintf(stderr, "INFO ACC/OpenCL: DEVICE -> \"%s : %s\" (%u)\n", platform_name, buffer, i);
              }
            }
          }
        }
      }
    }
    else { /* mark as initialized */
      c_dbcsr_acc_opencl_config.ndevices = -1;
    }
#  if defined(__DBCSR_ACC)
    /* DBCSR shall call c_dbcsr_acc_init as well as libsmm_acc_init (since both interfaces are used).
     * Also, libsmm_acc_init may privately call c_dbcsr_acc_init (as it depends on the ACC interface).
     * The implementation of c_dbcsr_acc_init should hence be safe against "over initialization".
     * However, DBCSR only calls c_dbcsr_acc_init (and expects an implicit libsmm_acc_init).
     */
    if (EXIT_SUCCESS == result) result = libsmm_acc_init();
#  endif
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


/* attempt to automatically initialize backend */
LIBXSMM_ATTRIBUTE_CTOR void c_dbcsr_acc_opencl_init(void) {
  if (NULL == c_dbcsr_acc_opencl_config.lock_main) { /* avoid to configure multiple times */
    c_dbcsr_acc_opencl_configure();
  }
#  if defined(ACC_OPENCL_PREINIT)
  ACC_OPENCL_EXPECT(EXIT_SUCCESS == c_dbcsr_acc_init());
#  endif
}


/* attempt to automatically finalize backend */
LIBXSMM_ATTRIBUTE_DTOR void c_dbcsr_acc_opencl_finalize(void) {
  assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_MAXNDEVS);
  if (0 != c_dbcsr_acc_opencl_config.ndevices) {
    int precision[] = {0, 1}, i;
    LIBXSMM_STDIO_ACQUIRE();
    c_dbcsr_acc_opencl_hist_print(stderr, c_dbcsr_acc_opencl_config.hist_h2d, "\nPROF ACC/OpenCL: H2D", precision, NULL /*adjust*/);
    c_dbcsr_acc_opencl_hist_print(stderr, c_dbcsr_acc_opencl_config.hist_d2h, "\nPROF ACC/OpenCL: D2H", precision, NULL /*adjust*/);
    c_dbcsr_acc_opencl_hist_print(stderr, c_dbcsr_acc_opencl_config.hist_d2d, "\nPROF ACC/OpenCL: D2D", precision, NULL /*adjust*/);
    LIBXSMM_STDIO_RELEASE();
    for (i = 0; i < ACC_OPENCL_MAXNDEVS; ++i) {
      const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[i];
      if (NULL != device_id) {
#  if defined(CL_VERSION_1_2) && 0 /* avoid potential segfault */
        ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseDevice(device_id));
#  endif
      }
    }
    if (NULL != c_dbcsr_acc_opencl_config.device.stream.queue) { /* release private stream */
      clReleaseCommandQueue(c_dbcsr_acc_opencl_config.device.stream.queue); /* ignore return code */
    }
    if (NULL != c_dbcsr_acc_opencl_config.device.context) {
      const cl_context context = c_dbcsr_acc_opencl_config.device.context;
      c_dbcsr_acc_opencl_config.device.context = NULL;
      clReleaseContext(context); /* ignore return code */
    }
    for (i = 0; i < ACC_OPENCL_NLOCKS; ++i) { /* destroy locks */
      ACC_OPENCL_DESTROY((ACC_OPENCL_LOCKTYPE*)(c_dbcsr_acc_opencl_locks + ACC_OPENCL_CACHELINE * i));
    }
    /* release/reset buffers */
    c_dbcsr_acc_opencl_hist_free(c_dbcsr_acc_opencl_config.hist_h2d);
    c_dbcsr_acc_opencl_hist_free(c_dbcsr_acc_opencl_config.hist_d2h);
    c_dbcsr_acc_opencl_hist_free(c_dbcsr_acc_opencl_config.hist_d2d);
    free(c_dbcsr_acc_opencl_config.memptrs);
    free(c_dbcsr_acc_opencl_config.memptr_data);
    free(c_dbcsr_acc_opencl_config.streams);
    free(c_dbcsr_acc_opencl_config.stream_data);
    free(c_dbcsr_acc_opencl_config.events);
    free(c_dbcsr_acc_opencl_config.event_data);
    /* clear entire configuration structure */
    memset(&c_dbcsr_acc_opencl_config, 0, sizeof(c_dbcsr_acc_opencl_config));
#  if defined(ACC_OPENCL_CACHE_DID)
    c_dbcsr_acc_opencl_active_id = 0; /* reset cached active device-ID */
#  endif
    libxsmm_finalize();
  }
}


int c_dbcsr_acc_finalize(void) {
#  if defined(_OPENMP)
  /* initialization/finalization is not meant to be thread-safe */
  int result = ((0 == omp_in_parallel() || /*main*/ 0 == omp_get_thread_num()) ? EXIT_SUCCESS : EXIT_FAILURE);
#  else
  int result = EXIT_SUCCESS;
#  endif
  static void (*cleanup)(void) = c_dbcsr_acc_opencl_finalize;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
  assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_MAXNDEVS);
  if (0 != c_dbcsr_acc_opencl_config.ndevices && NULL != cleanup) {
#  if defined(__DBCSR_ACC)
    /* DBCSR may call c_dbcsr_acc_init as well as libsmm_acc_init() since both interface are used.
     * libsmm_acc_init may privately call c_dbcsr_acc_init (as it depends on the ACC interface).
     * The implementation of c_dbcsr_acc_init should be safe against "over initialization".
     * However, DBCSR only calls c_dbcsr_acc_init and expects an implicit libsmm_acc_init().
     */
    if (EXIT_SUCCESS == result) result = libsmm_acc_finalize();
#  endif
    if (EXIT_SUCCESS == result) result = atexit(cleanup);
    cleanup = NULL;
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


void c_dbcsr_acc_clear_errors(void) {}


int c_dbcsr_acc_get_ndevices(int* ndevices) {
  int result;
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  int routine_handle;
  if (0 != c_dbcsr_acc_opencl_config.profile) {
    static const char* routine_name_ptr = LIBXSMM_FUNCNAME + ACC_OPENCL_PROFILE_DBCSR;
    static const int routine_name_len = (int)sizeof(LIBXSMM_FUNCNAME) - (ACC_OPENCL_PROFILE_DBCSR + 1);
    c_dbcsr_timeset((const char**)&routine_name_ptr, &routine_name_len, &routine_handle);
  }
#  endif
#  if defined(__DBCSR_ACC) /* lazy initialization */
  /* DBCSR calls c_dbcsr_acc_get_ndevices before calling c_dbcsr_acc_init. */
  result = c_dbcsr_acc_init();
  if (EXIT_SUCCESS == result)
#  endif
  {
    if (NULL != ndevices && 0 != c_dbcsr_acc_opencl_config.ndevices) {
      *ndevices = (0 < c_dbcsr_acc_opencl_config.ndevices ? c_dbcsr_acc_opencl_config.ndevices : 0);
      result = EXIT_SUCCESS;
    }
    else result = EXIT_FAILURE;
  }
#  if defined(ACC_OPENCL_PROFILE_DBCSR)
  if (0 != c_dbcsr_acc_opencl_config.profile) c_dbcsr_timestop(&routine_handle);
#  endif
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_device_id(cl_device_id device, int* device_id, int* global_id) {
  int result = EXIT_SUCCESS, i;
  assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_MAXNDEVS);
  assert(NULL != device_id || NULL != global_id);
  for (i = 0; i < c_dbcsr_acc_opencl_config.ndevices; ++i) {
    if (device == c_dbcsr_acc_opencl_config.devices[i]) break;
  }
  if (i < c_dbcsr_acc_opencl_config.ndevices) {
    if (NULL != device_id) *device_id = i;
    if (NULL != global_id) {
      *global_id = i;
      for (++i; i < ACC_OPENCL_MAXNDEVS; ++i) {
        if (NULL != c_dbcsr_acc_opencl_config.devices[i]) {
          if (device == c_dbcsr_acc_opencl_config.devices[i]) {
            *global_id = i;
            break;
          }
        }
        else break;
      }
    }
  }
  else {
    if (NULL != device_id) *device_id = -1;
    if (NULL != global_id) *global_id = -1;
    if (NULL != device) result = EXIT_FAILURE;
  }
  return result;
}


int c_dbcsr_acc_opencl_device_vendor(cl_device_id device, const char vendor[], int use_platform_name) {
  char buffer[ACC_OPENCL_BUFFERSIZE];
  int result = EXIT_SUCCESS;
  assert(NULL != device && NULL != vendor);
  if (0 == use_platform_name) {
    result = clGetDeviceInfo(device, CL_DEVICE_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
  }
  else {
    cl_platform_id platform;
    result = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
    if (EXIT_SUCCESS == result) {
      result = clGetPlatformInfo(
        platform, 1 == use_platform_name ? CL_PLATFORM_NAME : CL_PLATFORM_VENDOR, ACC_OPENCL_BUFFERSIZE, buffer, NULL);
    }
  }
  if (EXIT_SUCCESS == result) {
    result = (NULL != LIBXSMM_STRISTR(buffer, vendor) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  return result;
}


int c_dbcsr_acc_opencl_device_uid(cl_device_id device, const char devname[], unsigned int* uid) {
  int result;
  if (NULL != uid) {
    if (NULL != device && EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(device, "intel", 0 /*use_platform_name*/)) {
      result = clGetDeviceInfo(device, 0x4251 /*CL_DEVICE_ID_INTEL*/, sizeof(unsigned int), uid, NULL);
    }
    else result = EXIT_FAILURE;
    if (EXIT_SUCCESS != result) {
      if (NULL != devname && '\0' != *devname) {
        *uid = (unsigned int)strtoul(devname, NULL, 0);
        if (0 == *uid) {
          const char *const begin = strrchr(devname, '['), *const end = strrchr(devname, ']');
          if (NULL != begin && begin < end) {
            *uid = (unsigned int)strtoul(begin + 1, NULL, 0);
          }
          if (0 == *uid) {
            const size_t size = strlen(devname);
            const unsigned int hash = libxsmm_hash(devname, (unsigned int)size, 25071975 /*seed*/);
            *uid = libxsmm_hash(&hash, 4 /*size*/, hash >> 16 /*seed*/) & 0xFFFF;
          }
        }
        result = EXIT_SUCCESS;
      }
      else {
        result = EXIT_FAILURE;
        *uid = 0;
      }
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


int c_dbcsr_acc_opencl_device_name(
  cl_device_id device, char name[], size_t name_maxlen, char platform[], size_t platform_maxlen, int cleanup) {
  int result_name = 0, result_platform = 0;
  assert(NULL != name || NULL != platform);
  if (NULL == device && 0 < c_dbcsr_acc_opencl_config.ndevices) {
    device = c_dbcsr_acc_opencl_config.devices[0]; /* NULL-device refers to device 0 */
  }
  if (NULL != name && 0 != name_maxlen) {
    result_name = clGetDeviceInfo(device, CL_DEVICE_NAME, name_maxlen, name, NULL);
    if (0 != cleanup && EXIT_SUCCESS == result_name) {
      char* const part = strchr(name, ':');
      if (NULL != part) *part = '\0';
    }
  }
  if (NULL != platform && 0 != platform_maxlen) {
    cl_platform_id platform_id;
    result_platform = clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform_id, NULL);
    if (EXIT_SUCCESS == result_platform) {
      result_platform = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_maxlen, platform, NULL);
    }
  }
  return result_name | result_platform;
}


int c_dbcsr_acc_opencl_device_level(
  cl_device_id device, int std_clevel[2], int std_level[2], char std_flag[16], cl_device_type* type) {
  char buffer[ACC_OPENCL_BUFFERSIZE];
  unsigned int std_clevel_uint[2] = {0}, std_level_uint[2] = {0};
  int result = EXIT_SUCCESS;
  assert(NULL != device && (NULL != std_clevel || NULL != std_level || NULL != std_flag || NULL != type));
  result = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, ACC_OPENCL_BUFFERSIZE / 2, buffer, NULL);
  if (EXIT_SUCCESS == result && (NULL != std_clevel || NULL != std_flag)) {
    if (2 == sscanf(buffer, "OpenCL C %u.%u", std_clevel_uint, std_clevel_uint + 1)) {
      std_clevel[0] = (int)std_clevel_uint[0];
      std_clevel[1] = (int)std_clevel_uint[1];
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result && (NULL != std_level || NULL != std_flag)) {
    result = clGetDeviceInfo(
      device, CL_DEVICE_VERSION, ACC_OPENCL_BUFFERSIZE - ACC_OPENCL_BUFFERSIZE / 2, buffer + ACC_OPENCL_BUFFERSIZE / 2, NULL);
    if (EXIT_SUCCESS == result) {
      if (2 == sscanf(buffer + ACC_OPENCL_BUFFERSIZE / 2, "OpenCL %u.%u", std_level_uint, std_level_uint + 1)) {
        std_level[0] = (int)std_level_uint[0];
        std_level[1] = (int)std_level_uint[1];
      }
      else result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS == result && NULL != std_flag) {
    if (2 <= std_level_uint[0]) {
      const int nchar = LIBXSMM_SNPRINTF(std_flag, 16, "-cl-std=CL%u.0", std_level_uint[0]);
      if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
    }
    else if (1 <= std_level_uint[0]) {
      if (1 <= std_level_uint[1]) {
        const int nchar = LIBXSMM_SNPRINTF(std_flag, 16, "-cl-std=CL%u.%u", std_level_uint[0], std_level_uint[1]);
        if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
      }
      else if (1 <= std_clevel_uint[0]) { /* fallback */
        const int nchar = LIBXSMM_SNPRINTF(std_flag, 16, "-cl-std=CL%u.%u", std_clevel_uint[0], std_clevel_uint[1]);
        if (0 >= nchar || 16 <= nchar) result = EXIT_FAILURE;
      }
      else *std_flag = '\0'; /* not an error */
    }
    else *std_flag = '\0'; /* not an error */
  }
  if (EXIT_SUCCESS == result && NULL != type) {
    result = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), type, NULL);
  }
  if (EXIT_SUCCESS != result) {
    if (NULL != std_clevel) std_clevel[0] = std_clevel[1] = 0;
    if (NULL != std_level) std_level[0] = std_level[1] = 0;
    if (NULL != std_flag) *std_flag = '\0';
    if (NULL != type) *type = 0;
  }
  return result;
}


int c_dbcsr_acc_opencl_device_ext(cl_device_id device, const char* const extnames[], int num_exts) {
  int result = ((NULL != extnames && 0 < num_exts) ? EXIT_SUCCESS : EXIT_FAILURE);
  char extensions[ACC_OPENCL_BUFFERSIZE], buffer[ACC_OPENCL_BUFFERSIZE];
  assert(NULL != device);
  ACC_OPENCL_CHECK(
    result, clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ACC_OPENCL_BUFFERSIZE, extensions, NULL), "retrieve device extensions");
  if (EXIT_SUCCESS == result) {
    do {
      if (NULL != extnames[--num_exts]) {
        const char* const end = buffer + strlen(extnames[num_exts]); /* before strtok */
        char* ext = strtok(strncpy(buffer, extnames[num_exts], ACC_OPENCL_BUFFERSIZE - 1), ACC_OPENCL_DELIMS " \t");
        for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), ACC_OPENCL_DELIMS " \t") : NULL)) {
          if (NULL == strstr(extensions, ext)) {
            return EXIT_FAILURE;
          }
        }
      }
    } while (0 < num_exts);
  }
  return result;
}


int c_dbcsr_acc_opencl_create_context(cl_device_id active_id, cl_context* context) {
  cl_platform_id platform = NULL;
  int result;
  assert(0 < c_dbcsr_acc_opencl_config.ndevices);
  assert(NULL != active_id && NULL != context);
  result = clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
  assert(EXIT_SUCCESS != result || NULL != platform);
  if (EXIT_SUCCESS == result) {
    void (*const notify)(
      const char*, const void*, size_t, void*) = (0 != c_dbcsr_acc_opencl_config.verbosity ? c_dbcsr_acc_opencl_notify : NULL);
    cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, 0 /*placeholder*/, 0 /* end of properties */
    };
    cl_context ctx = NULL;
    properties[1] = (long)platform;
    ctx = clCreateContext(properties, 1 /*num_devices*/, &active_id, notify, NULL /* user_data*/, &result);
    if (EXIT_SUCCESS != result && CL_INVALID_DEVICE != result) { /* retry */
      ctx = clCreateContext(NULL /*properties*/, 1 /*num_devices*/, &active_id, notify, NULL /* user_data*/, &result);
    }
    if (EXIT_SUCCESS == result) {
      assert(NULL != ctx);
      *context = ctx;
      if (0 != c_dbcsr_acc_opencl_config.verbosity) {
        char buffer[ACC_OPENCL_BUFFERSIZE];
        int global_id = 0;
        if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_name(
                              active_id, buffer, ACC_OPENCL_BUFFERSIZE, NULL /*platform*/, 0 /*platform_maxlen*/, /*cleanup*/ 1) &&
            EXIT_SUCCESS == c_dbcsr_acc_opencl_device_id(active_id, NULL /*devid*/, &global_id))
        {
          const size_t size = strlen(buffer);
          unsigned int uid[] = {0, 0};
          if ((EXIT_SUCCESS == c_dbcsr_acc_opencl_device_uid(NULL /*device*/, buffer, uid + 1)) &&
              (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_uid(active_id, NULL /*devname*/, uid) || 0 != uid[1]) && uid[0] != uid[1])
          {
            ACC_OPENCL_EXPECT(0 < LIBXSMM_SNPRINTF(buffer + size, LIBXSMM_MAX(0, ACC_OPENCL_BUFFERSIZE - size), " [0x%04x]",
                                    0 != uid[0] ? uid[0] : uid[1]));
          }
          fprintf(stderr, "INFO ACC/OpenCL: ndevices=%i device%i=\"%s\" context=%p pid=%u nthreads=%i\n",
            c_dbcsr_acc_opencl_config.ndevices, global_id, buffer, (void*)ctx, libxsmm_get_pid(),
            c_dbcsr_acc_opencl_config.nthreads);
        }
      }
    }
    else {
      if (CL_INVALID_DEVICE == result &&
          EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/))
      {
        fprintf(stderr, "WARN ACC/OpenCL: if MPI-ranks target the same device in exclusive mode,\n"
                        "                    SMI must be used to enable sharing the device.\n");
      }
      *context = NULL;
    }
  }
  return result;
}


int c_dbcsr_acc_opencl_set_active_device(ACC_OPENCL_LOCKTYPE* lock, int device_id) {
  c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = EXIT_SUCCESS;
  assert(c_dbcsr_acc_opencl_config.ndevices < ACC_OPENCL_MAXNDEVS);
  if (0 <= device_id && device_id < c_dbcsr_acc_opencl_config.ndevices) {
    /* accessing devices is thread-safe (array is fixed after initialization) */
    const cl_device_id active_id = c_dbcsr_acc_opencl_config.devices[device_id];
    if (NULL != active_id) {
      cl_context context = NULL;
      if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
      context = devinfo->context;
      if (NULL != context) {
        if (device_id != c_dbcsr_acc_opencl_config.device_id) {
          const cl_device_id context_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
          assert(NULL != context_id);
#  if defined(CL_VERSION_1_2)
          ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseDevice(context_id));
#  endif
          result = clReleaseContext(context);
          context = NULL;
        }
      }
      if (EXIT_SUCCESS == result && (NULL == devinfo->context || device_id != c_dbcsr_acc_opencl_config.device_id)) {
        result = c_dbcsr_acc_opencl_create_context(active_id, &context);
        assert(NULL != context || EXIT_SUCCESS != result);
      }
      /* update/cache device-specific information */
      if (EXIT_SUCCESS == result && (NULL == devinfo->context || device_id != c_dbcsr_acc_opencl_config.device_id)) {
        if (NULL != devinfo->stream.queue) { /* release private stream */
          ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseCommandQueue(devinfo->stream.queue));
        }
        memset(devinfo, 0, sizeof(*devinfo));
        result = c_dbcsr_acc_opencl_device_level(
          active_id, devinfo->std_clevel, devinfo->std_level, devinfo->std_flag, &devinfo->type);
        if (EXIT_SUCCESS == result) {
          char devname[ACC_OPENCL_BUFFERSIZE] = "";
          const char* const sgexts[] = {"cl_intel_required_subgroup_size", "cl_intel_subgroups", "cl_khr_subgroups"};
          size_t sgsizes[16], nbytes = 0, sgmin = (size_t)-1, i;
#  if defined(ACC_OPENCL_CMDAGR)
          ACC_OPENCL_STREAM_PROPERTIES_TYPE properties[4] = {
            CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0 /* terminator */
          };
#  endif
          cl_platform_id platform = NULL;
          cl_bitfield bitfield = 0;
          devinfo->intel = (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "intel", 0 /*use_platform_name*/));
          devinfo->nv = (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "nvidia", 0 /*use_platform_name*/));
          if (EXIT_SUCCESS != c_dbcsr_acc_opencl_device_name(active_id, devname, ACC_OPENCL_BUFFERSIZE, NULL /*platform*/,
                                0 /*platform_maxlen*/, /*cleanup*/ 1) ||
              EXIT_SUCCESS != c_dbcsr_acc_opencl_device_uid(active_id, devname, &devinfo->uid))
          {
            devinfo->uid = (cl_uint)-1;
          }
          if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "amd", 0 /*use_platform_name*/) ||
              EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "amd", 1 /*use_platform_name*/))
          {
            devinfo->amd = 1;
            if ('\0' != *devname) {
              const char* const gfxname = LIBXSMM_STRISTR(devname, "gfx");
              if (NULL != gfxname && 90 <= atoi(gfxname + 3)) {
                devinfo->amd = 2;
              }
            }
          }
          if (EXIT_SUCCESS !=
              clGetDeviceInfo(active_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool) /*cl_int*/, &devinfo->unified, NULL))
          {
            devinfo->unified = CL_FALSE;
          }
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), devinfo->wgsize, NULL)) {
            devinfo->wgsize[0] = 1;
          }
          if (EXIT_SUCCESS != clGetDeviceInfo(active_id, 4199 /*CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE*/, sizeof(size_t),
                                devinfo->wgsize + 1, NULL)) /* CL_VERSION_3_0 */
          {
            devinfo->wgsize[1] = 1;
          }
          assert(0 == devinfo->wgsize[2]);
          if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(active_id, sgexts, 2) &&
              EXIT_SUCCESS ==
                clGetDeviceInfo(active_id, 0x4108 /*CL_DEVICE_SUB_GROUP_SIZES_INTEL*/, sizeof(sgsizes), sgsizes, &nbytes))
          {
            for (i = 0; (i * sizeof(size_t)) < nbytes; ++i) {
              const size_t sgsize = sgsizes[i];
              if (sgsize < sgmin) sgmin = sgsize;
              if (0 != devinfo->wgsize[1] && 0 == (sgsize % devinfo->wgsize[1]) && devinfo->wgsize[2] < sgsize) {
                if (devinfo->wgsize[1] < sgsize) devinfo->wgsize[1] = sgsize;
                devinfo->wgsize[2] = sgsize;
              }
            }
            if (0 != devinfo->wgsize[2]) devinfo->wgsize[2] = sgmin;
          }
          else devinfo->wgsize[2] = 0;
#  if defined(ACC_OPENCL_XHINTS)
          if (0 != (1 & c_dbcsr_acc_opencl_config.xhints) && 2 <= *devinfo->std_level && 0 != devinfo->intel &&
              0 == devinfo->unified && 0 == c_dbcsr_acc_opencl_config.profile &&
              EXIT_SUCCESS == clGetDeviceInfo(active_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL) &&
              EXIT_SUCCESS == c_dbcsr_acc_opencl_device_vendor(active_id, "intel", 2 /*platform vendor*/) &&
              EXIT_SUCCESS == clGetDeviceInfo(active_id, 0x4191 /*CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL*/, sizeof(cl_bitfield),
                                &bitfield, NULL) &&
              0 != bitfield) /* cl_intel_unified_shared_memory extension */
          {
            void* ptr[8] = {NULL};
            int i = 0, n = 0;
            ptr[0] = clGetExtensionFunctionAddressForPlatform(platform, "clSetKernelArgMemPointerINTEL");
            ptr[1] = clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL");
            ptr[2] = clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL");
            ptr[3] = clGetExtensionFunctionAddressForPlatform(platform, "clDeviceMemAllocINTEL");
            ptr[4] = clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL");
            for (; i < (int)(sizeof(ptr) / sizeof(*ptr)); ++i) {
              if (NULL != ptr[i]) ++n;
            }
            if (5 == n) {
              LIBXSMM_ASSIGN127(&devinfo->clSetKernelArgMemPointerINTEL, ptr + 0);
              LIBXSMM_ASSIGN127(&devinfo->clEnqueueMemFillINTEL, ptr + 1);
              LIBXSMM_ASSIGN127(&devinfo->clEnqueueMemcpyINTEL, ptr + 2);
              LIBXSMM_ASSIGN127(&devinfo->clDeviceMemAllocINTEL, ptr + 3);
              LIBXSMM_ASSIGN127(&devinfo->clMemFreeINTEL, ptr + 4);
            }
            else if (0 != n) {
              fprintf(stderr, "WARN ACC/OpenCL: inconsistent state discovered!\n");
            }
          }
#  endif
#  if defined(ACC_OPENCL_CMDAGR)
          if (0 != devinfo->intel) { /* device vendor (above) can now be used */
            int result_cmdagr = EXIT_SUCCESS;
            const cl_command_queue q = ACC_OPENCL_CREATE_COMMAND_QUEUE(context, active_id, properties, &result_cmdagr);
            if (EXIT_SUCCESS == result_cmdagr) {
              assert(NULL != q);
              clReleaseCommandQueue(q);
            }
          }
#  endif
          properties[1] = 0;
          devinfo->stream.queue = ACC_OPENCL_CREATE_COMMAND_QUEUE(context, active_id, properties, &result);
        }
        if (EXIT_SUCCESS == result) {
          if (NULL == devinfo->context || device_id != c_dbcsr_acc_opencl_config.device_id) {
            c_dbcsr_acc_opencl_config.device_id = device_id;
            devinfo->context = context;
          }
        }
        else memset(devinfo, 0, sizeof(*devinfo));
      }
      if (NULL != lock) ACC_OPENCL_RELEASE(lock);
    }
    else result = EXIT_FAILURE;
  }
  else result = EXIT_FAILURE;
  assert(EXIT_SUCCESS == result || NULL == devinfo->context);
  return result;
}


int c_dbcsr_acc_set_active_device(int device_id) {
  /* avoid ACC_OPENCL_PROFILE_DBCSR in this routine */
  int result = EXIT_SUCCESS;
  if (0 <= device_id) {
#  if defined(__DBCSR_ACC) && defined(__OFFLOAD_OPENCL)
    if (0 == c_dbcsr_acc_opencl_config.ndevices) { /* not initialized */
      result = c_dbcsr_acc_init();
    }
#  endif
  }
  else result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    if (device_id < c_dbcsr_acc_opencl_config.ndevices) {
#  if defined(ACC_OPENCL_CACHE_DID)
      if (c_dbcsr_acc_opencl_active_id != (device_id + 1))
#  endif
      {
        result = c_dbcsr_acc_opencl_set_active_device(c_dbcsr_acc_opencl_config.lock_main, device_id);
#  if defined(ACC_OPENCL_CACHE_DID)
        if (EXIT_SUCCESS == result) c_dbcsr_acc_opencl_active_id = device_id + 1;
#  endif
      }
    }
    else result = EXIT_FAILURE;
  }
  ACC_OPENCL_RETURN(result);
}


int c_dbcsr_acc_opencl_flags_atomics(const c_dbcsr_acc_opencl_device_t* devinfo, c_dbcsr_acc_opencl_atomic_fp_t kind,
  const char* exts[], size_t* exts_maxlen, char flags[], size_t flags_maxlen) {
  size_t ext1, ext2;
  int result = 0;
  for (ext1 = 0; ext1 < (NULL != exts_maxlen ? *exts_maxlen : 0); ++ext1) {
    if (NULL == exts[ext1] || '\0' == *exts[ext1]) break;
  }
  for (ext2 = ext1 + 1; ext2 < (NULL != exts_maxlen ? *exts_maxlen : 0); ++ext2) {
    if (NULL == exts[ext2] || '\0' == *exts[ext2]) break;
  }
  if (NULL != devinfo && NULL != exts_maxlen && ext2 < *exts_maxlen) {
    const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
    const char* atomic_type = "";
    switch (kind) {
      case c_dbcsr_acc_opencl_atomic_fp_64: {
        exts[ext1] = "cl_khr_fp64 cl_khr_int64_base_atomics cl_khr_int64_extended_atomics";
        if (2 <= *devinfo->std_level && EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, exts, ext2)) {
          atomic_type = "-DTA=long -DTA2=atomic_long -DTF=atomic_double";
        }
        else {
          exts[ext1] = "cl_khr_fp64 cl_khr_int64_base_atomics";
          if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, exts, ext2)) {
            atomic_type = "-DTA=long";
          }
          else { /* fallback */
            exts[ext1] = "cl_khr_fp64 cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics";
            if (2 <= *devinfo->std_level && EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, exts, ext2)) {
              atomic_type = "-DATOMIC32_ADD64 -DTA=int -DTA2=atomic_int -DTF=atomic_double";
            }
            else {
              exts[ext1] = "cl_khr_fp64 cl_khr_global_int32_base_atomics";
              if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, exts, ext2)) {
                atomic_type = "-DATOMIC32_ADD64 -DTA=int";
              }
              else kind = c_dbcsr_acc_opencl_atomic_fp_no;
            }
          }
        }
      } break;
      case c_dbcsr_acc_opencl_atomic_fp_32: {
        exts[ext1] = "cl_khr_global_int32_base_atomics cl_khr_global_int32_extended_atomics";
        if (2 <= *devinfo->std_level && EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, exts, ext2)) {
          exts[ext2] = "cl_khr_int64_base_atomics cl_khr_int64_extended_atomics";
          atomic_type = "-DTA=int -DTA2=atomic_int -DTF=atomic_float";
        }
        else {
          exts[ext1] = "cl_khr_global_int32_base_atomics";
          if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, exts, ext2)) {
            exts[ext2] = "cl_khr_int64_base_atomics";
            atomic_type = "-DTA=int";
          }
          else kind = c_dbcsr_acc_opencl_atomic_fp_no;
        }
      } break;
      default: assert(c_dbcsr_acc_opencl_atomic_fp_no == kind);
    }
    if (c_dbcsr_acc_opencl_atomic_fp_no != kind) {
      const char *barrier_expr = NULL, *atomic_exp = NULL, *atomic_ops = "";
      const char* const env_barrier = getenv("ACC_OPENCL_BARRIER");
      const char* const env_atomics = getenv("ACC_OPENCL_ATOMICS");
      if (NULL == env_barrier || '0' != *env_barrier) {
        barrier_expr = ((2 <= *devinfo->std_level && (0 == devinfo->intel || (CL_DEVICE_TYPE_CPU != devinfo->type)))
                          ? "-D\"BARRIER(A)=work_group_barrier(A,memory_scope_work_group)\""
                          : "-D\"BARRIER(A)=barrier(A)\"");
      }
      else barrier_expr = ""; /* no barrier */
      assert(NULL != barrier_expr);
      if (NULL == env_atomics || '0' != *env_atomics) {
        /* can signal/force atomics without confirmation */
        const int force_atomics = ((NULL == env_atomics || '\0' == *env_atomics) ? 0 : atoi(env_atomics));
        if (NULL == env_atomics || '\0' == *env_atomics || 0 != force_atomics) {
          cl_bitfield fp_atomics = 0;
          if (EXIT_SUCCESS == clGetDeviceInfo(device_id,
                                (cl_device_info)(c_dbcsr_acc_opencl_atomic_fp_64 == kind ? 0x4232 : 0x4231), sizeof(cl_bitfield),
                                &fp_atomics, NULL) &&
              0 != (/*add*/ (1 << 1) & fp_atomics))
          {
            exts[ext2] = "cl_ext_float_atomics";
#  if 1 /* enabling this permitted extension in source code causes compiler warning */
            *exts_maxlen = ext2; /* quietly report extension by reducing exts_maxlen */
#  endif
            atomic_exp = (c_dbcsr_acc_opencl_atomic_fp_64 == kind
                            ? "atomic_fetch_add_explicit((GLOBAL_VOLATILE(atomic_double)*)A,B,"
                              "memory_order_relaxed,memory_scope_work_group)"
                            : "atomic_fetch_add_explicit((GLOBAL_VOLATILE(atomic_float)*)A,B,"
                              "memory_order_relaxed,memory_scope_work_group)");
          }
          else if (0 != force_atomics || (0 != devinfo->intel && ((0x4905 != devinfo->uid && 0 == devinfo->unified)))) {
            if ((((0 != force_atomics || (0 != devinfo->intel && ((0x0bd0 <= devinfo->uid && 0x0bdb >= devinfo->uid) ||
                                                                   c_dbcsr_acc_opencl_atomic_fp_32 == kind))))))
            {
              if (0 == force_atomics && (0 == devinfo->intel || 0x0bd0 > devinfo->uid || 0x0bdb < devinfo->uid)) {
                exts[ext2] = "cl_intel_global_float_atomics";
                atomic_ops = "-Dcl_intel_global_float_atomics";
              }
              else {
                atomic_ops = ((2 > *devinfo->std_level && 2 > force_atomics)
                                ? "-DATOMIC_PROTOTYPES=1"
                                : (3 > force_atomics ? "-DATOMIC_PROTOTYPES=2" : "-DATOMIC_PROTOTYPES=3"));
              }
              atomic_exp = ((2 > *devinfo->std_level && 2 > force_atomics) ? "atomic_add(A,B)"
                                                                           : "atomic_fetch_add_explicit((GLOBAL_VOLATILE(TF)*)A,B,"
                                                                             "memory_order_relaxed,memory_scope_work_group)");
            }
            else {
              atomic_exp = "atomic_add_global_cmpxchg(A,B)";
              atomic_ops = "-DCMPXCHG=atom_cmpxchg";
            }
          }
          else if (0 == devinfo->nv) {
            if (1 >= devinfo->amd) {
              atomic_ops = (c_dbcsr_acc_opencl_atomic_fp_32 == kind ? "-DCMPXCHG=atomic_cmpxchg" : "-DCMPXCHG=atom_cmpxchg");
              atomic_exp = "atomic_add_global_cmpxchg(A,B)";
              exts[ext2] = NULL;
            }
            else { /* GCN */
              atomic_exp = (c_dbcsr_acc_opencl_atomic_fp_64 == kind
                              ? "__builtin_amdgcn_global_atomic_fadd_f64(A,B,__ATOMIC_RELAXED,__OPENCL_MEMORY_SCOPE_WORK_GROUP)"
                              : "__builtin_amdgcn_global_atomic_fadd_f32(A,B,__ATOMIC_RELAXED,__OPENCL_MEMORY_SCOPE_WORK_GROUP)");
            }
          }
          else { /* xchg */
            assert(NULL != atomic_ops && '\0' == *atomic_ops);
            atomic_exp = "atomic_add_global_xchg(A,B)";
          }
        }
        else if (NULL != LIBXSMM_STRISTR(env_atomics, "cmpxchg")) {
          atomic_ops = (c_dbcsr_acc_opencl_atomic_fp_32 == kind ? "-DCMPXCHG=atomic_cmpxchg" : "-DCMPXCHG=atom_cmpxchg");
          atomic_exp = "atomic_add_global_cmpxchg(A,B)";
          exts[ext2] = NULL;
        }
        else { /* xchg */
          atomic_exp = "atomic_add_global_xchg(A,B)";
          atomic_ops = (c_dbcsr_acc_opencl_atomic_fp_32 == kind ? "-DXCHG=atomic_xchg" : "-DXCHG=atom_xchg");
        }
      }
      else { /* unsynchronized */
        atomic_exp = "*(A)+=(B)"; /* non-atomic update */
      }
      assert(NULL != atomic_exp);
      /* compose build parameters and flags */
      result = LIBXSMM_SNPRINTF(flags, flags_maxlen, " -DTAN=%i %s %s -D\"ATOMIC_ADD_GLOBAL(A,B)=%s\" %s", kind, atomic_type,
        atomic_ops, atomic_exp, barrier_expr);
    }
  }
  return result;
}


int c_dbcsr_acc_opencl_defines(const char defines[], char buffer[], size_t buffer_size, int cleanup) {
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = 0;
  if (NULL != buffer && NULL != devinfo->context) {
    const int std_clevel = 100 * devinfo->std_clevel[0] + 10 * devinfo->std_clevel[1];
    const int std_level = 100 * devinfo->std_level[0] + 10 * devinfo->std_level[1];
    result = LIBXSMM_SNPRINTF(buffer, buffer_size, " -DACC_OPENCL_VERSION=%u -DACC_OPENCL_C_VERSION=%u%s", std_level, std_clevel,
      0 == c_dbcsr_acc_opencl_config.debug ? " -DNDEBUG" : "");
    if (0 < result && (int)buffer_size > result) {
      const int n = LIBXSMM_SNPRINTF(
        buffer + result, buffer_size - result, ' ' != buffer[result - 1] ? " %s" : "%s", NULL != defines ? defines : "");
      if (0 <= n) {
        if ((int)buffer_size > (result += n) && 0 != cleanup) {
          char* replace = strpbrk(buffer + result - n, "\""); /* more portable (system/cpp needs quotes to protect braces) */
          for (; NULL != replace; replace = strpbrk(replace + 1, "\"")) *replace = ' ';
        }
      }
      else result = -1;
    }
  }
  else result = -1;
  return result;
}


int c_dbcsr_acc_opencl_kernel_flags(const char build_params[], const char build_options[], const char try_options[],
  cl_program program, char buffer[], size_t buffer_size) {
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = EXIT_SUCCESS, nchar = 0;
  assert(NULL != program && (NULL != buffer || 0 == buffer_size));
  nchar = c_dbcsr_acc_opencl_defines(build_params, buffer, buffer_size, 1 /*cleanup*/);
  if (0 <= nchar && (int)buffer_size > nchar) {
    const int debug = (0 != c_dbcsr_acc_opencl_config.debug && 0 != devinfo->intel && CL_DEVICE_TYPE_CPU != devinfo->type);
    int n = LIBXSMM_SNPRINTF(buffer + nchar, buffer_size - nchar, " %s%s %s", 0 == debug ? "" : "-gline-tables-only ",
      devinfo->std_flag, NULL != build_options ? build_options : "");
    if (0 <= n) {
      nchar += n;
      if (NULL != try_options && '\0' != *try_options) { /* length is not reported in result */
        n = LIBXSMM_SNPRINTF(buffer + nchar, buffer_size - nchar, " %s", try_options);
        if (0 > n || (int)buffer_size <= (nchar + n)) buffer[nchar] = '\0';
      }
    }
    else nchar = n;
  }
  if (0 <= nchar && (int)buffer_size > nchar) { /* check if internal flags apply */
    const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
    result = clBuildProgram(program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
    if (EXIT_SUCCESS != result) { /* failed to apply internal flags */
      ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseProgram(program)); /* avoid unclean state */
      buffer[nchar] = '\0'; /* remove internal flags */
    }
  }
  else result = EXIT_FAILURE;
  return result;
}


int c_dbcsr_acc_opencl_kernel(int source_is_file, const char source[], const char kernel_name[], const char build_params[],
  const char build_options[], const char try_options[], int* try_ok, const char* const extnames[], size_t num_exts,
  cl_kernel* kernel) {
  char buffer[ACC_OPENCL_BUFFERSIZE] = "", buffer_name[ACC_OPENCL_MAXSTRLEN * 2];
  const cl_device_id device_id = c_dbcsr_acc_opencl_config.devices[c_dbcsr_acc_opencl_config.device_id];
  const c_dbcsr_acc_opencl_device_t* const devinfo = &c_dbcsr_acc_opencl_config.device;
  int result = ((NULL != source && NULL != kernel_name && '\0' != *kernel_name) ? EXIT_SUCCESS : EXIT_FAILURE);
  int ok = EXIT_SUCCESS, source_is_cl = 1, nchar = 0;
  size_t size_src = 0, size = 0;
  cl_program program = NULL;
  FILE* file_src = NULL;
  assert(NULL != devinfo->context);
  assert(NULL != kernel);
  *kernel = NULL;
  if (EXIT_SUCCESS == result && 0 != source_is_file) file_src = fopen(source, "rb");
  if (NULL != file_src) {
    if (EXIT_SUCCESS == result) {
      const char* const file_ext = strrchr(source, '.');
      char* src = NULL;
      source_is_cl = ((NULL != file_ext && NULL != LIBXSMM_STRISTR(file_ext + 1, "cl")) ? 1 : 0);
      size_src = (EXIT_SUCCESS == fseek(file_src, 0 /*offset*/, SEEK_END) ? ftell(file_src) : 0);
      src = (char*)((0 != size_src && EXIT_SUCCESS == fseek(file_src, 0 /*offset*/, SEEK_SET))
                      ? libxsmm_aligned_scratch(size_src + source_is_cl /*terminator?*/, 0 /*auto-align*/)
                      : NULL);
      if (NULL != src) {
        if (size_src == fread(src, 1 /*sizeof(char)*/, size_src /*count*/, file_src)) {
          if (0 != source_is_cl) src[size_src] = '\0'; /* terminator */
          source = src;
        }
        else {
          result = EXIT_FAILURE;
          libxsmm_free(src);
        }
      }
      else result = EXIT_FAILURE;
    }
    fclose(file_src);
  }
  if (EXIT_SUCCESS == result && 0 != source_is_cl) {
    const char* ext_source = source;
    size_src = strlen(ext_source);
    if (NULL != extnames) {
      int n = num_exts, nflat = 0;
      size_t size_ext = 0;
      for (; 0 < n; --n) {
        if (NULL != extnames[n - 1]) {
          const char* const end = buffer + strlen(extnames[n - 1]); /* before strtok */
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
              const char* const end = buffer_name + strlen(extnames[num_exts - 1]); /* before strtok */
              char* ext = strtok(
                strncpy(buffer_name, extnames[num_exts - 1], ACC_OPENCL_MAXSTRLEN * 2 - 1), ACC_OPENCL_DELIMS " \t");
              for (; NULL != ext; ext = ((ext + 1) < end ? strtok((ext + 1) + strlen(ext), ACC_OPENCL_DELIMS " \t") : NULL)) {
                const char* line = source;
                for (;;) {
                  if (2 != sscanf(line, "#pragma OPENCL EXTENSION %[^: ]%*[: ]%[^\n]", buffer, buffer + ACC_OPENCL_BUFFERSIZE / 2))
                  {
                    line = NULL;
                    break;
                  }
                  else if (0 == strncmp(buffer, ext, ACC_OPENCL_BUFFERSIZE / 2) &&
                           0 == strncmp(buffer + ACC_OPENCL_BUFFERSIZE / 2, "enable", ACC_OPENCL_BUFFERSIZE / 2))
                  {
                    break;
                  }
                  line = strchr(line, '\n');
                  if (NULL != line) {
                    ++line;
                  }
                  else break;
                }
#  if !defined(NDEBUG)
                if (EXIT_SUCCESS == c_dbcsr_acc_opencl_device_ext(device_id, (const char* const*)&ext, 1))
#  endif
                { /* NDEBUG: assume given extension is supported (confirmed upfront) */
                  if (NULL == line) { /* extension is not already part of source */
                    n += LIBXSMM_SNPRINTF(ext_source_buffer + n, size_src_ext + 1 /*terminator*/ - n, enable_ext, ext);
                  }
                }
#  if !defined(NDEBUG)
                else if (0 != strcmp("cl_intel_global_float_atomics", ext)) {
                  fprintf(stderr, "WARN ACC/OpenCL: extension \"%s\" is not supported.\n", ext);
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
    /* cpp: consider to preprocess kernel (failure does not impact result code) */
    if (0 != c_dbcsr_acc_opencl_config.dump && NULL == file_src) {
      char dump_filename[ACC_OPENCL_MAXSTRLEN];
      nchar = LIBXSMM_SNPRINTF(dump_filename, sizeof(dump_filename), "%s.cl", kernel_name);
      if (0 < nchar && (int)sizeof(dump_filename) > nchar) {
        const int std_flag_len = (int)strlen(devinfo->std_flag);
        const char* const env_cpp = getenv("ACC_OPENCL_CPP");
        const int cpp = (NULL == env_cpp ? 1 /*default*/ : atoi(env_cpp));
#  if defined(ACC_OPENCL_CPPBIN)
        FILE* const file_cpp = (0 != cpp ? fopen(ACC_OPENCL_CPPBIN, "rb") : NULL);
#  else
        FILE* const file_cpp = NULL;
#  endif
        int file_dmp = -1;
        if (NULL != file_cpp) {
          nchar = LIBXSMM_SNPRINTF(buffer_name, sizeof(buffer_name), ACC_OPENCL_TEMPDIR "/.%s.XXXXXX", kernel_name);
          if (0 < nchar && (int)sizeof(buffer_name) > nchar) file_dmp = mkstemp(buffer_name);
          fclose(file_cpp); /* existence-check */
        }
        else file_dmp = open(dump_filename, O_CREAT | O_TRUNC | O_RDWR, S_IREAD | S_IWRITE);
        if (0 <= file_dmp) {
          if ((0 != std_flag_len &&
                (3 != write(file_dmp, "/*\n", 3) || std_flag_len != write(file_dmp, devinfo->std_flag, std_flag_len) ||
                  4 != write(file_dmp, "\n*/\n", 4))) ||
              size_src != (size_t)write(file_dmp, ext_source, size_src))
          {
            file_dmp = -1;
          }
          ACC_OPENCL_EXPECT(EXIT_SUCCESS == close(file_dmp));
        }
#  if defined(ACC_OPENCL_CPPBIN)
        if (NULL != file_cpp && 0 <= file_dmp) { /* preprocess source-code */
          const char* sed_pattern = "";
#    if defined(ACC_OPENCL_SEDBIN)
          FILE* const file_sed = fopen(ACC_OPENCL_SEDBIN, "rb");
          if (NULL != file_sed) {
            sed_pattern = "| " ACC_OPENCL_SEDBIN " '/^[[:space:]]*\\(\\/\\/.*\\)*$/d'";
            fclose(file_sed); /* existence-check */
          }
#    endif
          nchar = LIBXSMM_SNPRINTF(
            buffer, ACC_OPENCL_BUFFERSIZE, ACC_OPENCL_CPPBIN " -P -C -nostdinc %s", 0 == devinfo->nv ? "" : "-D__NV_CL_C_VERSION ");
          if (0 < nchar && ACC_OPENCL_BUFFERSIZE > nchar) {
            int n = c_dbcsr_acc_opencl_defines(build_params, buffer + nchar, ACC_OPENCL_BUFFERSIZE - nchar, 0 /*cleanup*/);
            if (0 <= n && ACC_OPENCL_BUFFERSIZE > (nchar += n)) {
              n = LIBXSMM_SNPRINTF(buffer + nchar, ACC_OPENCL_BUFFERSIZE - nchar,
                ' ' != buffer[nchar - 1] ? " %s %s >%s" : "%s %s >%s", buffer_name, sed_pattern, dump_filename);
            }
            nchar = (0 <= n ? nchar : 0) + n;
          }
          if (0 < nchar && ACC_OPENCL_BUFFERSIZE > nchar && EXIT_SUCCESS == system(buffer)) {
            FILE* const file = fopen(dump_filename, "r");
            if (NULL != file) {
              const long int size_file = (EXIT_SUCCESS == fseek(file, 0 /*offset*/, SEEK_END) ? ftell(file) : 0);
              char* const src = (char*)(EXIT_SUCCESS == fseek(file, 0 /*offset*/, SEEK_SET)
                                          ? libxsmm_aligned_scratch(size_file + 1 /*terminator*/, 0 /*auto-align*/)
                                          : NULL);
              if (NULL != src) {
                if ((size_t)size_file == fread(src, 1 /*sizeof(char)*/, size_file /*count*/, file)) {
                  if (source != ext_source) {
                    void* p = NULL;
                    LIBXSMM_ASSIGN127(&p, &ext_source);
                    libxsmm_free(p);
                  }
                  src[size_file] = '\0';
                  ext_source = src;
                }
                else libxsmm_free(src);
              }
              ACC_OPENCL_EXPECT(EXIT_SUCCESS == fclose(file));
            }
          }
          ACC_OPENCL_EXPECT(EXIT_SUCCESS == unlink(buffer_name)); /* remove temporary file */
          buffer[0] = '\0'; /* reset to empty */
        }
#  endif
      }
    }
    program = clCreateProgramWithSource(devinfo->context, 1 /*nlines*/, &ext_source, NULL, &result);
    assert(EXIT_SUCCESS != result || NULL != program);
    if (EXIT_SUCCESS == result) {
      ok = c_dbcsr_acc_opencl_kernel_flags(build_params, build_options, try_options, program, buffer, ACC_OPENCL_BUFFERSIZE);
      if (EXIT_SUCCESS == ok) result = ok;
      else {
        program = clCreateProgramWithSource(devinfo->context, 1 /*nlines*/, &ext_source, NULL, &result);
        assert(EXIT_SUCCESS != result || NULL != program);
        if (EXIT_SUCCESS == result) {
          result = clBuildProgram(program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
          ok = EXIT_FAILURE;
        }
      }
    }
    if (EXIT_SUCCESS == result) {
      if (source != ext_source) {
        void* p = NULL;
        LIBXSMM_ASSIGN127(&p, &ext_source);
        libxsmm_free(p);
      }
      buffer[0] = '\0'; /* reset to empty */
      if (EXIT_SUCCESS == result) { /* extract kernel */
        *kernel = clCreateKernel(program, kernel_name, &result);
        if (EXIT_SUCCESS == result) {
          assert(NULL != *kernel);
          if (NULL == file_src && (2 <= c_dbcsr_acc_opencl_config.dump || 0 > c_dbcsr_acc_opencl_config.dump)) {
            unsigned char* binary = NULL;
            binary = (unsigned char*)(EXIT_SUCCESS ==
                                          clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL)
                                        ? libxsmm_aligned_scratch(size, 0 /*auto-align*/)
                                        : NULL);
            if (NULL != binary) {
              result = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary, NULL);
              if (EXIT_SUCCESS == result) { /* successfully queried program binary */
                FILE* file;
                nchar = LIBXSMM_SNPRINTF(buffer, ACC_OPENCL_BUFFERSIZE, "%s.dump", kernel_name);
                file = ((0 < nchar && ACC_OPENCL_BUFFERSIZE > nchar) ? fopen(buffer, "wb") : NULL);
                buffer[0] = '\0'; /* reset to empty */
                if (NULL != file) {
                  if (size != fwrite(binary, 1, size, file)) result = EXIT_FAILURE;
                  fclose(file);
                }
                else result = EXIT_FAILURE;
              }
              libxsmm_free(binary);
            }
            else result = EXIT_FAILURE;
          }
        }
      }
    }
    else if (source != ext_source) { /* error: creating program */
      void* p = NULL;
      LIBXSMM_ASSIGN127(&p, &ext_source);
      libxsmm_free(p);
    }
  }
  else if (EXIT_SUCCESS == result) { /* binary representation */
#  if defined(CL_VERSION_2_1)
    if (0 != c_dbcsr_acc_opencl_config.dump) program = clCreateProgramWithIL(devinfo->context, source, size_src, &result);
    else
#  endif
    {
      program = clCreateProgramWithBinary(
        devinfo->context, 1, &device_id, &size_src, (const unsigned char**)&source, NULL /*binary_status*/, &result);
    }
    assert(EXIT_SUCCESS != result || NULL != program);
    if (EXIT_SUCCESS == result) {
      ok = c_dbcsr_acc_opencl_kernel_flags(build_params, build_options, try_options, program, buffer, ACC_OPENCL_BUFFERSIZE);
      if (EXIT_SUCCESS == ok) result = ok;
      else {
#  if defined(CL_VERSION_2_1)
        if (0 != c_dbcsr_acc_opencl_config.dump) program = clCreateProgramWithIL(devinfo->context, source, size_src, &result);
        else
#  endif
        {
          program = clCreateProgramWithBinary(
            devinfo->context, 1, &device_id, &size_src, (const unsigned char**)&source, NULL /*binary_status*/, &result);
        }
        assert(EXIT_SUCCESS != result || NULL != program);
        if (EXIT_SUCCESS == result) {
          result = clBuildProgram(program, 1 /*num_devices*/, &device_id, buffer, NULL /*callback*/, NULL /*user_data*/);
          ok = EXIT_FAILURE;
        }
      }
    }
    if (EXIT_SUCCESS == result) {
      *kernel = clCreateKernel(program, kernel_name, &result);
#  if defined(CL_VERSION_1_2)
      /* error creating kernel: discover available kernels in program, and adopt the last kernel listed */
      if (EXIT_SUCCESS != result &&
          EXIT_SUCCESS == clGetProgramInfo(program, CL_PROGRAM_KERNEL_NAMES, sizeof(char*), buffer, NULL) && '\0' != *buffer)
      {
        const char *const semicolon = strrchr(buffer, ';'), *const name = (NULL == semicolon ? buffer : (semicolon + 1));
        *kernel = clCreateKernel(program, name, &result);
      }
#  endif
      assert(EXIT_SUCCESS != result || NULL != *kernel);
    }
  }
  if (NULL != file_src) {
    void* p = NULL;
    LIBXSMM_ASSIGN127(&p, (const void**)&source);
    assert(0 != source_is_file);
    libxsmm_free(p);
  }
  if (NULL != program) {
    if (EXIT_SUCCESS != result && NULL != *kernel) {
      ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseKernel(*kernel));
      *kernel = NULL;
    }
    if (2 <= c_dbcsr_acc_opencl_config.verbosity || 0 > c_dbcsr_acc_opencl_config.verbosity) {
      if (EXIT_SUCCESS == clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, ACC_OPENCL_BUFFERSIZE, buffer, &size)) {
        const char* info = buffer;
        while ('\0' != *info && NULL != strchr("\n\r\t ", *info)) ++info; /* remove preceding newline etc. */
        assert(NULL != kernel_name && '\0' != *kernel_name);
        if ('\0' != *info) fprintf(stderr, "INFO ACC/OpenCL: %s -> %s\n", kernel_name, info);
      }
      else buffer[0] = '\0'; /* reset to empty */
    }
    ACC_OPENCL_EXPECT(EXIT_SUCCESS == clReleaseProgram(program)); /* release in any case (EXIT_SUCCESS) */
  }
  if (NULL != try_ok) *try_ok = result | ok;
  ACC_OPENCL_RETURN(result, buffer);
}


int c_dbcsr_acc_opencl_set_kernel_ptr(cl_kernel kernel, cl_uint arg_index, const void* arg_value) {
  assert(NULL != c_dbcsr_acc_opencl_config.device.context);
  return (NULL != c_dbcsr_acc_opencl_config.device.clSetKernelArgMemPointerINTEL
            ? c_dbcsr_acc_opencl_config.device.clSetKernelArgMemPointerINTEL(kernel, arg_index, arg_value)
            : clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &arg_value));
}


double c_dbcsr_acc_opencl_duration(cl_event event, int* result_code) {
  cl_ulong begin = 0, end = 0;
  int r = EXIT_FAILURE;
  double result = 0;
  if (NULL != event) {
    r = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &begin, NULL);
    if (EXIT_SUCCESS == r) {
      r = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
      if (EXIT_SUCCESS == r) {
        result = 1E-9 * LIBXSMM_DELTA(begin, end); /* Nanoseconds->seconds */
      }
    }
  }
  if (NULL != result_code) *result_code = r;
  return result;
}


typedef struct c_dbcsr_acc_opencl_hist_t {
  c_dbcsr_acc_opencl_hist_update_fn* update;
  double *vals, min, max;
  int *buckets, nbuckets, nqueue, nvals, n;
} c_dbcsr_acc_opencl_hist_t;


void c_dbcsr_acc_opencl_hist_create(
  void** hist, int nbuckets, int nqueue, int nvals, const c_dbcsr_acc_opencl_hist_update_fn update[]) {
  c_dbcsr_acc_opencl_hist_t* h = (c_dbcsr_acc_opencl_hist_t*)malloc(sizeof(c_dbcsr_acc_opencl_hist_t));
  assert(NULL != hist && 0 < nbuckets && 0 < nqueue && 0 < nvals && NULL != update);
  if (NULL != h) {
    h->vals = (double*)malloc(sizeof(double) * LIBXSMM_MAX(nbuckets, nqueue) * nvals);
    h->update = (c_dbcsr_acc_opencl_hist_update_fn*)malloc(sizeof(c_dbcsr_acc_opencl_hist_update_fn) * nvals);
    h->buckets = (int*)calloc(nbuckets, sizeof(int));
    if (NULL != h->vals && NULL != h->buckets && NULL != h->update) {
      union {
        int raw;
        float value;
      } inf = {0};
#  if defined(INFINITY) && /*overflow warning*/ !defined(_CRAYC)
      inf.value = (float)(INFINITY);
#  else
      inf.raw = 0x7F800000;
#  endif
      h->min = +inf.value;
      h->max = -inf.value;
      h->nbuckets = nbuckets;
      h->nqueue = nqueue;
      h->nvals = nvals;
      /* if update[] is NULL, c_dbcsr_acc_opencl_hist_avg is assumed */
      for (h->n = 0; h->n < nvals; ++h->n) h->update[h->n] = update[h->n];
      h->n = 0;
    }
    else {
      free(h->buckets);
      free(h->vals);
      free(h);
      h = NULL;
    }
  }
  *hist = h;
}


void c_dbcsr_acc_opencl_hist_avg(double* dst, const double* src) {
  assert(NULL != dst && NULL != src);
  *dst = 0.5 * (*dst + *src);
}


void c_dbcsr_acc_opencl_hist_add(double* dst, const double* src) {
  assert(NULL != dst && NULL != src);
  *dst += *src;
}


void c_dbcsr_acc_opencl_hist_set(ACC_OPENCL_LOCKTYPE* lock, void* hist, const double vals[]) {
  if (NULL != hist) {
    c_dbcsr_acc_opencl_hist_t* const h = (c_dbcsr_acc_opencl_hist_t*)hist;
    int i, j, k;
    if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
    if (h->nqueue <= h->n) {
      const double *values, w = h->max - h->min;
      const int* buckets;
      if (h->nqueue == h->n) {
        c_dbcsr_acc_opencl_hist_get(NULL /*lock*/, hist, &buckets, NULL /*nbuckets*/, NULL /*range*/, &values, NULL /*nvals*/);
      }
      for (i = 1; i <= h->nbuckets; ++i) {
        const double q = h->min + i * w / h->nbuckets;
        if (vals[0] <= q || h->nbuckets == i) {
          for (k = 0, j = (i - 1) * h->nvals; k < h->nvals; ++k) {
            if (0 != h->buckets[i - 1]) {
              (NULL != h->update[k] ? h->update[k] : c_dbcsr_acc_opencl_hist_avg)(h->vals + (j + k), vals + k);
            }
            else h->vals[j + k] = vals[k]; /* initialize */
          }
          ++h->buckets[i - 1];
          break;
        }
      }
    }
    else { /* fill-phase */
      if (h->min > vals[0]) h->min = vals[0];
      if (h->max < vals[0]) h->max = vals[0];
      for (k = 0, j = h->nvals * h->n; k < h->nvals; ++k) {
        h->vals[j + k] = vals[k];
      }
    }
    ++h->n; /* count number of accumulated values */
    if (NULL != lock) ACC_OPENCL_RELEASE(lock);
  }
}


void c_dbcsr_acc_opencl_hist_get(
  ACC_OPENCL_LOCKTYPE* lock, void* hist, const int** buckets, int* nbuckets, double range[2], const double** vals, int* nvals) {
  int *b = NULL, m = 0, n = 0, i, j, k;
  double *v = NULL, r[] = {0, 0};
  assert(NULL != buckets || NULL != range || NULL != vals);
  if (NULL != hist) {
    c_dbcsr_acc_opencl_hist_t* const h = (c_dbcsr_acc_opencl_hist_t*)hist;
    if (NULL != lock) ACC_OPENCL_ACQUIRE(lock);
    if (h->n <= h->nqueue) {
      const double w = h->max - h->min;
      if (h->n < h->nbuckets) h->nbuckets = h->n;
      for (i = 1, j = 0; i <= h->nbuckets; j = h->nvals * i++) {
        const double p = h->min + (i - 1) * w / h->nbuckets, q = h->min + i * w / h->nbuckets;
        for (n = 0, m = 0; n < h->n; m = ++n * h->nvals) {
          if (0 == h->buckets[n] && (p < h->vals[m] || 1 == i) && (h->vals[m] <= q || h->nbuckets == i)) {
            if (j != m) {
              if (0 != h->buckets[i - 1]) { /* accumulate */
                for (k = 0; k < h->nvals; ++k) {
                  (NULL != h->update[k] ? h->update[k] : c_dbcsr_acc_opencl_hist_avg)(h->vals + (j + k), h->vals + (m + k));
                }
              }
              else { /* initialize/swap */
                for (k = 0; k < h->nvals; ++k) {
                  const double value = h->vals[m + k];
                  h->vals[m + k] = h->vals[j + k];
                  h->vals[j + k] = value;
                }
              }
            }
            ++h->buckets[i - 1];
          }
        }
      }
      h->nqueue = 0;
    }
    if (0 < h->n) {
      r[0] = h->min;
      r[1] = h->max;
      b = h->buckets;
      n = h->nbuckets;
      v = h->vals;
      m = h->nvals;
    }
    if (NULL != lock) ACC_OPENCL_RELEASE(lock);
  }
  if (NULL != nbuckets) *nbuckets = n;
  if (NULL != buckets) *buckets = b;
  if (NULL != nvals) *nvals = m;
  if (NULL != vals) *vals = v;
  if (NULL != range) {
    range[0] = r[0];
    range[1] = r[1];
  }
}


void c_dbcsr_acc_opencl_hist_print(
  FILE* stream, void* hist, const char title[], const int prec[], const c_dbcsr_acc_opencl_hist_adjust_fn adjust[]) {
  int nbuckets = 0, nvals = 0, i = 1, j = 0, k;
  const int* buckets = NULL;
  const double* vals = NULL;
  double range[2];
  c_dbcsr_acc_opencl_hist_get(NULL /*lock*/, hist, &buckets, &nbuckets, range, &vals, &nvals);
  if (NULL != stream && NULL != buckets && 0 < nbuckets && NULL != vals && 0 < nvals) {
    const double w = range[1] - range[0];
    if (NULL != title) fprintf(stream, "%s pid=%u\n", title, libxsmm_get_pid());
    for (; i <= nbuckets; j = nvals * i++) {
      const double q = range[0] + i * w / nbuckets, r = (i != nbuckets ? q : LIBXSMM_MAX(q, vals[j]));
      const int c = buckets[i - 1];
      if (NULL != prec) fprintf(stream, "\t#%i <= %.*f: %i", i, prec[0], r, c);
      else fprintf(stream, "\t#%i <= %f: %i", i, r, c);
      if (0 != c) {
        fprintf(stream, " ->");
        for (k = 0; k < nvals; ++k) {
          double value;
          if (NULL == adjust || NULL == adjust[k]) value = vals[j + k];
          else value = adjust[k](vals[j + k], c);
          if (NULL != prec) fprintf(stream, " %.*f", prec[k], value);
          else fprintf(stream, " %f", value);
        }
      }
      fprintf(stream, "\n");
    }
  }
}


void c_dbcsr_acc_opencl_hist_free(void* hist) {
  if (NULL != hist) {
    c_dbcsr_acc_opencl_hist_t* const h = (c_dbcsr_acc_opencl_hist_t*)hist;
    free(h->buckets);
    free(h->update);
    free(h->vals);
    free(h);
  }
}

#  if defined(__cplusplus)
}
#  endif

#endif /*__OPENCL*/
