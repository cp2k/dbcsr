set(SMM_ACC_KERNELS
    kernels/smm_acc_common.h
    kernels/smm_acc_dnt_largeDB1.h
    kernels/smm_acc_dnt_largeDB2.h
    kernels/smm_acc_dnt_medium.h
    kernels/smm_acc_dnt_small.h
    kernels/smm_acc_dnt_tiny.h
    kernels/smm_acc_transpose.h)

if(MULTI_ARCH AND MULTI_GPU_BUILD)
  # Multi-architecture build: create separate targets for each GPU
  foreach(GPU_ARCH IN LISTS WITH_GPU_LIST)
    # Set GPU_PARAMS for this architecture
    set(GPU_PARAMS ${GPU_ARCH})
    
    add_custom_target(
      parameters_${GPU_ARCH} ALL
      COMMAND
        ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py
        --gpu_version=${GPU_ARCH} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}/parameters
      DEPENDS generate_parameters.py parameters/parameters_${GPU_PARAMS}.json
      BYPRODUCTS parameters_${GPU_ARCH}.h
      COMMENT "libsmm_acc: generating parameters for GPU ${GPU_ARCH}")

    add_custom_target(
      smm_acc_kernels_${GPU_ARCH} ALL
      COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py
              ${CMAKE_CURRENT_SOURCE_DIR}/kernels
      DEPENDS generate_kernels.py ${SMM_ACC_KERNELS}
      BYPRODUCTS smm_acc_kernels_${GPU_ARCH}.h
      COMMENT "libsmm_acc: generating kernels for GPU ${GPU_ARCH}")

    # Create the libsmm_acc target for this architecture
    add_custom_target(libsmm_acc_${GPU_ARCH})
    add_dependencies(libsmm_acc_${GPU_ARCH} smm_acc_kernels_${GPU_ARCH} parameters_${GPU_ARCH})

    # Create interface library for this architecture
    add_library(libsmm_acc_interface_${GPU_ARCH} INTERFACE)
    target_include_directories(libsmm_acc_interface_${GPU_ARCH} INTERFACE 
                              ${CMAKE_CURRENT_BINARY_DIR}
                              ${CMAKE_CURRENT_SOURCE_DIR})
    
    message(STATUS "Created libsmm_acc targets for GPU architecture: ${GPU_ARCH}")
  endforeach()

  # Create a convenience target that builds all GPU variants
  add_custom_target(libsmm_acc_all_gpus)
  foreach(GPU_ARCH IN LISTS WITH_GPU_LIST)
    add_dependencies(libsmm_acc_all_gpus libsmm_acc_${GPU_ARCH})
  endforeach()
  
  # For backward compatibility, make dbcsr depend on all libsmm_acc targets
  foreach(GPU_ARCH IN LISTS WITH_GPU_LIST)
    if(TARGET dbcsr_${GPU_ARCH})
      add_dependencies(dbcsr_${GPU_ARCH} libsmm_acc_${GPU_ARCH})
      target_include_directories(dbcsr_${GPU_ARCH} PRIVATE 
                                ${CMAKE_CURRENT_BINARY_DIR}
                                ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
  endforeach()

else()
  # Single architecture build (existing logic)
  add_custom_target(
    parameters ALL
    COMMAND
      ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_parameters.py
      --gpu_version=${WITH_GPU} --base_dir=${CMAKE_CURRENT_SOURCE_DIR}/parameters
    DEPENDS generate_parameters.py parameters/parameters_${WITH_GPU_PARAMS}.json
    BYPRODUCTS parameters.h
    COMMENT "libsmm_acc: generating parameters for GPU ${WITH_GPU_PARAMS}")

  add_custom_target(
    smm_acc_kernels ALL
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernels.py
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    DEPENDS generate_kernels.py ${SMM_ACC_KERNELS}
    BYPRODUCTS smm_acc_kernels.h
    COMMENT "libsmm_acc: generating kernels")

  add_dependencies(dbcsr smm_acc_kernels parameters)
  target_include_directories(dbcsr PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
                                           ${CMAKE_CURRENT_SOURCE_DIR})

  # Note: this library is only used in some of the tests, it's just to get include
  # paths to generated header files.
  add_library(libsmm_acc INTERFACE)
  target_include_directories(libsmm_acc INTERFACE ${CMAKE_CURRENT_BINARY_DIR}
                                                  ${CMAKE_CURRENT_SOURCE_DIR})
endif()