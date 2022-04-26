# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_CSINN)
  find_package(ZLIB REQUIRED)

  set(OPENSSL_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/ssl/)
  set(OPENSSL_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../3rdparty/ssl/include/)
  if (NOT DEFINED OPENSSL_INCLUDE_DIR)
    message(FATAL_ERROR "OPENSSL dependency not found!")
  endif()
  include_directories(${OPENSSL_INCLUDE_DIR})
  # find_package(OpenSSL REQUIRED)
  set(OPENSSL_LIB "${OPENSSL_ROOT_DIR}/lib/libcrypto.a")
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OPENSSL_LIB})
  list(APPEND TVM_LINKER_LIBS ${OPENSSL_LIB})


  file(GLOB CSINN_RELAY_CONTRIB_SRC src/relay/backend/contrib/csinn/*.cc)
  list(APPEND COMPILER_SRCS ${CSINN_RELAY_CONTRIB_SRC})
  file(GLOB CSINN_RUNTIME_SRC src/runtime/hhb/hhb_runtime.cc)
  list(APPEND RUNTIME_SRCS ${CSINN_RUNTIME_SRC})
  file(GLOB CSINN_CONTRIB_SRC src/runtime/contrib/csinn/*.cc)
  #list(APPEND RUNTIME_SRCS ${CSINN_CONTRIB_SRC})
  if(USE_CSINN STREQUAL "ON")
    set(CSINN_PATH "${tvm_SOURCE_DIR}/install_nn2")
    include_directories(${CSINN_PATH}/include)
    set(CSINN_CONTRIB_LIB "${CSINN_PATH}/lib/libcsi_nn2_ref_x86.so")
    message("-- CSI-NN path: ${CSINN_PATH}")
  else()
    include_directories(${USE_CSINN}/include)
    find_library(CSINN_CONTRIB_LIB csi_nn2_nchw_ref_x86 ${USE_CSINN}/lib/)
    message("-- CSI-NN path: ${USE_CSINN}")
  endif()
  if(NOT EXISTS ${CSINN_CONTRIB_LIB})
    message(FATAL_ERROR "can not find CSI-NN lib at path: ${USE_CSINN}/lib/libcsi_nn2_nchw_ref_x86.so")
  endif()
  message("-- CSI-NN path: ${CSINN_CONTRIB_LIB}")
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CSINN_CONTRIB_LIB})
endif(USE_CSINN)
